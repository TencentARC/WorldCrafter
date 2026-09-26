"""Two GPU workers coordinated through an idle-safe CPU control pipe."""

from datetime import timedelta
import json
import multiprocessing as mp
import tempfile
import traceback

from .config import DATA
from .engine import ChunkCancelled, WorldCrafterEngine

WORKER_TIMEOUT = 300


def initialize_group(rank, rendezvous):
    import torch
    import torch.distributed as dist
    from worldcrafter.fast.parallel import install_query_parallel

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
        timeout=timedelta(seconds=WORKER_TIMEOUT),
    )
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=WORKER_TIMEOUT))
    install_query_parallel()
    return control


class SynchronizedCancel:
    def __init__(self, control, event=None):
        self.control, self.event = control, event

    def is_set(self):
        import torch.distributed as dist
        value = [bool(self.event and self.event.is_set())]
        dist.broadcast_object_list(value, src=0, group=self.control)
        return value[0]


def peer_main(connection, rendezvous):
    import torch.distributed as dist
    engine = None
    try:
        control = initialize_group(1, rendezvous)
        engine = WorldCrafterEngine(device="cuda:1")
        connection.send(("ready", engine.info))
        while True:
            command, args = connection.recv()
            try:
                if command == "shutdown":
                    break
                if command == "initialize":
                    engine.initialize_session(*args, cancel=SynchronizedCancel(control))
                    result = None
                elif command == "chunk":
                    result = engine.next_chunk(*args)
                    result = {key: result[key] for key in ("memory", "latent_sha256", "rgb_sha256", "rng_sha256")}
                elif command == "close":
                    engine.close_session()
                    result = None
                else:
                    raise ValueError(f"Unknown worker command: {command}")
                connection.send(("ok", result))
            except ChunkCancelled:
                connection.send(("cancelled", None))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        if engine is not None:
            engine.close_session()
        if dist.is_initialized():
            dist.destroy_process_group()
        connection.close()


class ParallelEngine:
    def __init__(self):
        import torch
        if torch.cuda.device_count() < 2:
            raise RuntimeError("Two visible GPUs are required for --devices 0,1")
        self._directory = tempfile.TemporaryDirectory(prefix="worldcrafter-parallel-")
        rendezvous = self._directory.name + "/rendezvous"
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.peer = context.Process(target=peer_main, args=(child, rendezvous))
        self.peer.start()
        child.close()
        self.engine = None
        try:
            self.control = initialize_group(0, rendezvous)
            self.engine = WorldCrafterEngine()
            self.receive(expected="ready")
            self.info = dict(self.engine.info, gpu_mode="query_parallel", gpu_count=2)
            (DATA / "reports" / "model_setup.json").write_text(
                json.dumps(self.info, indent=2) + "\n"
            )
        except BaseException:
            self.shutdown()
            raise
        self.on_step_completed = None

    def receive(self, expected="ok"):
        if not self.connection.poll(WORKER_TIMEOUT):
            raise RuntimeError("GPU worker response timed out")
        status, result = self.connection.recv()
        if status == "cancelled":
            raise ChunkCancelled()
        if status != expected:
            raise RuntimeError(f"GPU worker failed: {result}")
        return result

    def initialize_session(self, image, prompt, seed, max_chunks, cancel):
        args = (image, prompt, seed, max_chunks)
        self.connection.send(("initialize", args))
        self.engine.initialize_session(*args, cancel=SynchronizedCancel(self.control, cancel))
        self.receive()

    def next_chunk(self, action, poses=None, capture=False):
        self.connection.send(("chunk", (action, poses, False)))
        self.engine.on_step_completed = self.on_step_completed
        result = self.engine.next_chunk(action, poses, capture)
        peer = self.receive()
        for key in ("latent_sha256", "rgb_sha256", "rng_sha256"):
            if result[key] != peer[key]:
                raise RuntimeError(f"GPU workers diverged: {key}")
        devices = result["memory"]["devices"] + peer["memory"]["devices"]
        result["memory"] = dict(devices=devices, **{
            key: sum(d[key] for d in devices)
            for key in ("allocated", "reserved", "peak_allocated", "peak_reserved")
        })
        return result

    def close_session(self):
        self.connection.send(("close", ()))
        self.engine.close_session()
        # Drain cancellation acknowledgement before the close acknowledgement.
        try:
            self.receive()
        except ChunkCancelled:
            self.receive()

    def shutdown(self):
        import torch.distributed as dist
        if self.peer.is_alive():
            try:
                self.connection.send(("shutdown", ()))
            except (BrokenPipeError, EOFError):
                pass
        if self.engine is not None:
            self.engine.close_session()
        if dist.is_initialized():
            dist.destroy_process_group()
        self.peer.join(timeout=30)
        if self.peer.is_alive():
            self.peer.terminate()
            self.peer.join(timeout=10)
        self.connection.close()
        self._directory.cleanup()
