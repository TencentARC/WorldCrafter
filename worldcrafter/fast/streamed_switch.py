"""Overlap a reversible weight transition with the last forward of a stage.

Only completed transformer blocks may change. The next forward waits for the
background stream's completion event. This is serial CFG=1 inference only;
checkpointing, gradients and concurrent requests on one model are unsupported.
"""

import torch


class StreamedSwitchMixin:
    streaming = False

    def wait_pending(self):
        if getattr(self, "_pending", False):
            torch.cuda.current_stream(self.device).wait_event(self._done)
            self._pending = False

    def enable_streaming(self, early, late):
        self.wait_pending()
        for handle in getattr(self, "_stream_handles", []):
            handle.remove()
        self._stream_handles = []
        if not hasattr(self, "_group_graphs"):
            groups = {}
            for p in self.plans:
                if not p.bits:
                    continue
                group = (
                    ".".join(p.name.split(".")[:2])
                    if p.name.startswith("blocks.")
                    else "global"
                )
                groups.setdefault(group, []).append(p)
            self._group_graphs = {}
            self._events = {}
            self._stream = torch.cuda.Stream(device=self.device)
            for group, plans in groups.items():
                self._events[group] = torch.cuda.Event()
                for sign in [1, -1]:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=self._stream):
                        for p in plans:
                            p.apply(sign)
                    self._group_graphs[(group, sign)] = graph
            torch.cuda.synchronize(self.device)
            self._done = torch.cuda.Event()
            self._finish_event = torch.cuda.Event()
        self._target = None
        self._pending = False
        self.streaming = True
        for model in [early, late]:
            for i, block in enumerate(model.blocks):
                group = f"blocks.{i}"

                def hook(module, inputs, output, group=group):
                    if self._target is not None:
                        self._enqueue_group(group)

                self._stream_handles.append(block.register_forward_hook(hook))

            def finish(module, inputs, output):
                if self._target is None:
                    return
                sign = 1 if self._target == "old" else -1
                self._finish_event.record(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(self._stream):
                    self._stream.wait_event(self._finish_event)
                    graph = self._group_graphs.get(("global", sign))
                    if graph is not None:
                        graph.replay()
                    self._done.record(self._stream)
                self.active = self._target
                self._target = None
                self._pending = True
                self.switches += 1

            self._stream_handles.append(model.register_forward_hook(finish))
        self.report["streamed_switch"] = True

    def arm(self, target):
        if target is None:
            return
        if not self.streaming:
            raise RuntimeError("Streaming is not enabled")
        if torch.is_grad_enabled():
            raise RuntimeError("Streamed switching requires inference/no-grad mode")
        if self._target is not None or self._pending:
            raise RuntimeError("Previous transition not consumed")
        if target == self.active:
            raise RuntimeError("Transition must change branch")
        self._target = target

    def _enqueue_group(self, group):
        sign = 1 if self._target == "old" else -1
        graph = self._group_graphs.get((group, sign))
        if graph is None:
            return
        event = self._events[group]
        event.record(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._stream):
            self._stream.wait_event(event)
            graph.replay()
