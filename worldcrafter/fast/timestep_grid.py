"""Pyramid schedules and authenticated timestep tables for fast inference.

DMD inference consumes the serialized table bundled with each checkpoint.
Provenance fields are preserved verbatim because they participate in contract
validation and fingerprints; archived URIs are metadata, not download paths.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch

from .schedule_math import apply_schedule_shift, resolve_shift_mu

# The supported transformer patch size is (1, 2, 2), giving
# T * H * W / 4 tokens for a T x H x W latent grid.
_PATCH_ELEMS = 4

DMD_TIMESTEP_CONTRACT_FILENAME = "dmd_timestep_contract.json"
DMD_TIMESTEP_CONTRACT_SCHEMA_VERSION = 3
# These identity fields are part of the serialized checkpoint fingerprint.
_TEACHER_CONDITIONING = {
    "sigma_lookup": "bf16_condition_nearest_full_fp32_grid_first_index",
    "teacher_checkpoint_uri": (
        "s3://arcwm-code-us-west-2/karmyu/Helios_memory_camera_3drepresentation_clean/"
        "good_ckpt/natural32gpu_ospv1_dl3dv_mind111_checkpoint-1000"
    ),
    "training_config_sha256": "c8ac629cb438bc65552deed73e36442ac1b03e6acb2f4653e042e47be2f27e62",
    "training_log_sha256": "ba895b06d86e20d479a2ef9f820e6ce6c662d95e0628974d3beeabf117ad9be1",
}
NATURAL8000_TEACHER_PROVENANCE_SHA256 = (
    "267da9c65e3b92f2e4122bdd36b3a9029628acee7622193268a640ee36092d8d"
)
DEFAULT_TEACHER_PROVENANCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "accelerate_configs"
    / "natural8000_teacher_scheduler_provenance.json"
)

_LOCKED_LATENT_SHAPE = (16, 9, 48, 80)
_LOCKED_ROLLOUT_STEPS = (2, 2, 2)
_LOCKED_NORMAL_TIMESTEPS = (
    (998.5342, 833.9636),
    (742.8216, 547.1926),
    (385.4137, 253.9905),
)
_LOCKED_EMPTY_HISTORY_TIMESTEPS = (
    (998.5342, 902.2183, 833.9636, 783.0660),
    (742.8216, 640.0038, 547.1926, 462.9951),
    (385.4137, 328.6249, 253.9905, 151.5308),
)


def pyramid_image_seq_lens(
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_stages: int,
) -> list[int]:
    """Token count per pyramid stage, coarsest first.

    Stage ``i`` runs at half the resolution of stage ``i + 1``, so the finest stage
    is the full latent grid and the rest are powers of two below it.
    """
    lens = []
    for i_s in range(num_stages):
        scale = 2 ** (num_stages - 1 - i_s)
        lens.append(
            (latent_width // scale)
            * (latent_height // scale)
            * num_latent_frames
            // _PATCH_ELEMS
        )
    return lens


def student_stage_grid(
    scheduler,
    stage_index: int,
    num_inference_steps: int,
    image_seq_len: int,
    use_dynamic_shifting: bool = True,
    time_shift_type: Literal["exponential", "linear"] = "linear",
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """The ``num_inference_steps`` noise levels a distilled sampler visits at one stage.

    Construct one more point than the step budget, drop the trailing point,
    apply the configured shift, and rescale into the stage's timestep band.
    """
    stage_sigmas = scheduler.sigmas_per_stage[stage_index]
    band = scheduler.timesteps_per_stage[stage_index]

    sigmas = torch.linspace(
        stage_sigmas[0].item(),
        stage_sigmas[-1].item(),
        num_inference_steps + 1,
        dtype=torch.float64,
    )
    if not use_dynamic_shifting:
        timesteps = torch.linspace(
            band[0].item(),
            band[-1].item(),
            num_inference_steps + 1,
            dtype=torch.float64,
        )[:-1]
        return timesteps.to(device=device, dtype=dtype)

    # A distilled sampler carries the band's end alongside its steps, so the
    # trailing linspace point is replaced by a zero sigma before shifting.
    sigmas = torch.cat([sigmas[:-1], torch.zeros(1, dtype=torch.float64)])
    mu = resolve_shift_mu(
        image_seq_len,
        base_seq_len=base_seq_len,
        max_seq_len=max_seq_len,
        base_shift=base_shift,
        max_shift=max_shift,
        time_shift_type=time_shift_type,
    )
    sigmas = apply_schedule_shift(sigmas, None, mu=mu)
    low = band.min().to(torch.float64)
    timesteps = low + sigmas[:-1] * (band.max().to(torch.float64) - low)
    return timesteps.to(device=device, dtype=dtype)


def student_timestep_table(
    scheduler,
    num_inference_steps_list: Sequence[int],
    image_seq_lens: Sequence[int],
    **kwargs,
) -> list[torch.Tensor]:
    """``student_stage_grid`` for every pyramid stage, coarsest first."""
    if len(num_inference_steps_list) != len(image_seq_lens):
        raise ValueError(
            f"Got {len(num_inference_steps_list)} step counts for {len(image_seq_lens)} pyramid stages; "
            "they are matched by position."
        )
    return [
        student_stage_grid(scheduler, i_s, steps, image_seq_lens[i_s], **kwargs)
        for i_s, steps in enumerate(num_inference_steps_list)
    ]


def resolve_dmd_contract_path(checkpoint_path: str | Path) -> Path:
    """Return the canonical sidecar location for a checkpoint/run directory."""

    path = Path(checkpoint_path)
    if path.name == DMD_TIMESTEP_CONTRACT_FILENAME:
        return path
    if path.is_dir():
        return path / DMD_TIMESTEP_CONTRACT_FILENAME
    if path.suffix:
        return path.parent / DMD_TIMESTEP_CONTRACT_FILENAME
    return path / DMD_TIMESTEP_CONTRACT_FILENAME


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], name: str
) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"Invalid {name} keys: missing={missing}, unexpected={unexpected}"
        )


@dataclass(frozen=True)
class DmdTimestepStep:
    """One student query and the flow interpolation coefficients around it."""

    model_timestep: float
    current_sigma: float
    next_sigma: float

    def to_dict(self) -> dict[str, float]:
        return {
            "model_timestep": self.model_timestep,
            "current_sigma": self.current_sigma,
            "next_sigma": self.next_sigma,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DmdTimestepStep":
        _require_exact_keys(
            payload, {"model_timestep", "current_sigma", "next_sigma"}, "DMD step"
        )
        return cls(
            model_timestep=float(payload["model_timestep"]),
            current_sigma=float(payload["current_sigma"]),
            next_sigma=float(payload["next_sigma"]),
        )


@dataclass(frozen=True)
class DmdStudentSchedulerProvenance:
    """The fixed stage/band math from which the serialized student table came."""

    num_train_timesteps: int
    stages: int
    stage_range: tuple[float, ...]
    gamma: float
    shift: float
    version: str
    use_dynamic_shifting: bool = True
    time_shift_type: str = "linear"
    base_seq_len: int = 256
    max_seq_len: int = 4096
    base_shift: float = 0.5
    max_shift: float = 1.15

    @classmethod
    def from_scheduler(cls, scheduler: Any) -> "DmdStudentSchedulerProvenance":
        config = scheduler.config
        provenance = cls(
            num_train_timesteps=int(_config_value(config, "num_train_timesteps", 1000)),
            stages=int(
                _config_value(config, "stages", len(scheduler.timesteps_per_stage))
            ),
            stage_range=tuple(
                float(value) for value in _config_value(config, "stage_range")
            ),
            gamma=float(_config_value(config, "gamma")),
            shift=float(_config_value(config, "shift")),
            version=str(
                getattr(scheduler, "version", _config_value(config, "version", "v1"))
            ),
        )
        provenance.validate()
        return provenance

    def validate(self) -> None:
        if self.num_train_timesteps != 1000:
            raise ValueError(
                f"Direct DMD requires 1000 student training timesteps, got {self.num_train_timesteps}"
            )
        if self.stages != 3:
            raise ValueError(
                f"Direct DMD requires three pyramid stages, got {self.stages}"
            )
        expected_range = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
        if len(self.stage_range) != len(expected_range) or any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(self.stage_range, expected_range)
        ):
            raise ValueError(f"Unsupported DMD stage range: {self.stage_range}")
        if not math.isclose(self.gamma, 1.0 / 3.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Unsupported DMD stage gamma: {self.gamma}")
        if not math.isclose(self.shift, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Unsupported DMD stage shift: {self.shift}")
        if self.version != "v1":
            raise ValueError(f"Unsupported DMD stage scheduler version: {self.version}")
        if not self.use_dynamic_shifting or self.time_shift_type != "linear":
            raise ValueError(
                "Direct DMD student schedules require linear resolution-dependent shifting"
            )
        if (self.base_seq_len, self.max_seq_len) != (256, 4096):
            raise ValueError("Direct DMD student shift sequence-length anchors changed")
        if not math.isclose(self.base_shift, 0.5) or not math.isclose(
            self.max_shift, 1.15
        ):
            raise ValueError("Direct DMD student shift endpoints changed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_train_timesteps": self.num_train_timesteps,
            "stages": self.stages,
            "stage_range": list(self.stage_range),
            "gamma": self.gamma,
            "shift": self.shift,
            "version": self.version,
            "use_dynamic_shifting": self.use_dynamic_shifting,
            "time_shift_type": self.time_shift_type,
            "base_seq_len": self.base_seq_len,
            "max_seq_len": self.max_seq_len,
            "base_shift": self.base_shift,
            "max_shift": self.max_shift,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DmdStudentSchedulerProvenance":
        expected = {
            "num_train_timesteps",
            "stages",
            "stage_range",
            "gamma",
            "shift",
            "version",
            "use_dynamic_shifting",
            "time_shift_type",
            "base_seq_len",
            "max_seq_len",
            "base_shift",
            "max_shift",
        }
        _require_exact_keys(payload, expected, "student scheduler provenance")
        provenance = cls(
            num_train_timesteps=int(payload["num_train_timesteps"]),
            stages=int(payload["stages"]),
            stage_range=tuple(float(value) for value in payload["stage_range"]),
            gamma=float(payload["gamma"]),
            shift=float(payload["shift"]),
            version=str(payload["version"]),
            use_dynamic_shifting=bool(payload["use_dynamic_shifting"]),
            time_shift_type=str(payload["time_shift_type"]),
            base_seq_len=int(payload["base_seq_len"]),
            max_seq_len=int(payload["max_seq_len"]),
            base_shift=float(payload["base_shift"]),
            max_shift=float(payload["max_shift"]),
        )
        provenance.validate()
        return provenance


@dataclass(frozen=True)
class DmdTeacherSchedulerProvenance:
    """Archived scheduler identity required to authenticate checkpoint metadata."""

    provenance_path: str
    provenance_sha256: str
    class_name: str
    num_train_timesteps: int
    shift: float
    sigma_min: float
    extra_one_step: bool
    set_timesteps_training: bool
    grid_dtype: str
    condition_dtype: str
    use_dynamic_shifting: bool
    source_code_package_uri: str
    source_config_uri: str
    source_config_sha256: str
    source_log_uri: str
    source_log_sha256: str
    source_train_sha256: str
    source_scheduler_sha256: str

    @classmethod
    def from_path(
        cls, path: str | Path = DEFAULT_TEACHER_PROVENANCE_PATH
    ) -> "DmdTeacherSchedulerProvenance":
        path = Path(path)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != NATURAL8000_TEACHER_PROVENANCE_SHA256:
            raise ValueError(
                f"Natural8000 teacher provenance SHA256 mismatch for {path}: expected "
                f"{NATURAL8000_TEACHER_PROVENANCE_SHA256}, got {digest}"
            )
        config = json.loads(raw)
        repo_root = Path(__file__).resolve().parents[2]
        try:
            provenance_path = path.resolve().relative_to(repo_root).as_posix()
        except ValueError:
            provenance_path = path.as_posix()
        provenance = cls(
            provenance_path=provenance_path,
            provenance_sha256=digest,
            class_name=str(config.get("_class_name")),
            num_train_timesteps=int(config.get("num_train_timesteps", -1)),
            shift=float(config.get("shift", float("nan"))),
            sigma_min=float(config.get("sigma_min", float("nan"))),
            extra_one_step=bool(config.get("extra_one_step", False)),
            set_timesteps_training=bool(config.get("set_timesteps_training", False)),
            grid_dtype=str(config.get("grid_dtype")),
            condition_dtype=str(config.get("condition_dtype")),
            use_dynamic_shifting=bool(config.get("use_dynamic_shifting", True)),
            source_code_package_uri=str(config.get("source_code_package_uri")),
            source_config_uri=str(config.get("source_config_uri")),
            source_config_sha256=str(config.get("source_config_sha256")),
            source_log_uri=str(config.get("source_log_uri")),
            source_log_sha256=str(config.get("source_log_sha256")),
            source_train_sha256=str(config.get("source_train_sha256")),
            source_scheduler_sha256=str(config.get("source_scheduler_sha256")),
        )
        provenance.validate()
        return provenance

    def validate(self) -> None:
        if self.provenance_sha256 != NATURAL8000_TEACHER_PROVENANCE_SHA256:
            raise ValueError(
                f"Unrecognized natural8000 teacher provenance fingerprint: {self.provenance_sha256}; "
                f"expected {NATURAL8000_TEACHER_PROVENANCE_SHA256}"
            )
        expected_provenance_path = (
            "scripts/accelerate_configs/natural8000_teacher_scheduler_provenance.json"
        )
        if self.provenance_path != expected_provenance_path:
            raise ValueError(
                f"Natural8000 provenance path mismatch: expected {expected_provenance_path}, "
                f"got {self.provenance_path}"
            )
        if self.class_name != "helios.scheduler.flow_match_ucpe.FlowMatchScheduler":
            raise ValueError(f"Unexpected teacher scheduler class: {self.class_name}")
        if self.num_train_timesteps != 1000:
            raise ValueError(
                f"Teacher re-noise grid must contain 1000 points, got {self.num_train_timesteps}"
            )
        if not math.isclose(self.shift, 5.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(
                f"Natural8000 teacher re-noise shift must be 5.0, got {self.shift}"
            )
        if not math.isclose(self.sigma_min, 0.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(
                f"Natural8000 teacher sigma_min must be 0.0, got {self.sigma_min}"
            )
        if not self.extra_one_step or not self.set_timesteps_training:
            raise ValueError(
                "Natural8000 teacher requires extra_one_step=True and set_timesteps(..., training=True)"
            )
        if self.grid_dtype != "float32" or self.condition_dtype != "bfloat16":
            raise ValueError(
                "Natural8000 teacher requires an FP32 grid and BF16 model condition"
            )
        if self.use_dynamic_shifting:
            raise ValueError(
                "Teacher re-noise must not use resolution-dependent dynamic shifting"
            )
        expected_hashes = {
            "source_config_sha256": "68dd8e439709c41b5c45ca0f7dd291955dc125cefc5e3a0da8c407fe755fc23b",
            "source_log_sha256": "0ff031b75a29d0e991fd0221725cb34ecdef4b62c4a210dd4ad553dd9ed44ac8",
            "source_train_sha256": "d1eff87d68304f0f3ff76f0260e1df42f9902cce20a60f03fd583c3412946cc4",
            "source_scheduler_sha256": "052d3983d7774eb9d9a9cbca3bb76b38d75fb2f028c382b9f6acd29def50e5f8",
        }
        for field_name, expected in expected_hashes.items():
            actual = getattr(self, field_name)
            if actual != expected:
                raise ValueError(
                    f"Natural8000 provenance {field_name} mismatch: expected {expected}, got {actual}"
                )
        expected_uris = {
            "source_code_package_uri": (
                "s3://arcwm-code-us-west-2/karmyu/code_packages/"
                "Helios_memory_camera_3drepresentation_v1_ospv1_dl3dv_natural_32gpu_20260809_v1_slim"
            ),
            "source_config_uri": (
                "s3://arcwm-code-us-west-2/karmyu/Helios_memory_camera_3drepresentation_v1/experiments/"
                "stage_1_frozen_lagernvs5000_ucpe5000_dit_lora_32gpu_ospv1_dl3dv_natural_20260809/"
                "configs/config.resolved.yaml"
            ),
            "source_log_uri": (
                "s3://arcwm-code-us-west-2/karmyu/Helios_memory_camera_3drepresentation_v1/experiments/"
                "stage_1_frozen_lagernvs5000_ucpe5000_dit_lora_32gpu_ospv1_dl3dv_natural_20260809/"
                "logs/node-0/train.log"
            ),
        }
        for field_name, expected in expected_uris.items():
            actual = getattr(self, field_name)
            if actual != expected:
                raise ValueError(
                    f"Natural8000 provenance {field_name} mismatch: expected {expected}, got {actual}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance_path": self.provenance_path,
            "provenance_sha256": self.provenance_sha256,
            "class_name": self.class_name,
            "num_train_timesteps": self.num_train_timesteps,
            "shift": self.shift,
            "sigma_min": self.sigma_min,
            "extra_one_step": self.extra_one_step,
            "set_timesteps_training": self.set_timesteps_training,
            "grid_dtype": self.grid_dtype,
            "condition_dtype": self.condition_dtype,
            "use_dynamic_shifting": self.use_dynamic_shifting,
            "source_code_package_uri": self.source_code_package_uri,
            "source_config_uri": self.source_config_uri,
            "source_config_sha256": self.source_config_sha256,
            "source_log_uri": self.source_log_uri,
            "source_log_sha256": self.source_log_sha256,
            "source_train_sha256": self.source_train_sha256,
            "source_scheduler_sha256": self.source_scheduler_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DmdTeacherSchedulerProvenance":
        expected = {
            "provenance_path",
            "provenance_sha256",
            "class_name",
            "num_train_timesteps",
            "shift",
            "sigma_min",
            "extra_one_step",
            "set_timesteps_training",
            "grid_dtype",
            "condition_dtype",
            "use_dynamic_shifting",
            "source_code_package_uri",
            "source_config_uri",
            "source_config_sha256",
            "source_log_uri",
            "source_log_sha256",
            "source_train_sha256",
            "source_scheduler_sha256",
        }
        _require_exact_keys(payload, expected, "teacher scheduler provenance")
        provenance = cls(
            provenance_path=str(payload["provenance_path"]),
            provenance_sha256=str(payload["provenance_sha256"]),
            class_name=str(payload["class_name"]),
            num_train_timesteps=int(payload["num_train_timesteps"]),
            shift=float(payload["shift"]),
            sigma_min=float(payload["sigma_min"]),
            extra_one_step=bool(payload["extra_one_step"]),
            set_timesteps_training=bool(payload["set_timesteps_training"]),
            grid_dtype=str(payload["grid_dtype"]),
            condition_dtype=str(payload["condition_dtype"]),
            use_dynamic_shifting=bool(payload["use_dynamic_shifting"]),
            source_code_package_uri=str(payload["source_code_package_uri"]),
            source_config_uri=str(payload["source_config_uri"]),
            source_config_sha256=str(payload["source_config_sha256"]),
            source_log_uri=str(payload["source_log_uri"]),
            source_log_sha256=str(payload["source_log_sha256"]),
            source_train_sha256=str(payload["source_train_sha256"]),
            source_scheduler_sha256=str(payload["source_scheduler_sha256"]),
        )
        provenance.validate()
        return provenance


@dataclass(frozen=True)
class DmdTeacherGridPoint:
    grid_index: int
    timestep: float
    sigma: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "grid_index": self.grid_index,
            "timestep": self.timestep,
            "sigma": self.sigma,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DmdTeacherGridPoint":
        _require_exact_keys(
            payload, {"grid_index", "timestep", "sigma"}, "teacher grid point"
        )
        return cls(
            grid_index=int(payload["grid_index"]),
            timestep=float(payload["timestep"]),
            sigma=float(payload["sigma"]),
        )


@dataclass(frozen=True)
class TeacherTimestepSample:
    """Raw sampled timestep/index and the sigma resolved from BF16 condition.

    ``sigma_indices`` can differ from ``indices`` because the original teacher
    rounds its condition to BF16 before looking up the nearest full-grid point.
    """

    timestep: torch.Tensor
    sigma: torch.Tensor
    condition: torch.Tensor
    indices: torch.Tensor
    sigma_indices: torch.Tensor


def _teacher_grid(
    provenance: DmdTeacherSchedulerProvenance,
) -> tuple[DmdTeacherGridPoint, ...]:
    """Build the dense flow grid in FP32, with no resolution-dependent shift."""

    count = provenance.num_train_timesteps
    # Match the byte-exact natural8000 source package: its UCPE flow helper used
    # the default FP32 torch.linspace(1, 0, 1001)[:-1], then performed the
    # shift=5 rational transform and timestep multiplication in FP32.
    base_sigmas = torch.linspace(1.0, 0.0, count + 1, dtype=torch.float32)[:-1]
    shifted_sigmas = (
        provenance.shift * base_sigmas / (1.0 + (provenance.shift - 1.0) * base_sigmas)
    )
    timesteps = shifted_sigmas * float(count)
    return tuple(
        DmdTeacherGridPoint(
            grid_index=index, timestep=float(timestep), sigma=float(sigma)
        )
        for index, (timestep, sigma) in enumerate(
            zip(timesteps.tolist(), shifted_sigmas.tolist())
        )
    )


def _student_stage_steps(
    scheduler: Any,
    stage_index: int,
    num_steps: int,
    image_seq_len: int,
) -> tuple[DmdTimestepStep, ...]:
    if num_steps <= 0:
        raise ValueError(
            f"DMD rollout stage {stage_index} needs at least one step, got {num_steps}"
        )
    stage_sigmas = scheduler.sigmas_per_stage[stage_index]
    band = scheduler.timesteps_per_stage[stage_index]
    sigmas = torch.linspace(
        stage_sigmas[0].item(),
        stage_sigmas[-1].item(),
        num_steps + 1,
        dtype=torch.float64,
    )
    # The final transition is clean x0, not the stage grid's small non-zero tail.
    sigmas[-1] = 0.0
    mu = resolve_shift_mu(image_seq_len, time_shift_type="linear")
    shifted_sigmas = apply_schedule_shift(sigmas, None, mu=mu).to(torch.float32)
    low = band.min().to(torch.float64)
    model_timesteps = (
        low
        + shifted_sigmas[:-1].to(torch.float64) * (band.max().to(torch.float64) - low)
    ).to(torch.float32)
    return tuple(
        DmdTimestepStep(
            model_timestep=float(model_timesteps[index]),
            current_sigma=float(shifted_sigmas[index]),
            next_sigma=float(shifted_sigmas[index + 1]),
        )
        for index in range(num_steps)
    )


@dataclass(frozen=True)
class DmdTimestepContract:
    """Serialized DMD inference schedule with authenticated provenance.

    Rollout timesteps and archived scheduler grids occupy separate fields.
    Inference selects the normal or empty-history rollout table.
    """

    schema_version: int
    latent_shape: tuple[int, int, int, int]
    rollout_steps_per_stage: tuple[int, int, int]
    amplify_empty_history: bool
    normal_stages: tuple[tuple[DmdTimestepStep, ...], ...]
    empty_history_stages: tuple[tuple[DmdTimestepStep, ...], ...]
    student_scheduler: DmdStudentSchedulerProvenance
    teacher_scheduler: DmdTeacherSchedulerProvenance
    teacher_grid: tuple[DmdTeacherGridPoint, ...]
    teacher_valid_timestep_min: float = 20.0
    teacher_valid_timestep_max: float = 980.0

    def __post_init__(self) -> None:
        self._validate()

    @classmethod
    def build(
        cls,
        scheduler: Any,
        *,
        latent_shape: Sequence[int] = _LOCKED_LATENT_SHAPE,
        rollout_steps_per_stage: Sequence[int] = _LOCKED_ROLLOUT_STEPS,
        amplify_empty_history: bool = True,
        teacher_provenance_path: str | Path = DEFAULT_TEACHER_PROVENANCE_PATH,
    ) -> "DmdTimestepContract":
        latent_shape = tuple(int(value) for value in latent_shape)
        if len(latent_shape) != 4:
            raise ValueError(
                f"DMD latent_shape must be (channels, frames, height, width), got {latent_shape}"
            )
        rollout_steps = tuple(int(value) for value in rollout_steps_per_stage)
        if rollout_steps != _LOCKED_ROLLOUT_STEPS:
            raise ValueError(
                f"This DMD contract is distilled at {_LOCKED_ROLLOUT_STEPS} steps per stage; got {rollout_steps}"
            )
        student_provenance = DmdStudentSchedulerProvenance.from_scheduler(scheduler)
        image_seq_lens = pyramid_image_seq_lens(
            latent_shape[1], latent_shape[2], latent_shape[3], 3
        )
        normal = tuple(
            _student_stage_steps(
                scheduler,
                stage_index,
                rollout_steps[stage_index],
                image_seq_lens[stage_index],
            )
            for stage_index in range(3)
        )
        if amplify_empty_history:
            empty_history = tuple(
                _student_stage_steps(
                    scheduler,
                    stage_index,
                    rollout_steps[stage_index] * 2,
                    image_seq_lens[stage_index],
                )
                for stage_index in range(3)
            )
        else:
            empty_history = normal
        teacher_provenance = DmdTeacherSchedulerProvenance.from_path(
            teacher_provenance_path
        )
        return cls(
            schema_version=DMD_TIMESTEP_CONTRACT_SCHEMA_VERSION,
            latent_shape=latent_shape,
            rollout_steps_per_stage=rollout_steps,
            amplify_empty_history=bool(amplify_empty_history),
            normal_stages=normal,
            empty_history_stages=empty_history,
            student_scheduler=student_provenance,
            teacher_scheduler=teacher_provenance,
            teacher_grid=_teacher_grid(teacher_provenance),
        )

    @classmethod
    def create(cls, *args, **kwargs) -> "DmdTimestepContract":
        """Alias for :meth:`build` for callers that use factory terminology."""

        return cls.build(*args, **kwargs)

    def _validate(self) -> None:
        # Both schema versions support authenticated checkpoint inference.
        if self.schema_version not in (2, DMD_TIMESTEP_CONTRACT_SCHEMA_VERSION):
            raise ValueError(
                f"Unsupported DMD timestep contract schema {self.schema_version}; "
                f"expected 2 or {DMD_TIMESTEP_CONTRACT_SCHEMA_VERSION}"
            )
        if len(self.latent_shape) != 4 or any(
            value <= 0 for value in self.latent_shape
        ):
            raise ValueError(f"Invalid DMD latent shape: {self.latent_shape}")
        if self.rollout_steps_per_stage != _LOCKED_ROLLOUT_STEPS:
            raise ValueError(
                f"Unsupported DMD rollout step counts {self.rollout_steps_per_stage}; "
                f"expected {_LOCKED_ROLLOUT_STEPS}"
            )
        self.student_scheduler.validate()
        self.teacher_scheduler.validate()
        if len(self.normal_stages) != 3 or len(self.empty_history_stages) != 3:
            raise ValueError(
                "DMD student schedule must contain exactly three pyramid stages"
            )
        expected_empty_multiplier = 2 if self.amplify_empty_history else 1
        for stage_index, (normal, empty) in enumerate(
            zip(self.normal_stages, self.empty_history_stages)
        ):
            expected_normal = self.rollout_steps_per_stage[stage_index]
            if len(normal) != expected_normal:
                raise ValueError(
                    f"DMD normal stage {stage_index} contains {len(normal)} steps, expected {expected_normal}"
                )
            if len(empty) != expected_normal * expected_empty_multiplier:
                raise ValueError(
                    f"DMD empty-history stage {stage_index} contains {len(empty)} steps, expected "
                    f"{expected_normal * expected_empty_multiplier}"
                )
            self._validate_stage(stage_index, normal, "normal")
            self._validate_stage(stage_index, empty, "empty_history")
        if (
            not self.amplify_empty_history
            and self.empty_history_stages != self.normal_stages
        ):
            raise ValueError(
                "Unamplified DMD contracts must use the normal table for empty history"
            )
        if self.latent_shape == _LOCKED_LATENT_SHAPE:
            self._validate_locked_timesteps(
                self.normal_stages, _LOCKED_NORMAL_TIMESTEPS, "normal"
            )
            if self.amplify_empty_history:
                self._validate_locked_timesteps(
                    self.empty_history_stages,
                    _LOCKED_EMPTY_HISTORY_TIMESTEPS,
                    "empty_history",
                )
        if (self.teacher_valid_timestep_min, self.teacher_valid_timestep_max) != (
            20.0,
            980.0,
        ):
            raise ValueError(
                "Teacher re-noise range must be the closed interval [20, 980]"
            )
        expected_teacher_grid = _teacher_grid(self.teacher_scheduler)
        if self.teacher_grid != expected_teacher_grid:
            raise ValueError(
                "Serialized teacher re-noise grid does not match the pinned natural8000 shift=5 scheduler"
            )
        if not self.teacher_valid_grid_indices:
            raise ValueError(
                "Teacher re-noise range contains no legal scheduler grid points"
            )

    @staticmethod
    def _validate_stage(
        stage_index: int, steps: Sequence[DmdTimestepStep], name: str
    ) -> None:
        previous_timestep = float("inf")
        for index, step in enumerate(steps):
            values = (step.model_timestep, step.current_sigma, step.next_sigma)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"Non-finite value in {name} stage {stage_index}, step {index}: {values}"
                )
            if not 0.0 <= step.next_sigma < step.current_sigma <= 1.0:
                raise ValueError(
                    f"Invalid sigma transition in {name} stage {stage_index}, step {index}: {values}"
                )
            if step.model_timestep >= previous_timestep:
                raise ValueError(
                    f"Timesteps are not strictly descending in {name} stage {stage_index}"
                )
            previous_timestep = step.model_timestep
            if index + 1 < len(steps) and not math.isclose(
                step.next_sigma,
                steps[index + 1].current_sigma,
                rel_tol=0.0,
                abs_tol=1e-7,
            ):
                raise ValueError(
                    f"Broken x0 re-noise transition in {name} stage {stage_index}, step {index}"
                )
        if steps[-1].next_sigma != 0.0:
            raise ValueError(
                f"The final {name} transition in stage {stage_index} must end at clean x0"
            )

    @staticmethod
    def _validate_locked_timesteps(
        stages: Sequence[Sequence[DmdTimestepStep]],
        expected: Sequence[Sequence[float]],
        name: str,
    ) -> None:
        for stage_index, (actual_stage, expected_stage) in enumerate(
            zip(stages, expected)
        ):
            actual = [step.model_timestep for step in actual_stage]
            if len(actual) != len(expected_stage) or any(
                not math.isclose(value, target, rel_tol=0.0, abs_tol=1e-3)
                for value, target in zip(actual, expected_stage)
            ):
                raise ValueError(
                    f"Locked 384x640x9 DMD {name} schedule changed at stage {stage_index}: "
                    f"expected {list(expected_stage)}, got {actual}"
                )

    @property
    def teacher_valid_grid_indices(self) -> tuple[int, ...]:
        return tuple(
            point.grid_index
            for point in self.teacher_grid
            if self.teacher_valid_timestep_min
            <= point.timestep
            <= self.teacher_valid_timestep_max
        )

    def stage(
        self, stage_index: int, *, empty_history: bool = False
    ) -> tuple[DmdTimestepStep, ...]:
        if stage_index not in range(3):
            raise IndexError(f"DMD pyramid stage must be 0, 1 or 2, got {stage_index}")
        return (self.empty_history_stages if empty_history else self.normal_stages)[
            stage_index
        ]

    def stage_tensors(
        self,
        stage_index: int,
        *,
        empty_history: bool = False,
        device: str | torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        steps = self.stage(stage_index, empty_history=empty_history)
        return {
            "model_timestep": torch.tensor(
                [step.model_timestep for step in steps],
                device=device,
                dtype=torch.float32,
            ),
            "current_sigma": torch.tensor(
                [step.current_sigma for step in steps],
                device=device,
                dtype=torch.float32,
            ),
            "next_sigma": torch.tensor(
                [step.next_sigma for step in steps], device=device, dtype=torch.float32
            ),
        }

    @staticmethod
    def student_condition(
        model_timestep: float | DmdTimestepStep | torch.Tensor,
        batch_size: int,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        if isinstance(model_timestep, DmdTimestepStep):
            model_timestep = model_timestep.model_timestep
        condition = torch.as_tensor(model_timestep, dtype=torch.float32, device=device)
        if condition.ndim == 0 or condition.numel() == 1:
            return condition.reshape(1).expand(batch_size)
        if condition.shape != (batch_size,):
            raise ValueError(
                f"Student timestep condition must be scalar or shape ({batch_size},), got {tuple(condition.shape)}"
            )
        return condition

    @staticmethod
    def renoise_x0(
        x0: torch.Tensor,
        noise: torch.Tensor,
        next_sigma: float | DmdTimestepStep | torch.Tensor,
    ) -> torch.Tensor:
        """Apply the shared flow transition ``(1-sigma)*x0 + sigma*noise``."""

        if x0.shape != noise.shape:
            raise ValueError(
                f"x0/noise shapes differ: {tuple(x0.shape)} vs {tuple(noise.shape)}"
            )
        if isinstance(next_sigma, DmdTimestepStep):
            next_sigma = next_sigma.next_sigma
        sigma = torch.as_tensor(next_sigma, device=x0.device, dtype=torch.float32)
        if sigma.ndim == 0:
            pass
        elif sigma.ndim == 1 and sigma.shape[0] in (1, x0.shape[0]):
            sigma = sigma.reshape(sigma.shape[0], *([1] * (x0.ndim - 1)))
        else:
            raise ValueError(
                f"next_sigma must be scalar or one value per batch item; got shape {tuple(sigma.shape)}"
            )
        # The contract owns the numerical transition as well as its coefficients:
        # every caller keeps the rollout state in FP32 between model queries and
        # casts only the transformer's input. Returning ``x0.dtype`` here would
        # silently make a BF16 caller follow a different trajectory.
        return (1.0 - sigma) * x0.to(torch.float32) + sigma * noise.to(torch.float32)

    def teacher_grid_tensors(
        self,
        *,
        device: str | torch.device | None = None,
        legal_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        points: Sequence[DmdTeacherGridPoint]
        if legal_only:
            legal = set(self.teacher_valid_grid_indices)
            points = tuple(
                point for point in self.teacher_grid if point.grid_index in legal
            )
        else:
            points = self.teacher_grid
        return {
            "indices": torch.tensor(
                [point.grid_index for point in points], device=device, dtype=torch.int64
            ),
            "timestep": torch.tensor(
                [point.timestep for point in points], device=device, dtype=torch.float32
            ),
            "sigma": torch.tensor(
                [point.sigma for point in points], device=device, dtype=torch.float32
            ),
        }

    def sample_teacher(
        self,
        batch_size: int,
        *,
        device: str | torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> TeacherTimestepSample:
        """Sample legal indices, then reproduce the teacher's BF16 sigma lookup.

        Lookup uses the complete descending FP32 table (including points outside
        the sampling interval) and argmin's first-index tie rule, as in the
        archived scheduler. The returned sigma uses this exact lookup rule.
        """

        if self.schema_version != DMD_TIMESTEP_CONTRACT_SCHEMA_VERSION:
            raise ValueError(
                "Legacy DMD contract is inference-only: teacher training requires "
                "a new schema-3 contract with BF16-condition sigma lookup."
            )
        if batch_size <= 0:
            raise ValueError(
                f"Teacher timestep sample needs a positive batch size, got {batch_size}"
            )
        legal = self.teacher_grid_tensors(device=device, legal_only=True)
        choices = torch.randint(
            low=0,
            high=legal["indices"].numel(),
            size=(batch_size,),
            device=device,
            generator=generator,
        )
        indices = legal["indices"][choices]
        timestep = legal["timestep"][choices].to(torch.float32)
        condition = timestep.to(torch.bfloat16)
        dense = self.teacher_grid_tensors(device=device)
        sigma_indices = (
            (dense["timestep"][None, :] - condition.float()[:, None])
            .abs()
            .argmin(dim=1)
        )
        sigma = dense["sigma"][sigma_indices]
        return TeacherTimestepSample(
            timestep=timestep,
            sigma=sigma,
            condition=condition,
            indices=indices,
            sigma_indices=sigma_indices,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "latent_shape": list(self.latent_shape),
            "student": {
                "condition_dtype": "float32",
                "rollout_steps_per_stage": list(self.rollout_steps_per_stage),
                "amplify_empty_history": self.amplify_empty_history,
                "scheduler_provenance": self.student_scheduler.to_dict(),
                "normal_stages": [
                    [step.to_dict() for step in stage] for stage in self.normal_stages
                ],
                "empty_history_stages": [
                    [step.to_dict() for step in stage]
                    for stage in self.empty_history_stages
                ],
            },
            "teacher_renoise": {
                **(
                    {"conditioning": dict(_TEACHER_CONDITIONING)}
                    if self.schema_version >= 3
                    else {}
                ),
                "sigma_dtype": "float32",
                "condition_dtype": "bfloat16",
                "valid_timestep_range": [
                    self.teacher_valid_timestep_min,
                    self.teacher_valid_timestep_max,
                ],
                "scheduler_provenance": self.teacher_scheduler.to_dict(),
                "grid": [point.to_dict() for point in self.teacher_grid],
            },
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_payload())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_payload()
        return {**payload, "fingerprint": self.fingerprint}

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        target = (
            path
            if path.suffix.lower() == ".json" and not path.is_dir()
            else path / DMD_TIMESTEP_CONTRACT_FILENAME
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        return target

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DmdTimestepContract":
        _require_exact_keys(
            payload,
            {"schema_version", "latent_shape", "student", "teacher_renoise"},
            "contract",
        )
        student = payload["student"]
        _require_exact_keys(
            student,
            {
                "condition_dtype",
                "rollout_steps_per_stage",
                "amplify_empty_history",
                "scheduler_provenance",
                "normal_stages",
                "empty_history_stages",
            },
            "student contract",
        )
        if student["condition_dtype"] != "float32":
            raise ValueError(
                f"DMD student condition must be float32, got {student['condition_dtype']}"
            )
        teacher = payload["teacher_renoise"]
        schema_version = int(payload["schema_version"])
        _require_exact_keys(
            teacher,
            {
                "sigma_dtype",
                "condition_dtype",
                "valid_timestep_range",
                "scheduler_provenance",
                "grid",
            }
            | ({"conditioning"} if schema_version >= 3 else set()),
            "teacher re-noise contract",
        )
        if schema_version >= 3 and teacher["conditioning"] != _TEACHER_CONDITIONING:
            raise ValueError(
                "Teacher conditioning must match the pinned mind111 BF16 sigma lookup"
            )
        if (
            teacher["sigma_dtype"] != "float32"
            or teacher["condition_dtype"] != "bfloat16"
        ):
            raise ValueError(
                "Teacher re-noise must use FP32 sigma and BF16 model conditioning"
            )
        valid_range = teacher["valid_timestep_range"]
        if not isinstance(valid_range, Sequence) or len(valid_range) != 2:
            raise ValueError(f"Invalid teacher timestep range: {valid_range}")
        return cls(
            schema_version=int(payload["schema_version"]),
            latent_shape=tuple(int(value) for value in payload["latent_shape"]),
            rollout_steps_per_stage=tuple(
                int(value) for value in student["rollout_steps_per_stage"]
            ),
            amplify_empty_history=bool(student["amplify_empty_history"]),
            normal_stages=tuple(
                tuple(DmdTimestepStep.from_dict(step) for step in stage)
                for stage in student["normal_stages"]
            ),
            empty_history_stages=tuple(
                tuple(DmdTimestepStep.from_dict(step) for step in stage)
                for stage in student["empty_history_stages"]
            ),
            student_scheduler=DmdStudentSchedulerProvenance.from_dict(
                student["scheduler_provenance"]
            ),
            teacher_scheduler=DmdTeacherSchedulerProvenance.from_dict(
                teacher["scheduler_provenance"]
            ),
            teacher_grid=tuple(
                DmdTeacherGridPoint.from_dict(point) for point in teacher["grid"]
            ),
            teacher_valid_timestep_min=float(valid_range[0]),
            teacher_valid_timestep_max=float(valid_range[1]),
        )

    @classmethod
    def load_json(
        cls,
        path: str | Path,
        *,
        expected_latent_shape: Sequence[int] | None = None,
        expected_fingerprint: str | None = None,
    ) -> "DmdTimestepContract":
        path = Path(path)
        source = (
            path
            if path.suffix.lower() == ".json" and not path.is_dir()
            else resolve_dmd_contract_path(path)
        )
        if not source.is_file():
            raise FileNotFoundError(
                f"Required DMD timestep contract sidecar is missing: {source}"
            )
        document = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or "fingerprint" not in document:
            raise ValueError(f"DMD timestep contract has no fingerprint: {source}")
        claimed_fingerprint = str(document.pop("fingerprint"))
        computed_fingerprint = hashlib.sha256(_canonical_json(document)).hexdigest()
        if claimed_fingerprint != computed_fingerprint:
            raise ValueError(
                f"DMD timestep contract fingerprint mismatch in {source}: claimed {claimed_fingerprint}, "
                f"computed {computed_fingerprint}"
            )
        if (
            expected_fingerprint is not None
            and claimed_fingerprint != expected_fingerprint
        ):
            raise ValueError(
                f"DMD timestep contract fingerprint {claimed_fingerprint} does not match expected "
                f"{expected_fingerprint}"
            )
        contract = cls.from_payload(document)
        if contract.fingerprint != claimed_fingerprint:
            raise ValueError(f"DMD timestep contract changed while decoding {source}")
        if expected_latent_shape is not None:
            expected_shape = tuple(int(value) for value in expected_latent_shape)
            if contract.latent_shape != expected_shape:
                raise ValueError(
                    f"DMD latent shape mismatch: sidecar has {contract.latent_shape}, runtime has {expected_shape}"
                )
        return contract

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        expected_latent_shape: Sequence[int] | None = None,
        expected_fingerprint: str | None = None,
    ) -> "DmdTimestepContract":
        return cls.load_json(
            resolve_dmd_contract_path(checkpoint_path),
            expected_latent_shape=expected_latent_shape,
            expected_fingerprint=expected_fingerprint,
        )
