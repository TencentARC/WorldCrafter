"""Metric c2w controls: x right, y down, z forward, endpoint-exclusive frames."""

from dataclasses import asdict, dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Action:
    forward: float = 0.0
    right: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    speed: float = 1.0
    up: float = 0.0

    def normalized(self):
        values = [self.forward, self.right, self.up, self.yaw, self.pitch, self.speed]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Control values must be finite")
        if sum(v != 0 for v in values[:5]) > 1:
            raise ValueError("Only one movement or rotation may be active per chunk")
        return Action(
            forward=max(-1.0, min(1.0, self.forward)),
            right=max(-1.0, min(1.0, self.right)),
            up=max(-1.0, min(1.0, self.up)),
            yaw=max(-30.0, min(30.0, self.yaw)),
            pitch=max(-30.0, min(30.0, self.pitch)),
            speed=max(0.1, min(5.0, self.speed)),
        )

    def json(self):
        return asdict(self)


class ControlBuffer:
    """Latest fresh keydown wins until consume; never resume an overridden key."""

    KEYS = {
        "w": ("forward", 1),
        "s": ("forward", -1),
        "a": ("right", -1),
        "d": ("right", 1),
        "q": ("up", 1),
        "e": ("up", -1),
        "arrowleft": ("yaw", -1),
        "arrowright": ("yaw", 1),
        "arrowup": ("pitch", 1),
        "arrowdown": ("pitch", -1),
    }

    def __init__(self):
        self.clear()
        self.speed = 1.0
        self.vertical_speed = 1.0
        self.rotation_angle = 15.0

    def clear(self):
        self.held = set()
        self.selected_key = None
        self.pending_key = None

    def update(self, message):
        kind = message["type"]
        if kind == "blur":
            self.clear()
        elif kind == "key":
            key = str(message.get("key", "")).lower()
            if key not in self.KEYS:
                raise ValueError("Unsupported key")
            if message.get("down"):
                if key not in self.held:
                    self.selected_key = self.pending_key = key
                self.held.add(key)
            else:
                self.held.discard(key)
        elif kind in ("speed", "vertical_speed", "rotation_angle"):
            value = float(message["value"])
            if not math.isfinite(value):
                raise ValueError("Control setting must be finite")
            low, high = (1.0, 30.0) if kind == "rotation_angle" else (0.1, 5.0)
            setattr(self, kind, max(low, min(high, value)))
        else:
            raise ValueError("Unknown control")

    def peek(self):
        key = self.pending_key
        if key is None and self.selected_key in self.held:
            key = self.selected_key
        if key is None:
            return Action(speed=self.speed)
        field, sign = self.KEYS[key]
        value = sign * self.rotation_angle if field in ("yaw", "pitch") else sign
        speed = self.vertical_speed if field == "up" else self.speed
        return Action(**{field: value}, speed=speed).normalized()

    def consume(self):
        action = self.peek()
        self.pending_key = None
        return action

    def has_action(self):
        return self.pending_key is not None or self.selected_key in self.held


class Camera:
    def __init__(self):
        self.world = np.eye(4, dtype=np.float64)
        self.local_chunks, self.global_chunks = [], []

    @staticmethod
    def relative(action, alpha, up_direction=None):
        y, p = math.radians(action.yaw) * alpha, math.radians(action.pitch) * alpha
        cy, sy, cp, sp = math.cos(y), math.sin(y), math.cos(p), math.sin(p)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.array(
            [[cy, sy * sp, sy * cp], [0, cp, -sp], [-sy, cy * sp, cy * cp]]
        )
        pose[:3, 3] = [
            alpha * action.right * action.speed,
            -alpha * action.up * action.speed if action.up else 0.0,
            alpha * action.forward * action.speed,
        ]
        if action.up and up_direction is not None:
            pose[:3, 3] = alpha * action.up * action.speed * up_direction
        return pose

    def append(self, action):
        action = action.normalized()
        # Q/E always follows world vertical, including after a pitch rotation.
        up_direction = self.world[:3, :3].T @ np.array([0.0, -1.0, 0.0])
        local = np.stack(
            [self.relative(action, j / 33.0, up_direction) for j in range(33)]
        )
        global_pose = self.world[None] @ local
        if action.up:
            global_pose[:, :3, 3] = self.world[:3, 3]
            global_pose[:, 1, 3] -= np.arange(33) / 33.0 * action.up * action.speed
            self.world[1, 3] -= action.up * action.speed
        else:
            self.world = self.world @ self.relative(action, 1.0)
        self.local_chunks.append(local[:, :3].astype(np.float32))
        self.global_chunks.append(global_pose[:, :3].astype(np.float32))
        return self.poses()

    def poses(self):
        return np.concatenate(self.local_chunks), np.concatenate(self.global_chunks)
