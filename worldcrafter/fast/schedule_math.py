"""Schedule arithmetic retained from the authenticated DMD reference."""

import math
from typing import Literal


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def resolve_shift_mu(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    exp_max: float = 7.0,
    time_shift_type: Literal["exponential", "linear"] = "linear",
):
    """The shift strength for a sequence length, in the units ``apply_schedule_shift`` wants.

    The exponential schedule is the linear one with an exponentiated mu, so callers
    that pass ``mu`` explicitly have to exponentiate it themselves. Going through
    here keeps them from forgetting.
    """
    mu = calculate_shift(
        image_seq_len,
        base_seq_len if base_seq_len is not None else 256,
        max_seq_len if max_seq_len is not None else 4096,
        base_shift if base_shift is not None else 0.5,
        max_shift if max_shift is not None else 1.15,
    )
    if time_shift_type == "exponential":
        mu = math.exp(min(mu, math.log(exp_max)))
    return mu


def apply_schedule_shift(
    sigmas,
    noise,
    sigmas_two=None,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    exp_max: float = 7.0,
    time_shift_type: Literal["exponential", "linear"] = "linear",
    mu: float = None,
    return_mu: bool = False,
):
    if mu is None:
        # Resolution-dependent shifting of timestep schedules as per section 5.3.2 of SD3 paper
        image_seq_len = (
            noise.shape[-1] * noise.shape[-2] * noise.shape[-3]
        ) // 4  # patch size 1,2,2
        mu = resolve_shift_mu(
            image_seq_len,
            base_seq_len=base_seq_len,
            max_seq_len=max_seq_len,
            base_shift=base_shift,
            max_shift=max_shift,
            exp_max=exp_max,
            time_shift_type=time_shift_type,
        )

    if sigmas_two is not None:
        sigmas = (sigmas * mu) / (1 + (mu - 1) * sigmas)
        sigmas_two = (sigmas_two * mu) / (1 + (mu - 1) * sigmas_two)
        if return_mu:
            return sigmas, sigmas_two, mu
        else:
            return sigmas, sigmas_two
    else:
        sigmas = (sigmas * mu) / (1 + (mu - 1) * sigmas)
        if return_mu:
            return sigmas, mu
        else:
            return sigmas
