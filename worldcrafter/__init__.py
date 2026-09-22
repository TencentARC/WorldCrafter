from importlib import import_module


__all__ = [
    "InferenceResult",
    "RepEncoder",
    "WorldCrafter",
    "WorldCrafterPipeline",
    "WorldCrafterScheduler",
    "WorldCrafterTransformer3DModel",
]


def __getattr__(name):
    modules = {
        "InferenceResult": ".inference",
        "WorldCrafter": ".inference",
        "RepEncoder": ".repencoder",
        "WorldCrafterPipeline": ".diffusers",
        "WorldCrafterScheduler": ".diffusers",
        "WorldCrafterTransformer3DModel": ".diffusers",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
