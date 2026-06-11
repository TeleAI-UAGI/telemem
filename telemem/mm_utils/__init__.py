"""TeleMem multimodal (video) utilities.

Exports are resolved lazily (PEP 562) so that `import telemem` stays light:
the heavy video dependencies (opencv, yt-dlp, nano-vectordb, ...) are only
imported when one of these names is actually used. Install them with::

    pip install "telemem[video]"
"""

_LAZY_EXPORTS = {
    "MMCoreAgent": "telemem.mm_utils.core",
    "extract_choice_from_msg": "telemem.mm_utils.core",
    "process_video": "telemem.mm_utils.frame_caption",
    "init_single_video_db": "telemem.mm_utils.build_database",
    "decode_video_to_frames": "telemem.mm_utils.video_utils",
    "load_config": "telemem.mm_utils.memory_utils",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        import importlib

        try:
            module = importlib.import_module(_LAZY_EXPORTS[name])
        except ImportError as exc:
            raise ImportError(
                f"telemem.mm_utils.{name} requires TeleMem's video extras. "
                'Install them with: pip install "telemem[video]"'
            ) from exc
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
