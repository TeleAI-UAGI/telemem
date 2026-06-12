# Video Memory

Beyond text, TeleMem lets agents **store, retrieve, and reason over video content** the
same way they handle text memories. The design draws on
[Deep Video Discovery](https://github.com/microsoft/DeepVideoDiscovery)'s agentic search
and tool-use approach.

Install the video extras first:

```shell
pip install -e ".[video]"
```

## Pipeline

`add_mm()` turns a raw video into retrievable memory in three cached stages:

1. **Frame extraction** — decode the video to JPEG frames at the configured FPS
2. **Caption generation** — a VLM (e.g. Qwen3-Omni, GPT-4.1-mini) describes each clip
3. **Vector database** — clip captions are embedded for semantic retrieval

```python
result = memory.add_mm(
    video_path="data/samples/video/3EQLFHRHpag.mp4",
    output_dir="data/samples/video",
)
```

Artifacts land under `frames/`, `captions/`, and `vdb/` inside `output_dir`; stages whose
outputs already exist are skipped automatically.

## ReAct-style video QA

`search_mm()` runs `MMCoreAgent`, a THINK → ACTION → OBSERVATION loop with three tools:

| Tool | Function |
| ---- | -------- |
| `global_browse_tool` | Global overview of video events and themes |
| `clip_search_tool` | Semantic search for specific content |
| `frame_inspect_tool` | Inspect frame details in a time range |

```python
messages = memory.search_mm(
    question="""The problems people encounter in the video are caused by what?
    (A) Catastrophic weather. (B) Global warming. (C) Financial crisis. (D) Oil crisis.""",
    output_dir="data/samples/video",
    max_iterations=15,
)

from telemem.mm_utils import extract_choice_from_msg
print(extract_choice_from_msg(messages))   # "B"
```

The full runnable demo is
[`examples/quickstart_mm.py`](https://github.com/TeleAI-UAGI/telemem/blob/main/examples/quickstart_mm.py);
the repository ships a small sample video. VLM and embedding endpoints are configured in
the `vlm:` section of your config file.
