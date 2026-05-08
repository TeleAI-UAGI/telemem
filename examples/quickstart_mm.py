import os
from pathlib import Path

import telemem as mem0
from telemem.mm_utils.core import extract_choice_from_msg
from telemem.utils import load_config


def make_memory():
    config_path = os.getenv("TELEMEM_CONFIG")
    if config_path:
        return mem0.Memory(config=load_config(config_path))
    return mem0.Memory()


def main():
    memory = make_memory()

    repo_root = Path(__file__).resolve().parents[1]
    video_path = repo_root / "data" / "samples" / "video" / "3EQLFHRHpag.mp4"
    video_name = video_path.stem
    output_dir = video_path.parent

    vdb_json_path = output_dir / "vdb" / video_name / f"{video_name}_vdb.json"
    if not vdb_json_path.exists():
        result = memory.add_mm(
            video_path=str(video_path),
            output_dir=str(output_dir),
        )
        print(f"Video processing complete: {result}")
    else:
        print(f"VDB already exists: {vdb_json_path}")

    question = """The problems people encounter in the video are caused by what?
(A) Catastrophic weather.
(B) Global warming.
(C) Financial crisis.
(D) Oil crisis.
"""

    messages = memory.search_mm(
        question=question,
        output_dir=str(output_dir),
        max_iterations=15,
    )

    answer = extract_choice_from_msg(messages)
    print(f"Answer: ({answer})")


if __name__ == "__main__":
    main()
