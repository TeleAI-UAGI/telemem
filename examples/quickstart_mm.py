from vendor.TeleMem.TeleMemory import TeleMemory
from vendor.TeleMem.utils import load_config
import os

# Initialize
config = load_config("config/config.yaml")
memory = TeleMemory.from_config(config)

# Define paths
video_path = "data/samples/video/3EQLFHRHpag.mp4"
video_name = os.path.splitext(os.path.basename(video_path))[0]

# Step 1: Add video to memory (auto-processing)
if not os.path.exists(f"video/vdb/{video_name}/{video_name}_vdb.json"):
    result = memory.add_mm(
        video_path=video_path,
        frames_root="video/frames",
        captions_root="video/captions",
        vdb_root="video/vdb",
    )
    print(f"Video processing complete: {result}")

# Step 2: Query video content
question = """The problems people encounter in the video are caused by what?
(A) Catastrophic weather.
(B) Global warming.
(C) Financial crisis.
(D) Oil crisis.
"""

messages = memory.search_mm(
    question=question,
    video_db_path=f"vendor/TeleMem/video/vdb/{video_name}/{video_name}_vdb.json",
    video_caption_path=f"vendor/TeleMem/video/captions/{video_name}/captions.json",
    max_iterations=15,
)

# Extract final answer
from core import extract_choice_from_msg
answer = extract_choice_from_msg(messages)
print(f"Answer: ({answer})")
