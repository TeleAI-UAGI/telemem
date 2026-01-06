import copy
import functools
import json
import multiprocessing as mp
import os
from typing import Dict, List, Tuple, Optional

from tqdm import tqdm

# import config as config
from telemem.mm_utils.memory_utils import call_openai_model_with_tools, load_config
from openai import OpenAI
import openai

# from mm_utils.utils import load_config

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# def _ensure_cfg(cfg: Optional[dict]) -> dict:
#     """Return provided cfg or load default config if None."""
#     if cfg is not None:
#         return cfg
#     else:
#         config_path = os.environ.get("MM_CONFIG_PATH") or os.path.join(PARENT_DIR, "config.yaml")
#         return load_config(config_path)

# --------------------------------------------------------------------------- #
#                              Prompt templates                               #
# --------------------------------------------------------------------------- #

messages = [
    {
        "role": "system",
        "content": ""
    },
    {
        "role": "user",
        "content": "",
    },
]


CAPTION_PROMPT = """There are consecutive frames from a video. Please understand the video clip with the given transcript then output JSON in the template below.

Transcript of current clip:
TRANSCRIPT_PLACEHOLDER

Output template:
{
  "clip_start_time": CLIP_START_TIME,
  "clip_end_time": CLIP_END_TIME,
  "subject_registry": {
    <subject_i>: {
      "name": <fill with short identity if name is unknown>,
      "appearance": <list of appearance descriptions>,
      "identity": <list of identity descriptions>,
      "first_seen": <timestamp>
    },
    ...
  },
  "clip_description": <smooth and detailed natural narration of the video clip>
}
"""


MERGE_PROMPT = """You are given several partial `new_subject_registry` JSON objects extracted from different clips of the *same* video. They may contain duplicated subjects with slightly different IDs or descriptions.

Task:
1. Merge these partial registries into one coherent `subject_registry`.
2. Preserve all unique subjects.
3. If two subjects obviously refer to the same person, merge them
   (keep earliest `first_seen` time and union all fields).

Input (list of JSON objects):
REGISTRIES_PLACEHOLDER

Return *only* the merged `subject_registry` JSON object.
"""

SYSTEM_PROMPT = "You are a helpful assistant. You must write the output strictly in English. Do not use Chinese or any other language."

# --------------------------------------------------------------------------- #
#                               Local Client                                  #
# --------------------------------------------------------------------------- #

# api_config = json.load(open("api_config.json"))
# client = {}

# for model_name, conf in api_config.items():
#     # --- Azure OpenAI ---
#     if "azure_endpoint" in conf and conf["azure_endpoint"]:
#         client[model_name] = openai.AzureOpenAI(
#             azure_endpoint=conf["azure_endpoint"],
#             api_version=conf["api_version"],
#             api_key=conf["api_key"],
#         )
#     # --- OpenAI / local compatible endpoint ---
#     elif conf.get("base_url"):
#         client[model_name] = openai.OpenAI(
#             base_url=conf["base_url"],
#             api_key=conf["api_key"],
#         )
#     else:
#         print(f"⚠️ {model_name} skipped: no valid endpoint info")

# def get_response(model, messages, timeout=30):
#     """Get chat completion response from specified model."""
#     response = client[model].chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=1e-6,
#         timeout=timeout,
#         max_tokens=8192
#     )
#     return response.choices[0].message.content, response.usage.total_tokens
# --------------------------------------------------------------------------- #
#                               Helper utils                                  #
# --------------------------------------------------------------------------- #
def convert_seconds_to_hhmmss(seconds: float) -> str:
    h = int(seconds // 3600)
    seconds %= 3600
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"


def gather_frames_from_time_ranges(
    frame_folder: str, time_ranges: List[Tuple[int, int, str]]
) -> Dict[str, Dict]:
    """Return a dict keyed by 't1_t2' -> {files, transcript}."""
    frame_files = sorted(
        [f for f in os.listdir(frame_folder) if f.endswith(".jpg")],
        key=lambda x: float(x.split("_n")[-1].rstrip(".jpg")),
    )
    result = {}
    for t1, t2, text in time_ranges:
        files = frame_files[t1 : t2 + 1]
        result[f"{t1}_{t2}"] = {
            "files": [os.path.join(frame_folder, f) for f in files],
            "transcript": text or "No transcript.",
        }
    return result

def gather_clip_frames(
    video_frame_folder,
    clip_secs: int,
    subtitle_file_path: str = None,
    cfg: Optional[dict] = None,
) -> Dict[str, Dict]:
    # Fix possible typo in the earlier list-comprehension and gather frames again
    frame_files = sorted(
        [f for f in os.listdir(video_frame_folder) if f.startswith("frame") and f.endswith(".jpg")],
        key=lambda x: float(x.split("_n")[-1].rstrip(".jpg")),
    )
    if not frame_files:
        return {}

    # Optional subtitle information
    subtitle_map = (
        parse_srt_to_dict(subtitle_file_path) if subtitle_file_path else {}
    )

    # Map timestamps → file names for quick lookup
    # cfg = _ensure_cfg(cfg)
    frame_ts = [float(f.split("_n")[-1].rstrip(".jpg")) / cfg['VIDEO_FPS'] for f in frame_files]
    ts_to_file = dict(zip(frame_ts, frame_files))
    last_ts = int(max(frame_ts))

    result = []

    # Iterate over fixed-length clips
    clip_start = 0
    while clip_start <= last_ts:
        clip_end = min(clip_start + clip_secs - 1, last_ts)

        # Collect frames that fall inside the current clip
        clip_files = [
            os.path.join(video_frame_folder, ts_to_file[t])
            for t in frame_ts
            if clip_start <= t <= clip_end
        ]

        # Aggregate transcript text overlapping the clip interval
        transcript_parts: List[str] = []
        for key, text in subtitle_map.items():
            s, e = map(int, key.split("_"))
            if s <= clip_end and e >= clip_start:  # overlap check
                transcript_parts.append(text)
        transcript = " ".join(transcript_parts).strip() or "No transcript."

        result.append((
                f"{clip_start}_{clip_end}", 
                {"files": clip_files, "transcript": transcript}
        ))

        clip_start += clip_secs
    return result


# --------------------------------------------------------------------------- #
#                   Subtitle (.srt) parsing helper function                    #
# --------------------------------------------------------------------------- #
def _timestamp_to_seconds(ts: str) -> float:
    """Convert 'HH:MM:SS,mmm' to seconds (float)."""
    hh, mm, rest = ts.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def parse_srt_to_dict(srt_path: str) -> Dict[str, str]:
    """
    Parse an .srt file and return a mapping
    '{startSec_endSec}': 'subtitle text'.
    """
    if not os.path.isfile(srt_path):
        return {}

    result: Dict[str, str] = {}
    with open(srt_path, "r", encoding="utf-8") as fh:
        lines = [l.rstrip("\n") for l in fh]

    idx = 0
    n = len(lines)
    while idx < n:
        # Skip sequential index if present
        if lines[idx].strip().isdigit():
            idx += 1
        if idx >= n:
            break

        # Time-range line
        if "-->" not in lines[idx]:
            idx += 1
            continue
        start_ts, end_ts = [t.strip() for t in lines[idx].split("-->")]
        start_sec = int(_timestamp_to_seconds(start_ts))
        end_sec = int(_timestamp_to_seconds(end_ts))
        idx += 1

        # Collect subtitle text (may span multiple lines)
        subtitle_lines: List[str] = []
        while idx < n and lines[idx].strip():
            subtitle_lines.append(lines[idx].strip())
            idx += 1
        subtitle = " ".join(subtitle_lines)
        key = f"{start_sec}_{end_sec}"
        if key in result:  # append if duplicate key
            result[key] += " " + subtitle
        else:
            result[key] = subtitle
        # Skip blank line separating entries
        idx += 1
    return result


# --------------------------------------------------------------------------- #
#                        LLM wrappers (single clip)                           #
# --------------------------------------------------------------------------- #
def _caption_clip(task: Tuple[str, Dict], caption_ckpt_folder, cfg: Optional[dict] = None) -> Tuple[str, dict]:
    """LLM call for one clip. Returns (timestamp_key, parsed_json)."""
    timestamp, info = task
    files, transcript = info["files"], info["transcript"]

    
    clip_start_time = convert_seconds_to_hhmmss(float(timestamp.split("_")[0]))
    clip_end_time = convert_seconds_to_hhmmss(float(timestamp.split("_")[1]))

    send_messages = copy.deepcopy(messages)
    send_messages[0]["content"] = SYSTEM_PROMPT
    send_messages[1]["content"] = CAPTION_PROMPT.replace(
        "TRANSCRIPT_PLACEHOLDER", transcript).replace(
        "CLIP_START_TIME", clip_start_time).replace(
        "CLIP_END_TIME", clip_end_time)
    

    
    # cfg = _ensure_cfg(cfg)

    if os.path.exists(os.path.join(caption_ckpt_folder, f"{timestamp}.json")):
        # If the caption already exists, skip processing
        with open(os.path.join(caption_ckpt_folder, f"{timestamp}.json"), "r") as f:
            return timestamp, json.load(f)

    tries = 3
    while tries:
        tries -= 1
        resp = call_openai_model_with_tools(
            send_messages,
            endpoints=cfg['vlm_client'],
            model_name=cfg['vlm_model'],
            return_json=True,
            use_local=True,
            image_paths=files,
            # api_key=cfg['vlm_api_key'],
        )["content"]
        # resp, _ = get_response(
        #     model=cfg['vlm_model'],
        #     messages=send_messages,
        #     timeout=30,
        # )
        if resp is None:
            continue
        try:
            assert isinstance(resp, str), f"Response must be a JSON string instead of {type(resp)}:{resp}."
            parsed = json.loads(resp)
            desc = parsed.get("clip_description")
            if isinstance(desc, str) and desc.strip():
                parsed["clip_description"] = desc + f"\n\nTranscript during this video clip: {transcript}."
            else:
                
                parsed["clip_description"] = f"Transcript during this video clip: {transcript}."
            # parsed["clip_description"] += f"\n\nTranscript during this video clip: {transcript}." # add transcript to description
            

            resp = json.dumps(parsed)
            with open(os.path.join(caption_ckpt_folder, f"{timestamp}.json"), "w") as f:
                f.write(resp)
            return timestamp, parsed
        except json.JSONDecodeError:
            continue
    return timestamp, {}  # give up


# --------------------------------------------------------------------------- #
#                  LLM wrapper – merge subject registries                     #
# --------------------------------------------------------------------------- #


def merge_subject_registries(registries: List[dict], cfg: Optional[dict] = None) -> dict:
    if not registries:
        return {}

    # cfg = _ensure_cfg(cfg)

    BATCH_SIZE = 20

    def _merge_once(batch: List[dict], level: int, idx: int, cfg: dict) -> dict:
        print(
            f"[INFO] merge_subject_registries: level={level}, batch_idx={idx}, "
            f"batch_size={len(batch)}",
            flush=True,
        )

        send_messages = copy.deepcopy(messages)
        send_messages[0]["content"] = SYSTEM_PROMPT

        regs_json = json.dumps(batch)
        MAX_CHARS = 20000
        if len(regs_json) > MAX_CHARS:
            regs_json = regs_json[:MAX_CHARS]
            print("[WARN] merge_subject_registries: batch json truncated", flush=True)

        send_messages[1]["content"] = MERGE_PROMPT.replace(
            "REGISTRIES_PLACEHOLDER", regs_json
        )

        tries = 2  
        while tries:
            tries -= 1
            try:
                resp_obj = call_openai_model_with_tools(
                    send_messages,
                    endpoints=cfg['vlm_client'],
                    model_name=cfg['vlm_model'],
                    return_json=True,
                    use_local=True,
                    max_tokens=1024,
                    temperature=0.0,
                )
            except Exception as e:
                print(f"[ERROR] merge_once LLM error: {e!r}", flush=True)
                continue

            resp = resp_obj.get("content")
            if resp is None:
                print("[WARN] merge_once got None response", flush=True)
                continue
            try:
                return json.loads(resp)
            except json.JSONDecodeError as e:
                print(f"[WARN] merge_once json decode error: {e!r}, resp[:200]={str(resp)[:200]!r}", flush=True)
                continue

        print("[WARN] merge_once failed after retries, return {}", flush=True)
        return {}

    # Hierarchical merging
    current_regs = registries
    level = 0
    while len(current_regs) > 1:
        level += 1
        print(f"[INFO] merge_subject_registries: start level {level}, num_regs={len(current_regs)}", flush=True)
        next_round = []
        for i in range(0, len(current_regs), BATCH_SIZE):
            batch = current_regs[i:i + BATCH_SIZE]
            merged = _merge_once(batch, level=level, idx=i // BATCH_SIZE, cfg=cfg)
            if merged:
                next_round.append(merged)
        if not next_round:
            print("[WARN] merge_subject_registries: next_round empty, stop merging", flush=True)
            break
        current_regs = next_round

    return current_regs[0] if current_regs else {}


# --------------------------------------------------------------------------- #
#                     Process one video (parallel caption)                    #
# --------------------------------------------------------------------------- #
def process_video(
    frame_folder: str,
    output_caption_folder: str,
    subtitle_file_path: str = None,
    cfg: Optional[dict] = None,
):
    caption_ckpt_folder = os.path.join(output_caption_folder, "ckpt")
    os.makedirs(caption_ckpt_folder, exist_ok=True)

    # cfg = _ensure_cfg(cfg)

    clips = gather_clip_frames(frame_folder, cfg['CLIP_SECS'], subtitle_file_path, cfg=cfg)

    caption_clip = functools.partial(
        _caption_clip,
        caption_ckpt_folder=caption_ckpt_folder,
        cfg=cfg,
    )
    # ---------------- Parallel captioning --------------- #
    with mp.Pool(16) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(caption_clip, clips),
                total=len(clips),
                desc=f"Captioning {frame_folder}",
            )
        )

    # ---------------- Save per-clip JSON ---------------- #
    partial_registries = []
    frame_captions = {}
    results = sorted(results, key=lambda x: float(x[0].split("_")[0]))
    for ts, parsed in results:
        if parsed:
            frame_captions[ts] = {
                "caption": parsed["clip_description"],
            }
            subject_reg = parsed.get("subject_registry")
            if subject_reg:
                partial_registries.append(subject_reg)
            # partial_registries.append(parsed["subject_registry"])


    # ---------------- Merge subject registries ---------- #
    merged_registry = merge_subject_registries(partial_registries, cfg=cfg)
    frame_captions["subject_registry"] = merged_registry

    with open(
        os.path.join(output_caption_folder, "captions.json"), "w"
    ) as f:
        json.dump(frame_captions, f, indent=4)


def process_video_lite(
    output_caption_folder: str,
    subtitle_file_path: str,
):
    """
    Process video in LITE_MODE using SRT subtitles.
    """
    captions = parse_srt_to_dict(subtitle_file_path)
    frame_captions = {}
    for key, text in captions.items():
        frame_captions[key] = {
            "caption": f"\n\nTranscript during this video clip: {text}.",
        }
    frame_captions["subject_registry"] = {}
    with open(
        os.path.join(output_caption_folder, "captions.json"), "w"
    ) as f:
        json.dump(frame_captions, f, indent=4)


def process_all_videos_mme(
    frames_root: str,
    captions_root: str,
    chunk_range=range(1, 21),
    cfg: Optional[dict] = None,
    ):
    # cfg = _ensure_cfg(cfg)
    for i in chunk_range:
        chunk_name = f"videos_chunked_{i:02d}"
        chunk_frames_dir = os.path.join(frames_root, chunk_name)
        chunk_captions_dir = os.path.join(captions_root, chunk_name)

        if not os.path.isdir(chunk_frames_dir):
            print(f"[SKIP] {chunk_name}: frames dir not found -> {chunk_frames_dir}")
            continue

        os.makedirs(chunk_captions_dir, exist_ok=True)

        # Iterate through each video id directory under this chunk
        video_ids = [
            d for d in os.listdir(chunk_frames_dir)
            if os.path.isdir(os.path.join(chunk_frames_dir, d))
        ]
        if not video_ids:
            print(f"[SKIP] {chunk_name}: no video folders under {chunk_frames_dir}")
            continue

        print(f"\n=== Processing {chunk_name} ({len(video_ids)} videos) ===")

        for vid in tqdm(video_ids, desc=f"{chunk_name} videos", unit="vid"):
            frame_folder = os.path.join(chunk_frames_dir, vid, "frames")
            if not os.path.isdir(frame_folder):
                # Compatible with cases where frames are directly under the video directory
                frame_folder = os.path.join(chunk_frames_dir, vid)
                if not os.path.isdir(frame_folder):
                    print(f"[SKIP] {chunk_name}/{vid}: no frames folder")
                    continue

            output_caption_folder = os.path.join(chunk_captions_dir, vid)
            os.makedirs(output_caption_folder, exist_ok=True)
            
            captions_json_path = os.path.join(output_caption_folder, "captions.json")

            # Skip the entire video if captions.json already exists
            if os.path.isfile(captions_json_path):
                print(f"[SKIP] {chunk_name}/{vid}: captions.json already exists -> {captions_json_path}")
                continue
            
            # No subtitles passed here; extend as needed if SRT is available
            try:
                process_video(
                    frame_folder=frame_folder,
                    output_caption_folder=output_caption_folder,
                    subtitle_file_path=None,
                    cfg=cfg,
                )
            except Exception as e:
                print(f"[ERROR] {chunk_name}/{vid}: {e!r}")


# ...existing code...

def process_all_videos_m3_agent(
    frames_root: str,
    captions_root: str,
    cfg: Optional[dict] = None,
):
    """
    Batch process all scene videos under m3-agent_dvd/m3-agent_database.
    Directory structure example:
        frames_root/
            bedroom_01/
                frames/
                    frame_000001_n0.00.jpg
                    ...
            kitchen_01/
                frames/
                    ...
    Output:
        captions_root/
            bedroom_01/
                captions.json
            kitchen_01/
                captions.json
    """
    # cfg = _ensure_cfg(cfg)

    if not os.path.isdir(frames_root):
        print(f"[ERROR] m3-agent frames_root not found -> {frames_root}")
        return

    os.makedirs(captions_root, exist_ok=True)

    # Iterate through all scene directories
    scene_ids = [
        d for d in os.listdir(frames_root)
        if os.path.isdir(os.path.join(frames_root, d))
    ]
    if not scene_ids:
        print(f"[WARN] no scene folders under {frames_root}")
        return

    print(f"\n=== Processing m3-agent database ({len(scene_ids)} scenes) ===")

    for sid in tqdm(scene_ids, desc="m3-agent scenes", unit="scene"):
        scene_dir = os.path.join(frames_root, sid)
        frame_folder = os.path.join(scene_dir, "frames")
        if not os.path.isdir(frame_folder):
            # Compatible with cases where frames are directly under the scene directory
            frame_folder = scene_dir
            if not os.path.isdir(frame_folder):
                print(f"[SKIP] m3-agent/{sid}: no frames folder")
                continue

        output_caption_folder = os.path.join(captions_root, sid)
        os.makedirs(output_caption_folder, exist_ok=True)

        captions_json_path = os.path.join(output_caption_folder, "captions.json")
        # Skip if already exists to avoid regeneration
        if os.path.isfile(captions_json_path):
            print(f"[SKIP] m3-agent/{sid}: captions.json already exists -> {captions_json_path}")
            continue

        try:
            # m3-agent currently has no subtitles, subtitle_file_path is None
            process_video(
                frame_folder=frame_folder,
                output_caption_folder=output_caption_folder,
                subtitle_file_path=None,
                cfg=cfg,
            )
        except Exception as e:
            print(f"[ERROR] m3-agent/{sid}: {e!r}")
# --------------------------------------------------------------------------- #
#                                    main                                     #
# --------------------------------------------------------------------------- #
def main():
    # cfg = _ensure_cfg(None)

    # Existing Video-MME processing logic
    frames_root = "../videomme_dvd/videomme_database"
    captions_root = "../videomme_dvd/captions"
    subtitle_file_path = "/home/xiaoyizhang/DVD/video_database/raw/i2qSxMVeVLI.srt"
    process_all_videos_mme(
        frames_root=frames_root,
        captions_root=captions_root,
        chunk_range=range(9, 21),
        cfg=cfg,
    )

    # m3_frames_root = "../m3-agent_dvd/m3-agent_database/robot"
    # m3_captions_root = "../m3-agent_dvd/captions"
    # process_all_videos_m3_agent(
    #     frames_root=m3_frames_root,
    #     captions_root=m3_captions_root,
    #     cfg=cfg,
    # )


if __name__ == "__main__":
    main()