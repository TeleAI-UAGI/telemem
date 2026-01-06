import os
import shutil
from urllib.parse import urlparse
from typing import Any, Dict

import cv2
import yt_dlp

from tqdm import tqdm
from telemem.mm_utils.memory_utils import load_config



def _is_youtube_url(url: str) -> bool:
    """Checks if a URL is a valid YouTube URL."""
    parsed_url = urlparse(url)
    return parsed_url.netloc.lower().endswith(('youtube.com', 'youtu.be'))


def load_video(
    video_source: str,
    cfg: Dict[str, Any],
    with_subtitle: bool = False,
    subtitle_source: str | None = None,
) -> str:
    """
    Loads a video from YouTube or a local file into the video database.
    Subtitle support is limited to the SRT format only.

    Args:
        video_source: YouTube URL or local video file path.
        with_subtitle: If True, also downloads / copies subtitles (SRT only).
        subtitle_source: Language code for YouTube subtitles (e.g., 'en', 'auto')
                         or local *.srt file path when video_source is local.

    Returns:
        Absolute path to the video file stored in the database.

    Raises:
        ValueError, FileNotFoundError: On invalid inputs.
    """
    raw_video_dir = os.path.join(cfg['VIDEO_DATABASE_FOLDER'], 'raw')
    os.makedirs(raw_video_dir, exist_ok=True)

    # ------------------- YouTube source -------------------
    if video_source.startswith(('http://', 'https://')):
        if not _is_youtube_url(video_source):
            raise ValueError("Provided URL is not a valid YouTube link.")

        ydl_opts = {
            'format': (
                f'bestvideo[height<={cfg["VIDEO_RESOLUTION"]}][ext=mp4]'
                f'best[height<={cfg["VIDEO_RESOLUTION"]}][ext=mp4]'
            ),
            'outtmpl': os.path.join(raw_video_dir, '%(id)s.%(ext)s'),
            'merge_output_format': 'mp4',
        }
        if with_subtitle:
            ydl_opts.update({
                'writesubtitles': True,
                'subtitlesformat': 'srt',
                'overwritesubtitles': True,
            })

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_source, download=True)
            video_path = ydl.prepare_filename(info)

        # rename subtitle -> "<video_file_name>.srt"
        if with_subtitle:
            video_base = os.path.splitext(video_path)[0]
            for f in os.listdir(raw_video_dir):
                if f.startswith(info["id"]) and f.endswith(".srt"):
                    shutil.move(
                        os.path.join(raw_video_dir, f),
                        f"{video_base}.srt",
                    )
                    break

        return os.path.abspath(video_path)

    # ------------------- Local source -------------------
    if os.path.exists(video_source):
        if not os.path.isfile(video_source):
            raise ValueError(f"Source path '{video_source}' is a directory, not a file.")

        filename = os.path.basename(video_source)
        destination_path = os.path.join(raw_video_dir, filename)
        shutil.copy2(video_source, destination_path)

        # copy subtitle file if requested (must be *.srt) and rename
        if with_subtitle:
            if not subtitle_source:
                raise ValueError("subtitle_source must be provided for local videos.")
            if not subtitle_source.lower().endswith('.srt'):
                raise ValueError("Only SRT subtitle files are supported for local videos.")
            if not os.path.isfile(subtitle_source):
                raise FileNotFoundError(f"Subtitle file '{subtitle_source}' not found.")

            subtitle_destination = os.path.join(
                raw_video_dir,
                f"{os.path.splitext(filename)[0]}.srt",
            )
            shutil.copy2(subtitle_source, subtitle_destination)

def download_srt_subtitle(video_url: str, output_path: str):
    """Downloads an SRT subtitle from a YouTube URL."""
    if not _is_youtube_url(video_url):
        raise ValueError("Provided URL is not a valid YouTube link.")

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        'writesubtitles': True,
        'subtitlesformat': 'srt',
        'skip_download': True,
        'writeautomaticsub': True,
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        video_id = info['id']
        ydl.download([video_url])

    # Locate the downloaded subtitle file (yt-dlp names them as <id>.<lang>.srt)
    downloaded_subtitle_path = None
    for f in os.listdir(output_dir):
        if f.startswith(video_id) and f.endswith(".srt"):
            downloaded_subtitle_path = os.path.join(output_dir, f)
            break

    if downloaded_subtitle_path:
        shutil.move(downloaded_subtitle_path, output_path)
    else:
        raise FileNotFoundError(f"Could not find SRT subtitle for {video_url}")


def decode_video_to_frames(video_path: str, frames_dir: str, cfg: Dict[str, Any]) -> str:
    """
    Decodes a video into JPEG frames at the frame rate specified by cfg['VIDEO_FPS'].
    Frames are saved in cfg['VIDEO_DATABASE_PATH']/video_names/frames/.

    Args:
        video_path: The absolute path to the video file.

    Returns:
        The absolute path to the directory containing the extracted frames.

    Raises:
        FileNotFoundError: If the video file does not exist.
        Exception: If frame extraction fails.
    """

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file '{video_path}' does not exist.")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # frames_dir = os.path.join("./videomme_database/videos_chunked_01", video_name, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Failed to open video file '{video_path}'.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = cfg.get('VIDEO_FPS', fps)
    frame_interval = int(round(fps / target_fps)) if target_fps < fps else 1

    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(frames_dir, f"frame_n{saved_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    return os.path.abspath(frames_dir)

from concurrent.futures import ThreadPoolExecutor, as_completed

def batch_decode_videos_to_frames(
    input_dir: str,
    output_root: str,
    cfg: Dict[str, Any],
    max_workers: int = 4,
    ) -> None:
    """
    Batch decodes all videos in the input directory to frames.

    Args:
        input_dir: Directory containing video files.
        output_root: Root directory to save extracted frames.
        max_workers: Number of parallel threads for processing.
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")
    
    os.makedirs(output_root, exist_ok=True)

    video_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".mp4")
    ]
    if not video_files:
        print(f"No .mp4 files found in {input_dir}")
        return
    
    def _worker(video_path: str) -> tuple[str, str | None, Exception | None]:
        try:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            frames_dir = os.path.join(output_root, video_name, "frames")
            os.makedirs(frames_dir, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Failed to open video file '{video_path}'.")

            fps = cap.get(cv2.CAP_PROP_FPS)
            target_fps = cfg.get('VIDEO_FPS', fps)
            frame_interval = int(round(fps / target_fps)) if target_fps < fps else 1

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            pbar = tqdm(
                total=total_frames,
                desc=f"[{video_name}] decoding",
                unit="f",
                leave=False,
            )

            frame_count = 0
            saved_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    frame_filename = os.path.join(frames_dir, f"frame_n{saved_count:06d}.jpg")
                    cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 80],)
                    saved_count += 1
                frame_count += 1
                pbar.update(1)

            pbar.close()
            cap.release()
            return video_path, frames_dir, None
        except Exception as e:
            return video_path, None, e
    
    print(f"Found {len(video_files)} videos in {input_dir}. Start decoding with {max_workers} workers...")


    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, vp): vp for vp in video_files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Videos", unit="vid"):
            vp = futures[fut]
            _, out_dir, err = fut.result()
            if err is not None:
                print(f"[ERROR] {vp}: {err}")
            else:
                print(f"[OK] {vp} -> {out_dir}")


if __name__ == "__main__":
    # download_srt_subtitle("https://www.youtube.com/watch?v=PQFQ-3d2J-8", "./video_database/PQFQ-3d2J-8/subtitles.srt")
    PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT_CONFIG_PATH = os.path.join(PARENT_DIR, "config.yaml")
    cfg = load_config(DEFAULT_CONFIG_PATH)

    input_dir = "../m3-agent/robot/"
    output_root = "../m3-agent_dvd/videomme_database/robot/"

    batch_decode_videos_to_frames(
        input_dir=input_dir,
        output_root=output_root,
        cfg=cfg,
        max_workers=8,
    )
    print(f"Frames saved to: {output_root}")

    # root_input_base = "../Video-MME/"
    # root_output_base = "../videomme_dvd/videomme_database/"

    # for i in range(4, 21):
    #     chunk_name = f"videos_chunked_{i:02d}"
    #     input_dir = os.path.join(root_input_base, chunk_name, "data")
    #     output_root = os.path.join(root_output_base, chunk_name)

    #     print(f"\n=== Processing {chunk_name} ===")

    #     try:
    #         batch_decode_videos_to_frames(
    #             input_dir=input_dir,
    #             output_root=output_root,
    #             cfg=cfg,
    #             max_workers=8,
    #         )
    #     except FileNotFoundError as e:
    #         print(f"[SKIP] {chunk_name}: {e}")
    #     except Exception as e:
    #         print(f"[ERROR] {chunk_name}: {e}")

    # print("Frames saved Done")