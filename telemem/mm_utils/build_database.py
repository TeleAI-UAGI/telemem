import functools
import json
import multiprocessing
import os
from typing import Annotated as A, Optional

import numpy as np
from nano_vectordb import NanoVectorDB
from tqdm import tqdm

# import config as config
from func_call_shema import doc as D
from telemem.mm_utils.memory_utils import AzureOpenAIEmbeddingService, call_openai_model_with_tools

from telemem.mm_utils.memory_utils import load_config

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# def _ensure_cfg(cfg: Optional[dict]) -> dict:
#     """Return provided cfg or load default config if None."""
#     if cfg is not None:
#         return cfg
#     config_path = os.environ.get("MM_CONFIG_PATH") or os.path.join(PARENT_DIR, "config.yaml")
#     return load_config(config_path)

def frame_inspect_tool(
    database: A[NanoVectorDB, D("The database containing video metadata. Must be an instance of NanoVectorDB.")], 
    question: A[str, D("The specific detailed question to ask about the video content during the specified time ranges. No need to add time ranges in the question.")], 
    time_ranges_hhmmss: A[list[tuple], D("A list of tuples containing start and end times in HH:MM:SS format. If the time range is longer than 50 seconds, the function samples 50 evenly distributed frames.  Otherwise, it uses all frames within the specified range.")], 
    cfg: Optional[dict] = None,
) -> str:
    """
    Crop the video frames based on the time ranges and ask the model a detailed question about the cropped video clips.
    Returns:
        str: The model's response to the question. If no relevant content is found within the time range,
             returns an error message: "Error: Cannot find corresponding result in the given time range."
    """

    # cfg = _ensure_cfg(cfg)

    assert isinstance(database, NanoVectorDB), "Database must be an instance of NanoVectorDB"
    video_length_secs = convert_hhmmss_to_seconds(database.get_additional_data()['video_length'])
    video_meta = database.get_additional_data()
    video_file_root = video_meta['video_file_root']
    fps = video_meta.get('fps', cfg['VIDEO_FPS']) # config.VIDEO_FPS
    time_ranges_secs = []
    for time_range in time_ranges_hhmmss:
        start_secs = convert_hhmmss_to_seconds(time_range[0])
        end_secs = convert_hhmmss_to_seconds(time_range[1])
        if start_secs > video_length_secs:
            raise ValueError(f"One of start time {time_range[0]} exceeds video length {video_length_secs}")
        end_secs = min(end_secs, video_length_secs)
        time_ranges_secs.append((start_secs, end_secs))

    time_ranges_secs.sort(key=lambda x: x[0])  # Sort by start time
    # Calculate total time across all ranges
    total_time = sum(end - start for start, end in time_ranges_secs)
    
    # Maximum number of timepoints to sample
    max_timepoints = cfg['AOAI_TOOL_VLM_MAX_FRAME_NUM']
    timepoints = []

    assert total_time > 0 and max_timepoints > 0 

    # ① Uniformly sample on the flattened timeline
    #    endpoint=False ensures the last sample point < total_time
    offsets = np.linspace(
        0, total_time,
        num=max_timepoints,
        endpoint=False,
        dtype=float
    )

    # ② Calculate prefix sums for each segment, used to map offsets back to actual timestamps
    prefix_len = []          # (cumulative length, segment start, segment length)
    acc = 0
    for start, end in time_ranges_secs:
        seg_len   = end - start
        prefix_len.append((acc, start, seg_len))
        acc += seg_len

    # ③ Complete the mapping
    for off in offsets:
        # off = int(round(off))          # Ensure it is an integer
        for base, seg_start, seg_len in prefix_len:
            if off < base + seg_len:   # Find the corresponding segment
                timepoints.append(seg_start + (off - base))
                break
    # ④ Convert the sampled timestamps (seconds) to frame indices according to VIDEO_FPS
    max_frame_idx = int(video_length_secs * fps) - 1   # last valid frame index

    framepoints = [
        min(max(int(round(ts * fps)), 0), max_frame_idx)  # clamp to [0, max_frame_idx]
        for ts in timepoints
    ]
    framepoints = sorted(set(framepoints))[:max_timepoints]


    input_msgs = [
        {
            "role": "system",
            "content": "You are a helpful assistant to answer questions."
        },
        {
            "role": "user",
            "content": "Carefully watch the video frames. Pay attention to the cause and sequence of events, the detail and movement of objects and the action and pose of persons.\n\nBased on your observations, if you find content that can answer the question. If no relevant content is found within the given time range, return: `Error: Cannot find corresponding result in the given time range.`. \nQuestion: {question}\n",
        },
    ]

    input_msgs[1]['content'] = input_msgs[1]['content'].format(question=question)

    files = [
        os.path.join(video_file_root, "frames", f"frame_n{fn:06d}.jpg") for fn in framepoints
    ]
    msgs = call_openai_model_with_tools(
        messages=input_msgs,
        endpoints=cfg['vlm_client'],# AOAI_TOOL_VLM_ENDPOINT_LIST
        model_name=cfg['vlm_model'], # AOAI_TOOL_VLM_MODEL_NAME
        api_key=cfg['OPENAI_API_KEY'],
        image_paths=files,
        temperature=0,
        max_tokens=512,
    )
    if msgs is None:
        raise ValueError("No response from the model")
    return msgs["content"]

def clip_search_tool(
        database: A[NanoVectorDB, D("The database object that supports querying with embeddings.")],
        event_description: A[str, D("A textual description of the event to search for.")],
    top_k: A[int, D("The maximum number of top results to retrieve. Just use the default value.")] = 16,
    cfg: Optional[dict] = None,
) -> str:
    """
    Searches for events in a video clip database based on a given event description and retrieves the top-k most relevant video clip captions.

    Returns:
        str: A formatted string containing the concatenated captions of the searched video clip scripts.

    Notes:
        - This function utilizes the OpenAI Embedding Service to generate embeddings for the input text.
        - Use default values for `top_k` to limit the number of results returned.
    """
    # cfg = _ensure_cfg(cfg)

    assert isinstance(database, NanoVectorDB), "Database must be an instance of NanoVectorDB"
    query_emb = AzureOpenAIEmbeddingService.get_embeddings(
        endpoints=cfg['emb_client'],
        model_name=cfg['emb_model'],
        input_text=[event_description],
        api_key=cfg['emb_api_key'],
        use_local=True,
    )[0]['embedding']
    results = database.query(
        query_emb,
        top_k=top_k,
    )
    captions = [
        (data['time_start_secs'], data['caption'])
        for i, data in enumerate(results)
    ]
    captions = sorted(captions, key=lambda x: x[0])
    captions = "\n".join([cap[1] for cap in captions])
    return f"Here is the searched video clip scripts:\n\n" + captions


def global_browse_tool(
        database: A[NanoVectorDB, D("The database object that supports querying with embeddings.")],
        query: A[str, D("A textual description which will be used to search for relevant video clips in the database.")],
    cfg: Optional[dict] = None,
) -> str:
    """
    Analyzes a video database to answer a detailed question by first searching for relevant video clips and then generating a comprehensive answer based on their captions.

    Args:
        database (NanoVectorDB): The database object that supports querying with embeddings.
        query (str): A textual description or question to search for in the video.

    Returns:
        str: A JSON string containing the subject registry and the model's answer to the query based on the most relevant video clips.
    """
    # cfg = _ensure_cfg(cfg)

    # search related clips
    assert isinstance(database, NanoVectorDB), "Database must be an instance of NanoVectorDB"
    query_emb = AzureOpenAIEmbeddingService.get_embeddings(
        endpoints=cfg['emb_client'],
        model_name=cfg['emb_model'],
        input_text=[query],
        api_key=cfg['emb_dim'],
        use_local=True,
    )[0]['embedding']
    results = database.query(
        query_emb,
        top_k=cfg['GLOBAL_BROWSE_TOPK'],
    )
    captions = [
        (data['time_start_secs'], data['caption'])
        for i, data in enumerate(results)
    ]
    captions = sorted(captions, key=lambda x: x[0])
    captions = "\n".join([cap[1] for cap in captions])

    clip_captions = f"Here is the searched video clip scripts:\n\n" + captions

    # ask the question

    input_msgs = [
        {
            "role": "system",
            "content": "You are a knowledgeable assistant specializing in analyzing video content and providing detailed, insightful answers."
        },
        {
            "role": "user",
            "content": (
                "Below are descriptions of video clips, each with its corresponding timestamp. "
                "Carefully review the sequence of events, the details and movements of objects, and the actions and poses of people. "
                "Based on these observations, provide a thorough and specific answer to the following question, referencing key events and timestamps as appropriate.\n"
                "Question: {question}\n\n{clip_captions}"
            ),
        },
    ]

    input_msgs[1]['content'] = input_msgs[1]['content'].format(question=query, clip_captions=clip_captions)

    msgs = call_openai_model_with_tools(
        messages=input_msgs,
        endpoints=cfg['vlm_client'], # AOAI_TOOL_VLM_ENDPOINT_LIST
        model_name=cfg['vlm_model'], # AOAI_TOOL_VLM_MODEL_NAME
        api_key=cfg['OPENAI_API_KEY'],
        temperature=0,
        max_tokens=512,
    )
    if msgs is None:
        raise ValueError("No response from the model")

    subject_registry = database.get_additional_data()['subject_registry']
    return json.dumps({'subject_registry': subject_registry, 'query_related_event': msgs["content"]})


def convert_seconds_to_hhmmss(seconds):
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def convert_hhmmss_to_seconds(hhmmss):
    hhmmss = hhmmss.split('.')[0]
    parts = hhmmss.split(":")
    if len(parts) < 2:
        raise ValueError("Invalid time format. Expected HH:MM:SS.")
    elif len(parts) == 2:
        parts = ["00"] + parts
    hours, minutes, seconds = map(int, parts)
    return hours * 3600 + minutes * 60 + seconds

def is_covered(d,N):
    i=sorted((int(a),int(b))for a,b in(map(lambda x:x.split('_'),d)));c=0
    return all(s==c and not (c:=e) for s,e in i) and c==N

def init_single_video_db(video_caption_json_path, output_video_db_path, emb_dim, cfg: Optional[dict] = None):
    # cfg = _ensure_cfg(cfg)
    vdb = NanoVectorDB(emb_dim, storage_file=output_video_db_path)
    # with open(video_caption_json_path, "r") as f:
    #     captions = json.load(f)
    # subject_registry = captions.pop('subject_registry', captions.pop('character_registry', None))            
    # video_length = max([float(k.split("_")[1]) for k in captions.keys()])
    # if not is_covered(captions.keys(), video_length):
    #     error_msg = (f"Fail to build video database for video {video_caption_json_path.split("/")[-3]}. Get None video clip captions for some clips in the video.")
    #     raise ValueError(error_msg)
    # video_length = convert_seconds_to_hhmmss(video_length)
    if os.path.exists(output_video_db_path):
        print(f"Database {output_video_db_path} already exists.")
    else:
        cap2emb_list = preprocess_captions(video_caption_json_path, cfg=cfg)
        data = []
        for idx, (timestamp, cap, emb) in enumerate(cap2emb_list):
            t1 = convert_seconds_to_hhmmss(timestamp[0])
            t2 = convert_seconds_to_hhmmss(timestamp[1])
            prefix = f"[From {t1} to {t2} seconds]\n"
            data.append(
                {
                    "__vector__": np.array(emb),
                    "time_start_secs": timestamp[0],
                    "time_end_secs": timestamp[1],
                    "caption": prefix + cap['caption'],
                }
            )
        _ = vdb.upsert(data)
        with open(video_caption_json_path, "r") as f:
            captions = json.load(f)
        subject_registry = captions.pop('subject_registry', captions.pop('character_registry', None))          
        video_length = max([float(k.split("_")[1]) for k in captions.keys()])
        video_length = convert_seconds_to_hhmmss(video_length)
        addtional_data = {
            'subject_registry': subject_registry,
            'video_length': video_length,
            'video_file_root': os.path.dirname(os.path.dirname(video_caption_json_path)),
            'fps': cfg['VIDEO_FPS'],
        }
        vdb.store_additional_data(**addtional_data)
        vdb.save()
    return vdb

def preprocess_captions(caption_json_path, cfg: Optional[dict] = None):
    # cfg = _ensure_cfg(cfg)
    with open(caption_json_path, "r") as f:
        captions = json.load(f)
    scripts = []
    captions.pop('subject_registry', None)
    captions.pop('character_registry', None)
    for idx, (timestamp, cap_info) in enumerate(captions.items()):
        if cap_info.get('caption') is None or len(cap_info['caption']) == 0:
            print(f"Empty caption information for {timestamp} in {caption_json_path}")
            continue
        elif isinstance(cap_info['caption'], list):
            cap_info['caption'] = cap_info['caption'][0]
        elif not isinstance(cap_info['caption'], str):
            print(f"Invalid caption type for {cap_info['caption']}")
            cap_info['caption'] = str(cap_info['caption'])
        timestamp = list(map(float, timestamp.split("_")))
        scripts.append((timestamp, cap_info['caption'], cap_info))

    # batchify
    batch_size = 128
    batched_scripts = []
    print(f"Embedding {len(scripts)} captions...")
    for i in range(0, len(scripts), batch_size):
        batch = scripts[i:i+batch_size]
        batched_scripts.append(batch)
    cap2emb_list = []
    with multiprocessing.Pool(os.cpu_count() // 2) as pool:
        with tqdm(total=len(scripts), desc="Embedding captions...") as pbar:
            for result in pool.imap_unordered(
                functools.partial(single_batch_embedding_task, cfg=cfg),
                batched_scripts,
            ):
                cap2emb_list.extend(result)
                pbar.update(len(result))
    return cap2emb_list

def single_batch_embedding_task(data, cfg: Optional[dict] = None):
    # cfg = _ensure_cfg(cfg)
    timestamps, captions, cap_infos = map(list, (zip(*data)))
    embs = AzureOpenAIEmbeddingService.get_embeddings(
        endpoints=cfg['emb_client'],
        model_name=cfg['emb_model'],
        input_text=captions,
        # api_key=cfg['emb_api_key'],
        use_local=True,
    )
    max_tries = 3
    while embs is None or len(embs) != len(captions):
        max_tries -= 1
        if max_tries < 0:
            raise ValueError(f"Failed to get embeddings for {timestamps} {captions}")
        print(f"Error in embedding {timestamps} {captions} retrying...")
        embs = AzureOpenAIEmbeddingService.get_embeddings(
            endpoints=cfg['emb_client'],
            model_name=cfg['emb_model'],
            input_text=captions,
            # api_key=cfg['emb_api_key'],
            use_local=True,
        )
    return list(zip(timestamps, cap_infos, [d['embedding'] for d in embs]))

if __name__ == "__main__":
    # cfg = _ensure_cfg(None)

    # benchmark_metadata_path = "/home/xiaoyizhang/event_prediction_model/LVBench/data/video_info.meta.jsonl"
    # video_caption_folder = "/data/xiaoyizhang/LVBench/"
    # output_folder_name = "audio_segment_w_transcribe_0411"
    # video_db_folder = "./lvbench_vdb/"
    # os.makedirs(video_db_folder, exist_ok=True)
    # embedding_dim = cfg['AOAI_EMBEDDING_LARGE_DIM']

    # with open(benchmark_metadata_path, "r") as f:
    #     lines = f.readlines()

    # video_caption_json_path = "/gemini/space/linx/DeepVideoDiscovery/videomme_dvd/captions/videos_chunked_01/0ag_Qi5OEd0/captions.json"
    # vdb_dir = "/gemini/space/linx/DeepVideoDiscovery/videomme_dvd/vdb/videos_chunked_01/"
    # os.makedirs(vdb_dir, exist_ok=True)
    # output_video_db_path = os.path.join(vdb_dir, "0ag_Qi5OEd0_vdb.json")
    
    # embedding_dim = cfg['emb_dim']
    
    # video_db = init_single_video_db(video_caption_json_path, output_video_db_path, embedding_dim)
    # print("Done. VDB saved to:", output_video_db_path)

    caption_root = "/gemini/space/linx/DeepVideoDiscovery/m3-agent_dvd/captions"
    vdb_root     = "/gemini/space/linx/DeepVideoDiscovery/m3-agent_dvd/vdb/"

    # 只创建总的 vdb_root 目录
    os.makedirs(vdb_root, exist_ok=True)
    

    embedding_dim = cfg['emb_dim']

    for video_name in os.listdir(caption_root):
        video_caption_dir = os.path.join(caption_root, video_name)
        if not os.path.isdir(video_caption_dir):
            continue

        caption_json_path = os.path.join(video_caption_dir, "captions.json")
        if not os.path.exists(caption_json_path):
            # 没有 caption.json，则跳过并打印 video 名字，不创建任何 vdb 子目录
            print(f"Skip video {video_name}: captions.json not found.")
            continue

        # 只有在存在 caption.json 时才创建对应的 vdb 子目录
        video_vdb_dir = os.path.join(vdb_root, video_name)
        os.makedirs(video_vdb_dir, exist_ok=True)

        # 构造 vdb 文件路径，命名规则：[video_name]_vdb.json
        output_video_db_path = os.path.join(video_vdb_dir, f"{video_name}_vdb.json")

        print(f"Building VDB for video {video_name} ...")

        try:
            video_db = init_single_video_db(
                caption_json_path,
                output_video_db_path,
                embedding_dim,
                cfg=cfg,
            )
            print(f"Done. VDB saved to: {output_video_db_path}")
        except Exception as e:
            print(f"Error while building VDB for video {video_name}: {e}")