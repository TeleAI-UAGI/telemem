import json
import re
from pydantic import ValidationError
from typing import TypedDict
from ...models.utils import Promise
from ...models.database import GeneralBlob, UserProfile
from ...models.blob import OpenAICompatibleMessage
from ...models.response import CODE, IdData, IdsData, UserProfilesData
from ...utils import truncate_string, find_list_int_or_none
from ...env import TRACE_LOG, CONFIG
from ...prompts import pick_related_profiles as pick_prompt
from ...llms import llm_complete


class FilterProfilesResult(TypedDict):
    reason: str | None
    profiles: list[UserProfile]


JSON_BODY_REGEX = re.compile(r"({[\s\S]*})")


def try_json_reason(content: str) -> str | None:
    try:
        return json.loads(JSON_BODY_REGEX.search(content).group(1))["reason"]
    except Exception:
        return None


async def filter_profiles_with_chats(
    user_id: str,
    project_id: str,
    profiles: UserProfilesData,
    chats: list[OpenAICompatibleMessage],
    only_topics: list[str] | None = None,
    max_value_token_size: int = 10,
    max_previous_chats: int = 4,
    max_filter_num: int = 10,
) -> Promise[FilterProfilesResult]:
    """Filter profiles with chats"""
    if not len(chats) or not len(profiles.profiles):
        return Promise.reject(CODE.BAD_REQUEST, "No chats or profiles to filter")
    chats = chats[-(max_previous_chats + 1) :]
    if only_topics:
        only_topics = [t.strip() for t in only_topics]
        only_topics = set(only_topics)

    topics_index = [
        {
            "index": i,
            "topic": p.attributes["topic"],
            "sub_topic": p.attributes["sub_topic"],
            "content": truncate_string(p.content, max_value_token_size),
        }
        for i, p in enumerate(profiles.profiles)
        if only_topics is None or p.attributes["topic"].strip() in only_topics
    ]

    topics_index = sorted(topics_index, key=lambda x: (x["topic"], x["sub_topic"]))
    system_prompt = pick_prompt.get_prompt(max_num=max_filter_num)
    input_prompt = pick_prompt.get_input(chats, topics_index)
    r = await llm_complete(
        project_id,
        input_prompt,
        system_prompt=system_prompt,
        temperature=0.2,  # precise
        model=CONFIG.summary_llm_model,
        **pick_prompt.get_kwargs(),
    )
    if not r.ok():
        TRACE_LOG.error(
            project_id,
            user_id,
            f"Failed to pick related profiles: {r.msg()}",
        )
        return r
    found_ids = find_list_int_or_none(r.data())
    reason = try_json_reason(r.data())
    if found_ids is None:
        TRACE_LOG.error(
            project_id,
            user_id,
            f"Failed to pick related profiles: {r.data()}",
        )
        return Promise.reject(
            CODE.INTERNAL_SERVER_ERROR, "Failed to pick related profiles"
        )
    ids = [i for i in found_ids if i < len(topics_index)]
    profiles = [profiles.profiles[topics_index[i]["index"]] for i in ids]
    TRACE_LOG.info(
        project_id,
        user_id,
        f"Filter profiles with chats: {reason}, {found_ids}",
    )
    return Promise.resolve({"reason": reason, "profiles": profiles})
