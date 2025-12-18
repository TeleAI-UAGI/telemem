from sqlalchemy import cast, String, func, desc
from ..models.database import Project, User, UserProfile, UserEvent
from ..models.utils import Promise, CODE
from ..models.response import IdData, ProfileConfigData, ProjectUsersData, DailyUsage
from ..connectors import Session
from ..env import ProfileConfig, TelemetryKeyName
from ..telemetry.capture_key import get_int_key, date_past_key


async def get_project_secret(project_id: str) -> Promise[str]:
    with Session() as session:
        p = (
            session.query(Project)
            .filter(Project.project_id == project_id)
            .one_or_none()
        )
        if not p:
            return Promise.reject(CODE.NOT_FOUND, "Project not found")
        return Promise.resolve(p.project_secret)


async def get_project_status(project_id: str) -> Promise[str]:
    with Session() as session:
        p = (
            session.query(Project.status)
            .filter(Project.project_id == project_id)
            .one_or_none()
        )
        if not p:
            return Promise.reject(CODE.NOT_FOUND, "Project not found")
        return Promise.resolve(p.status)


async def get_project_profile_config(project_id: str) -> Promise[ProfileConfig]:
    with Session() as session:
        p = (
            session.query(Project.profile_config)
            .filter(Project.project_id == project_id)
            .one_or_none()
        )
        if not p:
            return Promise.reject(CODE.NOT_FOUND, "Project not found")
        if not p.profile_config:
            return Promise.resolve(ProfileConfig())
        p_parse = ProfileConfig.load_config_string(p.profile_config)
    return Promise.resolve(p_parse)


async def update_project_profile_config(
    project_id: str, profile_config: str | None
) -> Promise[None]:
    with Session() as session:
        p = (
            session.query(Project)
            .filter(Project.project_id == project_id)
            .one_or_none()
        )
        if not p:
            return Promise.reject(CODE.NOT_FOUND, "Project not found")
        p.profile_config = profile_config
        session.commit()
    return Promise.resolve(None)


async def get_project_profile_config_string(
    project_id: str,
) -> Promise[ProfileConfigData]:
    with Session() as session:
        p = (
            session.query(Project.profile_config)
            .filter(Project.project_id == project_id)
            .one_or_none()
        )
        if not p:
            return Promise.reject(CODE.NOT_FOUND, "Project not found")
        return Promise.resolve(ProfileConfigData(profile_config=p.profile_config or ""))


async def get_project_users(
    project_id: str,
    search: str = "",
    limit: int = 10,
    offset: int = 0,
    order_by: str = "updated_at",
    order_desc: bool = True,
) -> Promise[ProjectUsersData]:
    with Session() as session:
        profile_subq = (
            session.query(
                UserProfile.user_id.label("user_id"),
                func.count(UserProfile.id).label("profile_count"),
            )
            .filter(UserProfile.project_id == project_id)
            .group_by(UserProfile.user_id)
            .subquery()
        )

        event_subq = (
            session.query(
                UserEvent.user_id.label("user_id"),
                func.count(UserEvent.id).label("event_count"),
            )
            .filter(UserEvent.project_id == project_id)
            .group_by(UserEvent.user_id)
            .subquery()
        )

        query = (
            session.query(
                User,
                func.coalesce(profile_subq.c.profile_count, 0).label("profile_count"),
                func.coalesce(event_subq.c.event_count, 0).label("event_count"),
            )
            .filter(User.project_id == project_id)
            .filter(cast(User.id, String).like(f"%{search}%"))
            .outerjoin(profile_subq, profile_subq.c.user_id == User.id)
            .outerjoin(event_subq, event_subq.c.user_id == User.id)
        )

        if order_by == "profile_count":
            query = query.order_by(
                desc("profile_count") if order_desc else "profile_count"
            )
        elif order_by == "event_count":
            query = query.order_by(desc("event_count") if order_desc else "event_count")
        else:
            query = query.order_by(
                desc(User.updated_at) if order_desc else User.updated_at
            )

        count = (
            session.query(func.count())
            .filter(User.project_id == project_id)
            .filter(cast(User.id, String).like(f"%{search}%"))
            .scalar()
        )

        users_with_counts = query.limit(limit).offset(offset).all()

        user_dicts = []
        for user, profile_count, event_count in users_with_counts:
            user_data = user.__dict__.copy()
            user_data.pop("_sa_instance_state", None)
            user_data["profile_count"] = profile_count
            user_data["event_count"] = event_count
            user_dicts.append(user_data)

        return Promise.resolve(ProjectUsersData(users=user_dicts, count=count))


async def get_project_usage(
    project_id: str, last_days: int = 7
) -> Promise[list[DailyUsage]]:
    query_dates = [date_past_key(i) for i in range(last_days)]
    results = []
    for qd in query_dates:
        total_insert = await get_int_key(
            TelemetryKeyName.insert_blob_request, project_id, use_date=qd
        )
        total_success_insert = await get_int_key(
            TelemetryKeyName.insert_blob_success_request, project_id, use_date=qd
        )
        total_input_token = await get_int_key(
            TelemetryKeyName.llm_input_tokens, project_id, use_date=qd
        )
        total_output_token = await get_int_key(
            TelemetryKeyName.llm_output_tokens, project_id, use_date=qd
        )
        results.append(
            DailyUsage(
                date=qd,
                total_insert=total_insert,
                total_success_insert=total_success_insert,
                total_input_token=total_input_token,
                total_output_token=total_output_token,
            )
        )
    return Promise.resolve(results)
