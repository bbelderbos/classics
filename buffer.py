import requests
from decouple import config

API_URL = "https://api.buffer.com"
HTTP_TIMEOUT = 30


class BufferError(RuntimeError):
    pass


ORGS_QUERY = "query { account { organizations { id } } }"

CHANNELS_QUERY = """query Channels($org: OrganizationId!) {
  channels(input: { organizationId: $org }) {
    id
    displayName
    service
    isQueuePaused
  }
}"""

CREATE_POST = """mutation Create($input: CreatePostInput!) {
  createPost(input: $input) {
    ... on PostActionSuccess { post { id } }
    ... on MutationError { message }
  }
}"""


def build_post_input(
    text: str,
    channel_id: str,
    image_url: str | None = None,
    due_at: str | None = None,
    metadata: dict | None = None,
) -> dict:
    post: dict = {"text": text, "channelId": channel_id, "schedulingType": "automatic"}
    if due_at is not None:
        post["mode"] = "customScheduled"
        post["dueAt"] = due_at
    else:
        post["mode"] = "addToQueue"
    if image_url is not None:
        post["assets"] = [{"image": {"url": image_url}}]
    if metadata is not None:
        post["metadata"] = metadata
    return post


def metadata_for(channel: dict) -> dict | None:
    service = channel["service"]
    # Instagram rejects a post without a type; a quote card is a normal feed post
    if service == "instagram":
        return {"instagram": {"type": "post", "shouldShareToFeed": True}}
    return None


def parse_create_result(payload: dict) -> str:
    node = payload["data"]["createPost"]
    if node.get("message") is not None:
        raise BufferError(node["message"])
    post = node.get("post")
    if post is None:
        raise BufferError("unexpected createPost response")
    return post["id"]


def _graphql(query: str, variables: dict | None = None) -> dict:
    response = requests.post(
        API_URL,
        json={"query": query, "variables": variables or {}},
        headers={"Authorization": f"Bearer {config('BUFFER_TOKEN')}"},
        timeout=HTTP_TIMEOUT,
    )
    try:
        body = response.json()
    except ValueError:
        body = None
    # GraphQL validation errors come back as HTTP 400 with the detail in the body,
    # so read the body before raise_for_status or the real message is lost
    if body and body.get("errors") is not None:
        raise BufferError(body["errors"][0].get("message", "graphql error"))
    response.raise_for_status()
    if body is None:
        raise BufferError(f"non-JSON response ({response.status_code})")
    return body


def organizations() -> list[str]:
    body = _graphql(ORGS_QUERY)
    return [o["id"] for o in body["data"]["account"]["organizations"]]


def first_organization() -> str:
    orgs = organizations()
    if not orgs:
        raise BufferError("no Buffer organizations for this token")
    return orgs[0]


def channels(org_id: str) -> list[dict]:
    return _graphql(CHANNELS_QUERY, {"org": org_id})["data"]["channels"]


def create_post(
    text: str,
    channel_id: str,
    image_url: str | None = None,
    due_at: str | None = None,
    metadata: dict | None = None,
) -> str:
    payload = _graphql(
        CREATE_POST,
        {"input": build_post_input(text, channel_id, image_url, due_at, metadata)},
    )
    return parse_create_result(payload)


def queue(
    text: str,
    image_url: str | None,
    channels: list[dict],
    due_at: str | None = None,
) -> dict[str, dict]:
    # keep going when one channel fails, so a single bad network can't block the rest
    posted: dict[str, str] = {}
    errors: dict[str, str] = {}
    for channel in channels:
        try:
            posted[channel["id"]] = create_post(
                text, channel["id"], image_url, due_at, metadata_for(channel)
            )
        except BufferError as exc:
            errors[channel["id"]] = str(exc)
    return {"posted": posted, "errors": errors}
