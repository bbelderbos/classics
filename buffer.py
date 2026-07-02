import requests
from decouple import config

API_URL = "https://api.buffer.com"
HTTP_TIMEOUT = 30


class BufferError(RuntimeError):
    pass


ORGS_QUERY = "query { account { organizations { id } } }"

CHANNELS_QUERY = """query Channels($org: String!) {
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
) -> dict:
    post: dict = {"text": text, "channelId": channel_id, "schedulingType": "automatic"}
    if due_at:
        post["mode"] = "customScheduled"
        post["dueAt"] = due_at
    else:
        post["mode"] = "addToQueue"
    if image_url:
        post["assets"] = [{"image": {"url": image_url}}]
    return post


def parse_create_result(payload: dict) -> str:
    node = payload["data"]["createPost"]
    if node.get("message"):
        raise BufferError(node["message"])
    post = node.get("post")
    if not post:
        raise BufferError("unexpected createPost response")
    return post["id"]


def _graphql(query: str, variables: dict | None = None) -> dict:
    response = requests.post(
        API_URL,
        json={"query": query, "variables": variables or {}},
        headers={"Authorization": f"Bearer {config('BUFFER_TOKEN')}"},
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    body = response.json()
    if body.get("errors"):
        raise BufferError(body["errors"][0].get("message", "graphql error"))
    return body


def organizations() -> list[str]:
    body = _graphql(ORGS_QUERY)
    return [o["id"] for o in body["data"]["account"]["organizations"]]


def channels(org_id: str) -> list[dict]:
    return _graphql(CHANNELS_QUERY, {"org": org_id})["data"]["channels"]


def create_post(
    text: str,
    channel_id: str,
    image_url: str | None = None,
    due_at: str | None = None,
) -> str:
    payload = _graphql(
        CREATE_POST, {"input": build_post_input(text, channel_id, image_url, due_at)}
    )
    return parse_create_result(payload)


def queue(
    text: str,
    image_url: str | None,
    channel_ids: list[str],
    due_at: str | None = None,
) -> dict[str, str]:
    return {cid: create_post(text, cid, image_url, due_at) for cid in channel_ids}
