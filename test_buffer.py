import pytest

import buffer
from buffer import BufferError, build_post_input, parse_create_result


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise buffer.requests.HTTPError(str(self.status_code))


def test_build_post_input_defaults_to_queue():
    got = build_post_input("hi", "chan1")
    assert got == {
        "text": "hi",
        "channelId": "chan1",
        "schedulingType": "automatic",
        "mode": "addToQueue",
    }


def test_build_post_input_with_image_and_schedule():
    got = build_post_input("hi", "chan1", "https://x/card.png", "2026-07-10T09:00:00Z")
    assert got["mode"] == "customScheduled"
    assert got["dueAt"] == "2026-07-10T09:00:00Z"
    assert got["assets"] == [{"image": {"url": "https://x/card.png"}}]


def test_parse_create_result_returns_post_id():
    payload = {"data": {"createPost": {"post": {"id": "post_42"}}}}
    assert parse_create_result(payload) == "post_42"


def test_parse_create_result_raises_on_mutation_error():
    payload = {"data": {"createPost": {"message": "channel disconnected"}}}
    with pytest.raises(BufferError, match="channel disconnected"):
        parse_create_result(payload)


def test_graphql_surfaces_error_body_on_400(monkeypatch):
    monkeypatch.setattr(buffer, "config", lambda *a, **k: "token")
    errors = {"errors": [{"message": "expected type OrganizationId!"}]}
    monkeypatch.setattr(
        buffer.requests, "post", lambda *a, **k: _FakeResponse(errors, status=400)
    )
    with pytest.raises(BufferError, match="OrganizationId"):
        buffer._graphql("query {}")
