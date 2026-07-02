import pytest

from buffer import BufferError, build_post_input, parse_create_result


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
