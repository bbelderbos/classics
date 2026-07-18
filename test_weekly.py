from datetime import UTC, datetime

import pytest

import weekly
from weekly import (
    Candidate,
    Card,
    build_card,
    build_prompt,
    compose_text,
    fresh_questions,
    normalize,
    on_cooldown,
    parse_suggestions,
    record_posted,
    verify_quote,
)
from main import Passage

NOW = datetime(2026, 7, 1, tzinfo=UTC)


def test_normalize_collapses_case_and_whitespace():
    assert normalize("  How  Should\tI  Cope? ") == "how should i cope?"


def test_on_cooldown_true_within_window():
    posted = {"how do i cope?": datetime(2026, 6, 20, tzinfo=UTC).isoformat()}
    assert on_cooldown(posted, "How do I cope?", 30, NOW)


def test_on_cooldown_false_past_window():
    posted = {"how do i cope?": datetime(2026, 5, 1, tzinfo=UTC).isoformat()}
    assert not on_cooldown(posted, "how do i cope?", 30, NOW)


def test_on_cooldown_false_when_never_posted():
    assert not on_cooldown({}, "new question", 30, NOW)


def test_fresh_questions_filters_recent():
    posted = {"a": datetime(2026, 6, 28, tzinfo=UTC).isoformat()}
    assert fresh_questions(["a", "b"], posted, 30, NOW) == ["b"]


def test_record_posted_writes_normalized_key():
    posted = record_posted({}, "  Let Go  of Anger ", NOW)
    assert posted == {"let go of anger": NOW.isoformat()}


def test_parse_suggestions_extracts_fenced_json():
    out = 'sure!\n```json\n[{"index": 0, "why": "x"}]\n```\n'
    assert parse_suggestions(out) == [{"index": 0, "why": "x"}]


def test_parse_suggestions_raises_without_array():
    with pytest.raises(ValueError):
        parse_suggestions("no json here")


def test_parse_suggestions_ignores_trailing_prose():
    assert parse_suggestions('[{"index": 0}]\n\nHope that helps!') == [{"index": 0}]


def test_parse_suggestions_stops_at_first_array():
    # greedy regex would swallow the trailing "[2, 3]" and fail to parse
    assert parse_suggestions('Here:\n[{"index": 1}]\nand ignore [2, 3]') == [
        {"index": 1}
    ]


def test_build_prompt_includes_question_and_indexed_passage():
    passage = Passage("Meditations", "Aurelius, Marcus", "BOOK II", "the body text")
    prompt = build_prompt("how to let go?", [Candidate(passage, 0.71)])
    assert "how to let go?" in prompt
    assert "[0]" in prompt
    assert "Meditations" in prompt
    assert "the body text" in prompt


def test_verify_quote_matches_ignoring_whitespace():
    assert verify_quote("let go\n of anger", "you must  let go of anger today")


def test_verify_quote_rejects_paraphrase():
    assert not verify_quote("release your rage", "you must let go of anger")


def test_verify_quote_rejects_empty():
    assert not verify_quote("   ", "any passage text")


def test_build_card_keeps_verbatim_quote():
    passage = Passage(
        "Meditations", "Aurelius, Marcus", "", "the best revenge is not to be like that"
    )
    card = build_card(
        "how to respond?",
        Candidate(passage, 0.7),
        {
            "index": 0,
            "quote": "the best revenge is not to be like that",
            "caption": "cap",
            "hashtags": ["#stoic"],
        },
    )
    assert card.verbatim
    assert card.text == "the best revenge is not to be like that"
    assert card.author == "Aurelius, Marcus"
    assert card.caption == "cap"


def test_build_card_falls_back_when_not_verbatim(monkeypatch):
    monkeypatch.setattr(weekly, "best_excerpt", lambda text, query: "EXCERPT")
    passage = Passage(
        "Meditations", "Aurelius, Marcus", "", "some other words entirely"
    )
    card = build_card(
        "q", Candidate(passage, 0.7), {"index": 0, "quote": "a line not in the source"}
    )
    assert not card.verbatim
    assert card.text == "EXCERPT"


def _card(caption="", author="Aurelius, Marcus"):
    return Card("t", author, "ti", "q", caption, True)


def test_compose_text_appends_classics_and_author():
    assert compose_text(_card("A hook")) == "A hook\n\n#classics #Aurelius"


def test_compose_text_omits_empty_caption():
    assert compose_text(_card("", author="Epictetus")) == "#classics #Epictetus"


def test_author_hashtag_uses_surname():
    assert weekly.author_hashtag("Aurelius, Marcus") == "#Aurelius"


def test_author_hashtag_single_name():
    assert weekly.author_hashtag("Epictetus") == "#Epictetus"


def test_author_hashtag_multi_comma_keeps_surname():
    assert weekly.author_hashtag("Augustine, of Hippo, Saint") == "#Augustine"


def test_author_hashtag_drops_parenthetical():
    assert (
        weekly.author_hashtag("Fitzgerald, F. Scott (Francis Scott)") == "#Fitzgerald"
    )
