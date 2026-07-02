from publish import card_url


def test_card_url_joins_parts():
    assert (
        card_url("https://askthecanon.com/cards", "on-anger", "card-1.png")
        == "https://askthecanon.com/cards/on-anger/card-1.png"
    )


def test_card_url_strips_trailing_slash_on_base():
    assert (
        card_url("https://x.com/cards/", "s", "c.png") == "https://x.com/cards/s/c.png"
    )
