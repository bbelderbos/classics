import sys
from typing import NamedTuple

import requests

SEARCH_URL = "https://gutendex.com/books/?search="


class Book(NamedTuple):
    id: int
    title: str
    authors: list[str]
    summary: str
    languages: list[str]
    download_count: int


def search_book(title: str) -> list[Book]:
    search_url = SEARCH_URL + title
    response = requests.get(search_url).json()

    return [
        Book(
            id=int(book["id"]),
            title=book["title"],
            authors=[entry["name"] for entry in book["authors"]],
            summary=book["summaries"][0] if book["summaries"] else "",
            languages=book["languages"],
            download_count=int(book["download_count"]),
        )
        for book in response["results"]
    ]


def main(*args: str) -> None:
    search_term = " ".join(args)
    books = search_book(search_term)
    for book in books:
        print(f"Title: {book.title}")
        print(f"Authors: {', '.join(author for author in book.authors)}")
        print(
            f"id: {book.id}, download_count: {book.download_count}, languages: {', '.join(book.languages)}"
        )
        print("-" * 40)


if __name__ == "__main__":
    main(*sys.argv[1:])
