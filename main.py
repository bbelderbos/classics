import sys
from pathlib import Path
from typing import NamedTuple
import requests

BOOKS_DIR = Path("books")

SEARCH_URL = "https://gutendex.com/books/?search="
BOOK_URL = "https://gutendex.com/books/"


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


def get_book_text(book_id: int) -> str:
    """Fetches the plain text content of a book by its Gutenberg ID."""
    # 1. Query the specific book endpoint to get the formats
    url = f"{BOOK_URL}{book_id}"
    response = requests.get(url)
    response.raise_for_status()
    book_data = response.json()

    # 2. Extract the plain text URL (checking utf-8 first, then fallback)
    formats = book_data.get("formats", {})
    text_url = formats.get("text/plain; charset=utf-8") or formats.get("text/plain")

    if not text_url:
        raise ValueError(f"No plain text format found for book ID {book_id}")

    # 3. Fetch the actual text file
    text_response = requests.get(text_url)
    text_response.raise_for_status()

    return text_response.text


def main(*args: str) -> None:
    if not args:
        print("Please provide a search term or a book ID.")
        return

    # If the first argument is a number, treat it as a request to fetch that book
    if args[0].isdigit():
        book_id = int(args[0])
        try:
            print(f"Fetching text for book ID {book_id}...")
            full_text = get_book_text(book_id)

            BOOKS_DIR.mkdir(exist_ok=True)
            path = BOOKS_DIR / f"{book_id}.txt"
            path.write_text(full_text, encoding="utf-8")
            print(f"Saved book {book_id} to {path}")

        except Exception as e:
            print(f"Error: {e}")

    else:
        # Otherwise, run your original search logic
        search_term = " ".join(args)
        books = search_book(search_term)
        for book in books:
            print(f"Title: {book.title}")
            print(f"Authors: {', '.join(book.authors)}")
            print(
                f"id: {book.id}, download_count: {book.download_count}, languages: {', '.join(book.languages)}"
            )
            print("-" * 40)


if __name__ == "__main__":
    main(*sys.argv[1:])
