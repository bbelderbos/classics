import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NamedTuple

from rich.console import Console

import buffer
from main import (
    Passage,
    best_excerpt,
    load_library,
    reflow,
    search_passages,
)
from publish import publish_cards
from render import render_cards

console = Console()

POSTED_FILE = Path(__file__).parent / "posted.json"
CARDS_DIR = Path(__file__).parent / "cards"
COOLDOWN_DAYS = 30

# representative fallback questions for validating the pipeline offline, until the
# source is switched to real user searches (top_searches from /api/stats)
SEED_QUESTIONS = [
    "how should I deal with people who wrong me?",
    "how do I become more confident?",
    "how do I find meaning in my work?",
    "how do I cope with the fear of death?",
    "how do I let go of anger?",
    "what makes a friendship last?",
]


def _now() -> datetime:
    return datetime.now(UTC)


def normalize(question: str) -> str:
    return " ".join(question.lower().split())


def load_posted(path: Path = POSTED_FILE) -> dict[str, str]:
    return json.loads(path.read_text()) if path.exists() else {}


def save_posted(posted: dict[str, str], path: Path = POSTED_FILE) -> None:
    path.write_text(json.dumps(posted, indent=2, sort_keys=True))


def on_cooldown(
    posted: dict[str, str],
    question: str,
    cooldown_days: int,
    now: datetime | None = None,
) -> bool:
    last = posted.get(normalize(question))
    if not last:
        return False
    return (now or _now()) - datetime.fromisoformat(last) < timedelta(
        days=cooldown_days
    )


def fresh_questions(
    candidates: list[str],
    posted: dict[str, str],
    cooldown_days: int,
    now: datetime | None = None,
) -> list[str]:
    return [q for q in candidates if not on_cooldown(posted, q, cooldown_days, now)]


def record_posted(
    posted: dict[str, str], question: str, now: datetime | None = None
) -> dict[str, str]:
    posted[normalize(question)] = (now or _now()).isoformat()
    return posted


class Candidate(NamedTuple):
    passage: Passage
    score: float


def gather(
    question: str, passages: list[Passage], vectors, k: int = 5
) -> list[Candidate]:
    ranked = search_passages(question, passages, vectors, k)
    return [Candidate(passages[i], score) for i, score in ranked]


PROMPT_TEMPLATE = """You are a growth-focused social media editor for askthecanon.com, a \
semantic search engine over public-domain classics. We turn a real reader question into \
shareable quote cards for X, LinkedIn, Threads, Pinterest, and Instagram.

Question of the week: "{question}"

Below are the full candidate passages the engine surfaced, each with an index. For the best \
ones, pull the single verbatim line or two that would work as a standalone social quote card — \
timeless, punchy, emotionally resonant, and readable without the question for context. Copy the \
words exactly from the passage; never paraphrase.

{candidates}

Return ONLY a JSON array, best first, each item:
{{"index": <int>, "quote": "<verbatim words for the card, ~12-40 words, no surrounding quotation marks>", \
"why": "<one line on why it lands>", "caption": "<hook caption, no hashtags>", \
"hashtags": ["<3-5 tags>"]}}"""


def build_prompt(question: str, candidates: list[Candidate]) -> str:
    blocks = [
        f"[{i}] {c.passage.cite()} (score {c.score:.2f})\n{reflow(c.passage.text)}"
        for i, c in enumerate(candidates)
    ]
    return PROMPT_TEMPLATE.format(question=question, candidates="\n\n".join(blocks))


def run_claude(prompt: str) -> str:
    result = subprocess.run(
        ["claude", "-p"], input=prompt, text=True, capture_output=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "claude -p failed")
    return result.stdout


def parse_suggestions(output: str) -> list[dict]:
    start = output.find("[")
    if start == -1:
        raise ValueError("no JSON array found in claude output")
    try:
        result, _ = json.JSONDecoder().raw_decode(output[start:])
    except json.JSONDecodeError as e:
        raise ValueError(f"could not parse JSON array from claude output: {e}") from e
    return result


class Card(NamedTuple):
    text: str  # verbatim quote rendered on the image
    author: str
    title: str
    query: str
    caption: str
    hashtags: list[str]
    verbatim: bool  # False when we fell back to best_excerpt


def _squish(text: str) -> str:
    return " ".join(text.split())


def verify_quote(quote: str, passage_text: str) -> bool:
    return bool(quote.strip()) and _squish(quote) in _squish(passage_text)


def build_card(question: str, candidate: Candidate, suggestion: dict) -> Card:
    passage = candidate.passage
    quote = suggestion.get("quote", "").strip()
    verbatim = verify_quote(quote, passage.text)
    if not verbatim:
        quote = best_excerpt(passage.text, question)
    return Card(
        text=quote,
        author=passage.author,
        title=passage.title,
        query=question,
        caption=suggestion.get("caption", ""),
        hashtags=suggestion.get("hashtags", []),
        verbatim=verbatim,
    )


def choose(
    question: str, candidates: list[Candidate], suggestions: list[dict]
) -> list[Card]:
    rows = [
        (build_card(question, candidates[i], s), candidates[i].passage.cite(), s)
        for s in suggestions
        if isinstance(i := s.get("index", -1), int) and 0 <= i < len(candidates)
    ]
    console.print("\n[bold]Claude's ranking[/]\n")
    for rank, (card, cite, s) in enumerate(rows, 1):
        flag = "" if card.verbatim else " [yellow](fell back to excerpt)[/]"
        console.print(f"  [bold cyan]{rank}[/]  [dim]{cite}[/]{flag}")
        console.print(f"      [dim]on the card:[/] [italic]“{card.text}”[/]")
        console.print(f"      [dim]why:[/] {s.get('why', '')}")
        console.print("      [dim]post body:[/]")
        for line in compose_text(card).splitlines():
            console.print(f"        [green]{line}[/]" if line else "")
        console.print()

    picks = input("pick numbers to keep (comma-separated, enter to skip) > ").strip()
    chosen = []
    for token in picks.split(","):
        if token.strip().isdigit():
            rank = int(token.strip())
            if 1 <= rank <= len(rows):
                chosen.append(rows[rank - 1][0])
    return chosen


def render_chosen(question: str, cards: list[Card]) -> list[Path]:
    dicts = [
        {"text": c.text, "author": c.author, "title": c.title, "query": c.query}
        for c in cards
    ]
    return render_cards(dicts, CARDS_DIR / (_slug(question)))


def _slug(text: str) -> str:
    base = re.sub(r"[^\w]+", "-", text.lower()).strip("-")
    return base or "card"


def compose_text(card: Card) -> str:
    parts = [card.caption.strip(), " ".join(card.hashtags)]
    return "\n\n".join(p for p in parts if p)


def publish_and_queue(
    question: str, cards: list[Card], paths: list[Path], services: str
) -> None:
    urls = publish_cards(_slug(question), paths) if paths else [None] * len(cards)
    available = buffer.channels(buffer.organizations()[0])
    if services:
        wanted = {s.strip() for s in services.split(",")}
        available = [c for c in available if c["service"] in wanted]
    targets = [c for c in available if not c["isQueuePaused"]]
    for card, url in zip(cards, urls):
        result = buffer.queue(compose_text(card), url, targets)
        console.print(
            f"[green]queued[/] {len(result['posted'])} channel(s): “{card.text[:48]}…”"
        )
        for channel_id, message in result["errors"].items():
            service = next(
                (t["service"] for t in targets if t["id"] == channel_id), channel_id
            )
            console.print(f"  [red]{service} failed:[/] {message}")


def pick_questions(args: argparse.Namespace) -> list[str]:
    source = [args.question] if args.question else SEED_QUESTIONS
    posted = load_posted()
    fresh = fresh_questions(source, posted, args.cooldown_days)
    if not fresh:
        console.print("[yellow]all candidate questions are on cooldown[/]")
    return fresh[: args.limit]


def run(args: argparse.Namespace) -> None:
    questions = pick_questions(args)
    if not questions:
        return
    passages, vectors = load_library()
    posted = load_posted()

    for question in questions:
        candidates = gather(question, passages, vectors, args.k)
        if not candidates:
            console.print(f'[yellow]nothing strong enough for[/] "{question}"')
            continue
        prompt = build_prompt(question, candidates)
        if args.dry_run:
            console.print(prompt)
            continue
        try:
            suggestions = parse_suggestions(run_claude(prompt))
        except (RuntimeError, ValueError) as e:
            console.print(f"[red]claude step failed:[/] {e}")
            continue
        chosen = choose(question, candidates, suggestions)
        if not chosen:
            continue
        paths: list[Path] = []
        if not args.no_images:
            paths = render_chosen(question, chosen)
            for path in paths:
                console.print(f"[green]rendered[/] {path}")
        if args.to_buffer:
            publish_and_queue(question, chosen, paths, args.channels)
        if not args.no_record:
            record_posted(posted, question)
            save_posted(posted)
            console.print(f"[green]recorded[/] question ({len(chosen)} card(s))")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Turn reader questions into ranked social quote cards."
    )
    parser.add_argument(
        "question", nargs="?", help="one question (default: cycle the seed list)"
    )
    parser.add_argument("-k", type=int, default=5, help="passages to rank per question")
    parser.add_argument(
        "--limit", type=int, default=1, help="how many fresh questions to process"
    )
    parser.add_argument(
        "--cooldown-days",
        type=int,
        default=COOLDOWN_DAYS,
        help="skip questions posted within this many days",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the prompt, skip claude"
    )
    parser.add_argument(
        "--no-record", action="store_true", help="don't write to the posted cache"
    )
    parser.add_argument(
        "--no-images", action="store_true", help="skip Playwright card rendering"
    )
    parser.add_argument(
        "--to-buffer",
        action="store_true",
        help="publish cards to askthecanon and queue posts to Buffer",
    )
    parser.add_argument(
        "--channels",
        default="",
        help="comma-separated services to queue (default: all connected)",
    )
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
