# Social quote-card pipeline

Turn a reader question into ranked, on-brand quote cards and queue them to Buffer.
One weekly, human-in-the-loop job driven by `weekly.py`.

## Flow

```
question → search_passages → claude -p ranks → you pick → render PNGs
         → rsync to askthecanon → queue to Buffer → record cooldown
```

- **Retrieval** reuses the search engine directly (`main.search_passages`), no HTTP.
- **Ranking** pipes the full passages to `claude -p`; Claude returns a verbatim quote
  per card. Non-verbatim quotes are rejected and fall back to `best_excerpt`, so nothing
  misquotes the source.
- **Cards** are rendered by driving the site's own `renderImage()` (`static/index.html`)
  headless with Playwright — the browser card generator is the single source of truth.
- **Images** are hosted from `askthecanon.com/cards/...`. Buffer fetches each by URL when
  the post is queued and re-hosts its own copy, so `cards/` is a staging area, not a
  gallery — prune it whenever.
- **Cooldown** (`posted.json`) skips any question posted within `--cooldown-days` (30).

## Setup

Copy `.env.example` to `.env` and fill in:

| Key | Purpose |
|-----|---------|
| `BUFFER_TOKEN` | Buffer → Settings → API |
| `CARDS_RSYNC_DEST` | rsync target, e.g. `root@<ip>:/root/classics/cards` |
| `CARDS_BASE_URL` | public base, e.g. `https://askthecanon.com/cards` |
| `SSH_KEY` | optional ssh key for the rsync |

The `/cards` static mount lives in `web.py`, so deploy it once on the droplet
(`git pull && systemctl restart classics`). Set your Buffer posting schedule (e.g.
2 slots/day) in the Buffer UI — `addToQueue` fills those slots.

## Usage

```bash
# rank + pick + render locally, nothing leaves the machine
uv run weekly.py "how do I let go of anger?"

# see the prompt without spending a claude call
uv run weekly.py "how do I let go of anger?" --dry-run

# full run: render, publish to askthecanon, queue to Buffer
uv run weekly.py "how do I let go of anger?" --to-buffer

# only some networks (default: all connected channels)
uv run weekly.py "..." --to-buffer --channels x,linkedin,threads
```

Flags: `-k` passages to rank, `--limit` questions per run, `--cooldown-days`,
`--no-images`, `--no-record`.

Queued posts land in Buffer's *queue* — review and edit them there before they publish.

## Notes

- Cards use Georgia, a **system** font. It renders correctly on macOS; a Linux box
  without Georgia falls back to a generic serif, so keep rendering local.
- With no question argument, `weekly.py` cycles a small seed list of representative
  questions (skipping any on cooldown).
