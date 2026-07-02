import subprocess
from pathlib import Path

from decouple import config


def card_url(base: str, slug: str, name: str) -> str:
    return f"{base.rstrip('/')}/{slug}/{name}"


def publish_cards(slug: str, paths: list[Path]) -> list[str]:
    dest = config("CARDS_RSYNC_DEST")  # e.g. root@1.2.3.4:/root/classics/cards
    base = config("CARDS_BASE_URL")  # e.g. https://askthecanon.com/cards
    key = config("SSH_KEY", default="")

    local_dir = paths[0].parent
    cmd = ["rsync", "-az"]
    if key:
        cmd += ["-e", f"ssh -i {key}"]
    cmd += [f"{local_dir}/", f"{dest.rstrip('/')}/{slug}/"]
    subprocess.run(cmd, check=True)

    return [card_url(base, slug, p.name) for p in paths]
