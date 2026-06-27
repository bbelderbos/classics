# Digital Ocean deployment notes

Prerequisite: have an SSH key setup so you can ssh into the box (e.g. `ssh -i ~/.ssh/id_do root@ip`)

Steps to get this deployed once you have a Digital Ocean (Ubuntu) droplet:

1. The app code

```bash
git clone https://github.com/bbelderbos/classics.git
cd classics
# make canon folder
mkdir books
```

2. Dependencies

## Memory considerations

If you're on a small droplet (< 2G), consider adding a swap file. Not sure how this pans out yet, but the following commands will create a swap file on the SSD:

```
# Disable old swap if any
swapoff -a 2>/dev/null

# Allocate a robust 2GB swap file directly on the SSD
fallocate -l 2G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile

# Make it permanent across reboots
if ! grep -q "/swapfile" /etc/fstab; then
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
```
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
# in project folder
uv sync
# mdweaver = weasyprint, required some extra libraries
apt update && apt upgrade -y
# sqlite3 is not needed but handy to have given we maintain a sqlite database of some metrics
apt install -y caddy sqlite3 python3-pip libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0
```

3. The model and embeddings

Copy the huggingface cache (the embedding model) to the droplet — the web app needs it at runtime
to embed each incoming query:

```bash
scp -i ~/.ssh/id_do -r ~/.cache/huggingface root@ip:/root/.cache/
```

For the `books/` embeddings, **build locally and rsync the artifacts up** — this is the default.
Embedding is the only expensive step, and your laptop is far faster than the droplet (and avoids
loading the model into the droplet's RAM at all). `library.txt` is the source of truth: edit it,
reconcile locally, then mirror `books/` to the droplet.

```bash
# local: build new embeddings, prune removed ones
uv run main.py sync

# mirror the artifacts up — --delete propagates removals, the trailing slash on books/ matters
rsync -az --delete -e "ssh -i ~/.ssh/id_do" books/ root@ip:/root/classics/books/

# the app caches the library at startup, so restart to pick up the change
ssh -i ~/.ssh/id_do root@ip systemctl restart classics
```

Use plain `scp` only if you don't have rsync — but note it won't delete the files for books you
dropped from `library.txt`, leaving orphans the app will still load.

> Fallback — building on the droplet. If you can't build locally, `git pull` and run
> `uv run main.py sync` on the droplet to fetch + embed there. It loads the model, so on a small
> droplet `systemctl stop classics` first (you restart afterward anyway) to avoid holding the model
> in RAM twice and risking an OOM.

4. FastAPI service

You want to have the FastAPI app running as a service so it starts on boot and can be restarted easily. Create a systemd service file for it:

```bash
vi /etc/systemd/system/classics.service

[Unit]
Description=Classics FastAPI Application
After=network.target

[Service]
User=root
WorkingDirectory=/root/classics
ExecStart=/root/.local/bin/uv run gunicorn web:app -w 1 -k uvicorn.workers.UvicornWorker -b 127.0.0.1:8000
Restart=always

[Install]
WantedBy=multi-user.target
```

You can invoke and restart the service with:

```bash
systemctl daemon-reload
systemctl enable classics
systemctl start classics
# after pulling in code changes:
systemctl restart classics
```

To look at the logs:

```bash
journalctl -u classics -f
```

# 5. [Optional] Domain and HTTPS

Caddy is a web server that can handle HTTPS automatically. You can configure it to reverse proxy to your FastAPI app. I bought a domain name and set up the DNS to point to the droplet's IP address. Then I created this Caddyfile (`/etc/caddy/Caddyfile`):

```
askthecanon.com, www.askthecanon.com {
    reverse_proxy 127.0.0.1:8000
}
```

Then restart Caddy to apply the changes:

```
systemctl restart caddy
```

---

And that's it! You should now have a working deployment of the Classics FastAPI app on your Digital Ocean droplet.
