# Set uv as the default shell wrapper if you want, or explicitly call it.
# This ensures we run everything in the correct virtual environment context.

default:
    @just --list

# Sync virtual environment dependencies
[group('setup')]
sync:
    uv sync

# Run the FastAPI development server
[group('dev')]
server: sync
    uv run fastapi dev web.py

# Find a book's Gutenberg ID (e.g., just search "tolstoy war and peace")
[group('library')]
search *query: sync
    uv run main.py search {{ query }}

# Download a book's text by ID (e.g., just fetch 2600)
[group('library')]
fetch id: sync
    uv run main.py fetch {{ id }}

# Reconcile books/ embeddings with library.txt (edit the file, then 'just reindex')
[group('library')]
reindex: sync
    uv run main.py sync

# Ask a question across the library (e.g., just ask "the meaning of suffering" "-k 8")
[group('search')]
ask query *flags="": sync
    uv run main.py ask "{{ query }}" {{ flags }}
