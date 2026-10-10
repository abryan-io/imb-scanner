setup:
    git rev-parse --git-dir >/dev/null 2>&1 || git init -b main
    uv sync
    pre-commit install

test:
    uv run ruff check .
    uv run pytest -q

test-all:
    uv run pytest -m "" -n auto

lint:
    uv run ruff check --fix .
    uv run ruff format .

run:
    uv run streamlit run app.py

[positional-arguments]
bws-run *cmd:
    #!/usr/bin/env bash
    set -euo pipefail
    token_file="${BWS_TOKEN_FILE:-$HOME/.config/bws/access-token}"
    [[ -r $token_file ]] || { echo "bws-run: no bws token at $token_file" >&2; exit 1; }
    token="$(< "$token_file")"
    id="$(BWS_ACCESS_TOKEN="$token" bws project list --output json | jq -er --arg n 'dev' \
      '[.[] | select(.name == $n)] | if length == 1 then .[0].id else error("want one SM project named \($n), found \(length)") end')"
    BWS_ACCESS_TOKEN="$token" exec bws run --shell bash --project-id "$id" -- "$(printf '%q ' "$@")"

update-template:
    uvx copier update --conflict inline
