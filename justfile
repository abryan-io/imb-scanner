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

update-template:
    uvx copier update --conflict rej
