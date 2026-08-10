.PHONY: install dev test ci up down

install:
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

dev:
	.venv/bin/uvicorn apps.gateway.main:app --reload --port 8000

test:
	.venv/bin/pytest -q

ci:
	.venv/bin/pytest -q

up:
	docker compose up --build -d

down:
	docker compose down
