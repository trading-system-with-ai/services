.PHONY: install dev test ci verify up down

install:
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

dev:
	.venv/bin/uvicorn apps.gateway.main:app --reload --port 8000

test:
	.venv/bin/pytest -q

ci:
	.venv/bin/pytest -q

verify:
	.venv/bin/pytest -q
	docker compose config -q
	.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"

up:
	docker compose up --build -d

down:
	docker compose down
