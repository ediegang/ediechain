# `make` runs what CI runs: lint, types, tests.
all: lint types test

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

install:  ## create the environment with every extra
	uv sync --all-extras

lint:  ## ruff
	uv run ruff check ./src ./tests

format:  ## ruff format, then fix what is fixable
	uv run ruff format ./src ./tests
	uv run ruff check --fix ./src ./tests

types:  ## ty
	uv run ty check src tests

test:  ## pytest
	uv run pytest tests

build:  ## wheel and sdist into dist/
	rm -rf dist
	uv sync
	uv build

publish: all build  ## build and push to seznam-pypi-staging; needs SASANKA_TOKEN
	@test -n "$$SASANKA_TOKEN" || { echo "SASANKA_TOKEN is not set"; exit 1; }
	@echo 'Publishing version:' && uv version --short
	@TOKEN=`curl -f "https://sasanka.seznam.net/api/v1/users/me/jwt/pypi" \
		-H "accept: application/json" -H "Authorization: Bearer $$SASANKA_TOKEN" | jq -r '.token'` && \
	uv publish --index seznam-pypi-staging --token $$TOKEN

clean:  ## remove build output, caches, and the stores the examples write
	rm -rf dist .pytest_cache .ruff_cache
	find examples -name .artifacts -type d -prune -exec rm -rf {} +
	find src tests examples -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: all help install lint format types test build publish clean
