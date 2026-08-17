# =============================================================================
# Makefile - Common development and operations tasks
# =============================================================================
.PHONY: help install install-dev lint typecheck test test-unit test-integration \
        test-graph test-security migrate up down restart logs shell \
        ollama-pull models-preload clean docker-build docker-push \
        db-init db-migrate db-rollback db-seed \
        run run-dev run-prod \
        eval evaluation coverage \
        format check fmt

## Default
help: ## Show this help message
	@echo "$(PROJECT_NAME) - $(VERSION)"
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make <target>\n\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*?##/ {printf "  %-25s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

## Setup
install: ## Install dependencies
	pip install --upgrade pip
	pip install -e ".[dev]"

install-dev: ## Install with development tools
	pip install -e ".[all]"
	pre-commit install

## Docker
up: ## Start all Docker Compose services
	docker compose up -d

down: ## Stop and remove all services
	docker compose down -v

restart: down up ## Restart all services

logs: ## Tail logs from all services
	docker compose logs -f --tail=100

docker-build: ## Build all images
	docker compose build

shell: ## Enter the app container
	docker compose exec legal-ai /bin/bash

ollama-pull: ## Pull required Ollama models
	@echo "Pulling models..."
	docker compose exec ollama ollama pull qwen2.5:7b-instruct-q4_K_M
	docker compose exec ollama ollama pull qwen2.5:14b-instruct-q4_K_M
	docker compose exec ollama ollama pull qwen2.5:3b-instruct-q4_K_M
	@echo "Models pulled successfully"

## Run
run: ## Run the application
	uvicorn app.main:app --host 0.0.0.0 --port $(API_PORT) --reload

run-dev: ## Run in development mode
	APP_ENV=development APP_DEBUG=true uvicorn app.main:app --host 0.0.0.0 --port $(API_PORT) --reload

run-prod: ## Run in production mode
	gunicorn -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$(API_PORT) app.main:app

## Database
db-init: ## Initialize migrations
	alembic init -t async migrations

db-migrate: ## Run migrations
	alembic upgrade head

db-rollback: ## Rollback migrations
	alembic downgrade -1

db-seed: ## Seed the database with test data
	python scripts/seed_db.py

## Quality
lint: ## Run linters
	ruff check app/ tests/

format: ## Run formatter
	ruff format app/ tests/

fmt: format ## Alias for format

typecheck: ## Run type checker
	mypy app/ --config-file pyproject.toml

check: lint typecheck ## Run all quality checks

## Testing
test: ## Run all tests
	pytest tests/ -v --tb=short

test-unit: ## Run unit tests
	pytest tests/unit/ -v --tb=short

test-integration: ## Run integration tests
	pytest tests/integration/ -v --tb=short

test-graph: ## Run graph tests
	pytest tests/graph/ -v --tb=short

test-security: ## Run security tests
	pytest tests/security/ -v --tb=short

coverage: ## Run tests with coverage
	pytest tests/ --cov=app --cov-report=term-missing --cov-report=html

## Evaluation
eval: ## Run evaluation suite
	python -m app.evaluation.run_suite
