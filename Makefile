.PHONY: help install dev-install test lint format type-check clean build docs docker-build docker-build-prod docker-build-gpu docker-up docker-down docker-test docker-clean release-check release security-scan
.DEFAULT_GOAL := help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package
	pip install -e .

dev-install: ## Install package with development dependencies
	pip install -e .[dev]
	pre-commit install

test: ## Run tests
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=acousto_gen --cov-report=html --cov-report=term-missing

lint: ## Lint code
	ruff check acousto_gen/ tests/

format: ## Format code
	ruff format acousto_gen/ tests/

type-check: ## Run type checking
	mypy acousto_gen/

quality: lint type-check ## Run all quality checks

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean ## Build distribution packages
	python -m build

docs: ## Build documentation
	cd docs && make html

docs-serve: docs ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

# Docker targets
docker-build: ## Build Docker development image
	docker build --target development -t acousto-gen:dev .

docker-build-prod: ## Build Docker production image
	docker build --target production -t acousto-gen:prod .

docker-build-gpu: ## Build Docker GPU image
	docker build --target gpu -t acousto-gen:gpu .

docker-up: ## Start development environment
	docker-compose up -d acousto-gen-dev

docker-down: ## Stop all containers
	docker-compose down

docker-test: ## Run tests in Docker
	docker-compose --profile test up acousto-gen-test

docker-clean: ## Clean Docker resources
	docker-compose down -v
	docker system prune -f

# Release targets
release-check: quality test ## Check release readiness
	python -m build --check

release: clean build ## Create release packages
	python -m build
	twine check dist/*

# Security targets
security-scan: ## Run security scans
	bandit -r acousto_gen/ -ll
	safety check
	pip-audit