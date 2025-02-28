.DEFAULT_GOAL := help
.PHONY: help tests
SHELL := /bin/bash

# General Commands
help:
	cat Makefile

create_venv:
	sh scripts/run_app.sh create_venv

# Install dependencies
install:
    pip install -r requirements.txt

# Install development dependencies
install-dev:
    pip install -r requirements-dev.txt

# Run tests
test: install
    pytest tests --junitxml=report.xml

# Development Commands
lint: install
    prospector

types: install
    mypy .

coverage: install
    coverage run -m unittest discover tests
    coverage report

format: install
    yapf -i *.py **/*.py **/**/*.py

format_check: install
    yapf --diff *.py **/*.py **/**/*.py

pycodestyle: install
    pycodestyle

qa: lint types test format_check pycodestyle

# Clean build artifacts
clean:
    rm -rf __pycache__ .pytest_cache .mypy_cache .coverage report.xml

# Docker Commands
docker-build:
    docker-compose build

docker-up:
    docker-compose up

docker-down:
    docker-compose down

# Application Specific Commands
run:
    sh scripts/run_app.sh run

requirements:
    sh scripts/run_app.sh requirements
