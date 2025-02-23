# Makefile for a Poetry-based Python project

# "PHONY" ensures these targets are always run even if a file with the same name exists
.PHONY: install download-model run test

# Default: installs dependencies
install:
	poetry install

# Installs the spaCy English model used in the project
download-model:
	poetry run python -m spacy download en_core_web_sm

# Runs the main entry script
run:
	poetry run python main.py

# Runs tests with pytest
test:
	poetry run pytest
