PWD=$(shell pwd)
PYTHON=poetry run python
PYTEST=poetry run pytest
PYSEN=poetry run pysen
MODULE=xallennlp

.PHONY: all
all: format lint test

install:
	poetry install --extras all
	$(PYTHON) -c 'import nltk; [nltk.download(p) for p in ("wordnet", "wordnet_ic", "sentiwordnet", "omw", "omw-1.4")]'

.PHONY: format
format:
	$(PYSEN) run format

.PHONY: lint
lint:
	$(PYSEN) run lint

.PHONY: test
test:
	PYTHONPATH=$(PWD) $(PYTEST)

.PHONY: clean
clean: clean-pyc clean-build

.PHONY: clean-pyc
clean-pyc:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-build
clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf $(MODULE).egg-info/
	rm -rf pip-wheel-metadata/
