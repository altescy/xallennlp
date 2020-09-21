PWD=$(shell pwd)
PYTHON=poetry run python
PYLINT=poetry run pylint
PYLINTRC=$(PWD)/.pylintrc
MYPY=poetry run mypy
PYTEST=poetry run pytest
MODULE=xallennlp


lint:
	$(PYLINT) --rcfile=$(PYLINTRC) $(MODULE)

mypy:
	$(MYPY) $(MODULE)

test:
	PYTHONPATH=$(PWD) $(PYTEST)

clean: clean-pyc clean-build

clean-pyc:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf $(MODULE).egg-info/
	rm -rf pip-wheel-metadata/
