.PHONY:	jupyter format_black sphinx_build

IMAGE_NAME=alpacadb/alpaca-containers:forecast-exp-v0.0.2

MAKEFILE_PATH := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_ROOT := $(abspath $(MAKEFILE_PATH))

UNIT_TEST_OPTS=\
	-ra \
	--maxfail=1 \
	--durations=10 \
	--flake8 \
	$(TEST_OPTS) \
	/project/tests


DOCKER_OPTS=\
	--env-file $(PROJECT_ROOT)/.env \
	-e PYTHONPATH=/project/src

_mount_src:
	docker rm -f mysrc || echo
	docker create -v /project --name mysrc alpine:3.4 /bin/true
	docker cp $(PROJECT_ROOT)/. mysrc:/project/

_install_pypy:
	docker run $(DOCKER_OPTS) \
		-it --volumes-from mysrc \
		$(IMAGE_NAME) \
		bash -c 'sudo apt-get update -y; sudo apt-get install pypy3 -y'

mypy: _mount_src
	docker pull alpacadb/alpaca-containers:mypy-v0.0.1
	docker run -it -w /project --volumes-from mysrc \
		--entrypoint mypy \
		alpacadb/alpaca-containers:mypy-v0.0.1 \
		--ignore-missing-imports --strict-optional --disallow-untyped-defs --disallow-untyped-calls /project/src

check_black: _mount_src
	docker pull unibeautify/black
	docker run -it -w /project --volumes-from mysrc \
		--entrypoint black \
		unibeautify/black \
		--line-length 88 --check /project

format_black:
	docker pull unibeautify/black
	docker run -it -v $(PROJECT_ROOT):/workdir -w /workdir unibeautify/black --line-length 88 /workdir

flake8: _mount_src
	docker run $(DOCKER_OPTS) \
		-it --volumes-from mysrc \
		$(IMAGE_NAME) \
		bash -c "flake8 /project/src /project/tests"

unittest: _mount_src
	docker run $(DOCKER_OPTS) \
		-it --volumes-from mysrc \
		$(IMAGE_NAME) \
		bash -c 'PYTHONIOENCODING=UTF-8 py.test $(UNIT_TEST_OPTS)'

unittest_pypy: _mount_src _install_pypy
	docker run $(DOCKER_OPTS) \
		-it --volumes-from mysrc \
		$(IMAGE_NAME) \
		bash -c 'PYTHONIOENCODING=UTF-8 pypy3 -m pytest $(UNIT_TEST_OPTS)'


build:
	python setup.py build

dist: sphinx_build
	python setup.py sdist

sphinx_build:
	sphinx-apidoc -F -f \
		-o ./docs/source/ \
		-V "$(shell python -c "from src.backlight import __version__; print(__version__)")" \
		-A "$(shell python -c "from src.backlight import __author__; print(__author__)")" \
		./src/
	sphinx-build -b html docs/source docs/build/sphinx/html

sphinx_autobuild:
	sphinx-autobuild -b html docs/source docs/build/sphinx/html

jupyter:
	bash -c "PYTHONPATH=`pwd`/src:${PYTHONPATH} jupyter notebook"
