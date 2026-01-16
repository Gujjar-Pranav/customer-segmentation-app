run:
	python -m src.main --config config/config.yaml

format:
	black src

lint:
	ruff src

check: format lint
