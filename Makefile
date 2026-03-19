.PHONY: install run lint clean

install:
	uv sync

run:
	uv run streamlit run main.py

lint:
	uv run ruff check .

clean:
	rm -rf outputs/*.png __pycache__ .ruff_cache
