.PHONY: install run animate lint clean

install:
	uv sync

run:
	uv run streamlit run main.py

animate:
	uv run manim -pql animate_sird.py SIRDAnimation

lint:
	uv run ruff check .

clean:
	rm -rf outputs/*.png __pycache__ .ruff_cache
