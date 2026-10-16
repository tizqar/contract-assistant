.PHONY: test test-slow lint demo
test:        ; .venv/bin/python -m pytest -q -m "not slow"
test-slow:   ; .venv/bin/python -m pytest -q -m slow
lint:        ; .venv/bin/ruff check .
demo:        ; .venv/bin/contract-assistant serve
