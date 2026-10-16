.PHONY: test test-slow lint demo ui-dev import-demo
test:        ; .venv/bin/python -m pytest -q -m "not slow"
test-slow:   ; .venv/bin/python -m pytest -q -m slow
lint:        ; .venv/bin/ruff check .
demo:        ; .venv/bin/contract-assistant serve
ui-dev:      ; PYTHONPATH=. .venv/bin/gradio tests/support/dev_app.py --watch-dirs contract_assistant
import-demo: ; .venv/bin/contract-assistant import-demo
