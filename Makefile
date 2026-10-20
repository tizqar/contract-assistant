.PHONY: test test-slow lint demo ui-dev import-demo router-eval e2e summary-eval
test:        ; .venv/bin/python -m pytest -q -m "not slow"
test-slow:   ; .venv/bin/python -m pytest -q -m slow
lint:        ; .venv/bin/ruff check .
demo:        ; .venv/bin/contract-assistant serve
ui-dev:      ; PYTHONPATH=. .venv/bin/gradio tests/support/dev_app.py --watch-dirs contract_assistant
import-demo: ; .venv/bin/contract-assistant import-demo
router-eval: ; .venv/bin/contract-assistant router-eval --router llm --split test
e2e:         ; .venv/bin/contract-assistant e2e-eval --router llm
summary-eval: ; .venv/bin/contract-assistant summary-eval
