.PHONY: test test-slow lint demo ui-dev web-dev import-demo router-eval e2e summary-eval term-sheet-eval passage-eval
test:        ; .venv/bin/python -m pytest -q -m "not slow"
test-slow:   ; .venv/bin/python -m pytest -q -m slow
lint:        ; .venv/bin/ruff check .
demo:        ; .venv/bin/contract-assistant serve
ui-dev:      ; PYTHONPATH=. .venv/bin/gradio tests/support/dev_app.py --watch-dirs contract_assistant
web-dev:     ; PYTHONPATH=. .venv/bin/uvicorn tests.support.dev_web:app --reload --port 7861
import-demo: ; .venv/bin/contract-assistant import-demo
router-eval: ; .venv/bin/contract-assistant router-eval --router llm --split test
e2e:         ; .venv/bin/contract-assistant e2e-eval --router llm
summary-eval: ; .venv/bin/contract-assistant summary-eval
term-sheet-eval: ; .venv/bin/contract-assistant term-sheet-eval
passage-eval: ; .venv/bin/contract-assistant passage-eval
