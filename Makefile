.PHONY: test test-slow test-pg lint db-up db-down db-psql db-backup db-restore db-upgrade db-status import-cache demo ui-dev web-dev import-demo router-eval e2e summary-eval term-sheet-eval passage-eval
test:        ; .venv/bin/python -m pytest -q -m "not slow"
test-slow:   ; .venv/bin/python -m pytest -q -m slow
test-pg:     db-up ; .venv/bin/python -m pytest -q -m pg
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

# The database of the web app (docker-compose.yml). Its data lives in the volume pgdata.
BACKUPS ?= backups
db-up:       ; docker compose up -d --wait db
db-down:     ; docker compose stop db
db-psql:     ; docker compose exec db psql -U contract_assistant contract_assistant
db-backup:   ; mkdir -p $(BACKUPS) && docker compose exec -T db pg_dump -U contract_assistant -Fc contract_assistant > $(BACKUPS)/contract-assistant-$$(date +%Y%m%d-%H%M%S).dump && ls -t $(BACKUPS) | head -1
db-restore:  ; test -n "$(FILE)" && docker compose exec -T db pg_restore -U contract_assistant -d contract_assistant --clean --if-exists < $(FILE)
db-upgrade:  ; .venv/bin/contract-assistant db upgrade
db-status:   ; .venv/bin/contract-assistant db status
# One-time: give OWNER (an account's email) a contract for each analysis of the cache folder.
import-cache: ; test -n "$(OWNER)" && .venv/bin/contract-assistant import-cache --owner "$(OWNER)"
