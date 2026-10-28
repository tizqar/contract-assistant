"""The web app: a FastAPI application over the workspace, with the routes of each area."""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from contract_assistant.access import AccessError
from contract_assistant.accounts import AccountError
from contract_assistant.background import run_in_background, warm
from contract_assistant.db.engine import Database
from contract_assistant.services import Services
from contract_assistant.web import auth, chat, decide, i18n, insights, library
from contract_assistant.web.core import (
    KEEP_WARM_S,
    STATIC_DIR,
    Workspace,
    WorkspaceError,
    language,
)
from contract_assistant.web.i18n import translate

logger = logging.getLogger(__name__)

TITLE = "Contract Assistant"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860


def create_app(
    services: Services,
    database: Database,
    keep_warm: bool = False,
    precompute: bool = False,
) -> FastAPI:
    """Build the app over a database.

    With keep_warm, a thread reloads the language models while idle; with precompute, the
    summaries, key terms and obligations missing for any contract are written at start-up.
    """
    stop = threading.Event()
    workspace = Workspace(services, database)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if keep_warm:
            threading.Thread(target=_keep_warm, args=(services, stop), daemon=True).start()
        if precompute and services.gen is not None:
            run_in_background(workspace.precompute_all)
        yield
        stop.set()

    app = FastAPI(title=TITLE, lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.workspace = workspace
    app.state.precompute = precompute
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(auth.router)
    app.include_router(library.router)
    app.include_router(chat.router)
    app.include_router(insights.router)
    app.include_router(decide.router)
    app.include_router(i18n.router)

    # The middleware added last runs first: the language is known before the guard answers.
    app.middleware("http")(auth.guard)

    @app.middleware("http")
    async def remember_language(request: Request, call_next):
        request.state.lang = i18n.language_of(request)
        return await call_next(request)

    @app.exception_handler(WorkspaceError)
    async def workspace_error(request: Request, error: WorkspaceError):
        return PlainTextResponse(error.message, status_code=error.status_code)

    @app.exception_handler(AccessError)
    async def access_error(request: Request, error: AccessError):
        message = translate(error.message, language(request))
        return PlainTextResponse(message, status_code=error.status_code)

    @app.exception_handler(AccountError)
    async def account_error(request: Request, error: AccountError):
        message = translate(error.message, language(request))
        return PlainTextResponse(message, status_code=error.status_code)

    return app


def _keep_warm(services: Services, stop: threading.Event) -> None:
    while not stop.wait(KEEP_WARM_S):
        warm([services.gen, getattr(services.router, "client", None)])


def run_server(app: FastAPI, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Serve the app with uvicorn until interrupted."""
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")
