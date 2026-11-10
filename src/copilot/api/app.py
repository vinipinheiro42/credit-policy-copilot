"""FastAPI application. Run with: uvicorn copilot.api.app:app"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request

from copilot import __version__
from copilot.api.schemas import AskRequest, AskResponse, HealthResponse
from copilot.config import get_settings
from copilot.observability import (
    configure_logging,
    configure_tracing,
    new_request_id,
    reset_request_id,
    set_request_id,
)
from copilot.service import CopilotService

REQUEST_ID_HEADER = "X-Request-ID"


def create_app(service: CopilotService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = get_settings()
        configure_logging(s.log_level, s.log_json)
        configure_tracing(s.otel_exporter)
        app.state.service = service or CopilotService.from_settings(s)
        yield

    app = FastAPI(title="Credit Policy Copilot", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        # accept a caller-provided id (alphanumeric, <= 64 chars) so traces join across services
        rid = request.headers.get(REQUEST_ID_HEADER, "")
        if not (rid.replace("-", "").isalnum() and len(rid) <= 64):
            rid = new_request_id()
        token = set_request_id(rid)
        try:
            response = await call_next(request)
        finally:
            reset_request_id(token)
        response.headers[REQUEST_ID_HEADER] = rid
        return response

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        svc: CopilotService = request.app.state.service
        try:
            st = svc.store.stats()
            status = "ok" if st.get("chunks", 0) > 0 else "degraded"
        except Exception as exc:  # store down: report instead of crashing the probe
            st, status = {"error": str(exc)}, "degraded"
        return HealthResponse(
            status=status,
            version=__version__,
            store=st,
            embedder=svc.embedder.name,
            llm_provider=svc.generator.name,
            retrieval_mode=svc.settings.retrieval_mode,
            rerank=svc.retriever.reranker is not None,
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest, request: Request) -> AskResponse:
        svc: CopilotService = request.app.state.service
        if req.rerank and svc.retriever.reranker is None:
            raise HTTPException(400, "Reranker not loaded; start the API with COPILOT_RERANK=true")
        result = svc.ask(req.question, top_k=req.top_k, mode=req.mode, rerank=req.rerank)
        return AskResponse(**asdict(result))

    return app


app = create_app()
