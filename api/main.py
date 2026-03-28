"""KlarKI FastAPI application factory with lifespan, CORS, and health check."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from services.chroma_client import ChromaClient
from models.schemas import APIResponse

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: initialise shared services. Shutdown: clean up connections."""
    logger.info("klarki_startup", debug=settings.debug, model=settings.ollama_model)

    # Initialise ChromaDB client and attach to app state
    chroma = ChromaClient(host=settings.chromadb_host)
    app.state.chroma = chroma

    # Phase 2: EmbeddingService loaded here to avoid cold-start on first request
    # from services.embedding_service import EmbeddingService
    # app.state.embeddings = EmbeddingService()

    logger.info("klarki_ready")
    yield

    logger.info("klarki_shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="KlarKI — EU AI Act & GDPR Compliance Auditor",
        description=(
            "Local-first compliance auditor for German SMEs. "
            "Analyses documentation against EU AI Act Articles 9–15 and GDPR."
        ),
        version="0.1.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    # CORS — allow frontend dev server and production nginx
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:80", "http://localhost"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get("/api/v1/health", response_model=APIResponse, tags=["system"])
    async def health_check() -> APIResponse:
        """Return liveness status of API and dependent services.

        Returns:
            APIResponse with service health flags for chromadb and ollama.
        """
        chroma: ChromaClient = app.state.chroma
        chroma_ok = await chroma.health_check()

        # Ollama health — lightweight HTTP check (Phase 2 will use OllamaClient)
        import httpx
        ollama_ok = False
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{settings.ollama_host}/api/tags")
                ollama_ok = resp.status_code == 200
        except Exception:
            ollama_ok = False

        return APIResponse(
            status="ok",
            data={
                "services": {
                    "chromadb": chroma_ok,
                    "ollama": ollama_ok,
                }
            },
        )

    # ── Routers (Phase 2) ─────────────────────────────────────────────────────
    # from routers.audit import router as audit_router
    # from routers.reports import router as reports_router
    # app.include_router(audit_router)
    # app.include_router(reports_router)

    return app


app = create_app()
