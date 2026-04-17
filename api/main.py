"""KlarKI FastAPI application factory with lifespan, CORS, and health check."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from models.schemas import APIResponse
from services.chroma_client import ChromaClient
from services.embedding_service import EmbeddingService
from services.ollama_client import OllamaClient
from services.rag_engine import build_bm25_index

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: initialise shared services. Shutdown: clean up."""
    logger.info("klarki_startup", debug=settings.debug, model=settings.ollama_model)

    # ChromaDB client
    chroma = ChromaClient(host=settings.chromadb_host)
    app.state.chroma = chroma

    # Embedding model — loaded once, shared across all requests
    # Runs on CPU; keeps VRAM free for Ollama
    app.state.embeddings = EmbeddingService(model_name=settings.embedding_model)

    # BM25 index — built from ChromaDB corpus once at startup
    # Also pre-loads the cross-encoder to avoid cold-start on first request
    await build_bm25_index(chroma)

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
        version="0.2.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:80", "http://localhost", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/v1/health", response_model=APIResponse, tags=["system"])
    async def health_check() -> APIResponse:
        """Return liveness status of API and dependent services."""
        chroma: ChromaClient = app.state.chroma
        chroma_ok = await chroma.health_check()

        ollama = OllamaClient(host=settings.ollama_host, model=settings.ollama_model)
        ollama_ok = await ollama.health_check()

        return APIResponse(
            status="ok",
            data={"services": {"chromadb": chroma_ok, "ollama": ollama_ok}},
        )

    from routers.audit import router as audit_router
    from routers.reports import router as reports_router
    from routers.wizard import router as wizard_router
    from routers.metrics import router as metrics_router

    app.include_router(audit_router)
    app.include_router(reports_router)
    app.include_router(wizard_router)
    app.include_router(metrics_router)

    return app


app = create_app()
