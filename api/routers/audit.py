"""Audit router — document upload and status polling endpoints.

Endpoints:
    POST /api/v1/audit/upload       — Upload files or raw text, start pipeline
    GET  /api/v1/audit/{audit_id}   — Fetch full AuditResponse
    GET  /api/v1/audit/{audit_id}/status — Lightweight status poll
"""

import asyncio
import os
import uuid
from pathlib import Path
from typing import Annotated

import aiofiles
import structlog
from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, Request, UploadFile

from config import settings
from models.schemas import (
    APIResponse,
    ArticleDomain,
    AuditResponse,
    AuditStatus,
    RiskTier,
)
from services.actor_classifier import classify_actor
from services.applicability_engine import check_applicability
from services.classifier import classify_chunks
from services.emotion_module import check_emotion_recognition
from services.compliance_scorer import ARTICLE_DOMAINS, score_audit
from services.document_parser import parse_document, SUPPORTED_EXTENSIONS
from services.chunker import proposition_chunk_text
from services.evidence_mapper import map_evidence
from services.gap_analyser import analyse_article
from services.language_detector import detect_language
from services.ner_service import enrich_chunks_with_ner
from services.ollama_client import OllamaClient
from services.rag_engine import retrieve_requirements
from services.monitoring_stats import stats as _monitor

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/audit", tags=["audit"])

# In-memory audit store — replace with Redis or a DB for multi-worker deployments.
_audits: dict[str, AuditResponse] = {}


@router.post("/upload", response_model=APIResponse)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile | None = None,
    raw_text: Annotated[str | None, Form()] = None,
    wizard_risk_tier: Annotated[str | None, Form()] = None,
) -> APIResponse:
    """Accept a document file or raw text and start the compliance audit pipeline.

    One of `file` or `raw_text` must be provided.

    Args:
        file: Uploaded PDF, DOCX, TXT, or MD file (max 10 MB).
        raw_text: Plain text pasted directly into the form.
        wizard_risk_tier: Optional risk tier from the Annex III wizard (pre-audit self-assessment).

    Returns:
        APIResponse with audit_id to poll for status.
    """
    if file is None and not raw_text:
        raise HTTPException(status_code=400, detail="Provide a file or raw_text.")

    audit_id = str(uuid.uuid4())

    # Validate file type and size
    if file is not None:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{ext}'. Accepted: {', '.join(SUPPORTED_EXTENSIONS)}",
            )
        content = await file.read()
        if len(content) > settings.upload_max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds {settings.upload_max_size_mb} MB limit.",
            )

        # Save to disk
        upload_path = Path(settings.upload_dir) / f"{audit_id}{ext}"
        async with aiofiles.open(upload_path, "wb") as f_out:
            await f_out.write(content)

        filename = file.filename or f"upload{ext}"
        file_path = str(upload_path)
    else:
        # Raw text — save as .txt
        filename = "paste.txt"
        upload_path = Path(settings.upload_dir) / f"{audit_id}.txt"
        async with aiofiles.open(upload_path, "w", encoding="utf-8") as f_out:
            await f_out.write(raw_text)  # type: ignore[arg-type]
        file_path = str(upload_path)

    # Register audit as UPLOADING
    _audits[audit_id] = AuditResponse(audit_id=audit_id, status=AuditStatus.UPLOADING)
    _monitor.audit_started(audit_id)

    # Parse wizard tier if provided
    parsed_wizard_tier: RiskTier | None = None
    if wizard_risk_tier:
        try:
            parsed_wizard_tier = RiskTier(wizard_risk_tier)
        except ValueError:
            pass  # Ignore invalid tier values

    # Kick off pipeline in background
    ollama = OllamaClient(host=settings.ollama_host, model=settings.ollama_model)
    background_tasks.add_task(
        _run_pipeline,
        audit_id=audit_id,
        file_path=file_path,
        filename=filename,
        request=request,
        ollama=ollama,
        wizard_risk_tier=parsed_wizard_tier,
    )

    logger.info("audit_started", audit_id=audit_id, filename=filename)
    return APIResponse(status="success", data={"audit_id": audit_id})


@router.get("/{audit_id}", response_model=AuditResponse)
async def get_audit(audit_id: str) -> AuditResponse:
    """Return full AuditResponse including ComplianceReport when COMPLETE.

    Args:
        audit_id: The audit identifier returned by /upload.

    Returns:
        AuditResponse with status and optional report.
    """
    audit = _audits.get(audit_id)
    if audit is None:
        raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found.")
    return audit


@router.get("/{audit_id}/status", response_model=APIResponse)
async def get_audit_status(audit_id: str) -> APIResponse:
    """Return current AuditStatus for lightweight polling.

    Args:
        audit_id: The audit identifier.

    Returns:
        APIResponse with status string.
    """
    audit = _audits.get(audit_id)
    if audit is None:
        raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found.")
    return APIResponse(status="success", data={"status": audit.status.value})


async def _run_pipeline(
    audit_id: str,
    file_path: str,
    filename: str,
    request: Request,
    ollama: OllamaClient,
    wizard_risk_tier: RiskTier | None = None,
) -> None:
    """Full compliance audit pipeline executed as a BackgroundTask.

    Stages: parse → chunk → detect language → classify → RAG → gap analysis → score
    Updates _audits[audit_id].status at each stage.
    """

    def _set_status(status: AuditStatus) -> None:
        if audit_id in _audits:
            _audits[audit_id] = AuditResponse(
                audit_id=audit_id,
                status=status,
                report=_audits[audit_id].report,
            )
    
    async def process_article(
        article_num,
        domain,
        domain_chunks,
        embeddings,
        chroma,
        ollama,
        applicable_articles,
    ):
        art_chunks = domain_chunks.get(domain, [])
        query_chunks = art_chunks[:3]

        reg_passages = []

        if art_chunks:
            for c in query_chunks:
                passages = await retrieve_requirements(
                    chunk=c,
                    embedding_service=embeddings,
                    chroma_client=chroma,
                    top_k=5,
                    applicable_articles=applicable_articles,
                    regulation="eu_ai_act",
                )
                reg_passages.extend(passages)

        seen = set()
        unique_passages = []
        for p in reg_passages:
            key = p.get("id") or p.get("text")
            if key not in seen:
                seen.add(key)
                unique_passages.append(p)

        reg_passages = unique_passages[:5]

        score = await analyse_article(
            article_num=article_num,
            domain=domain,
            user_chunks=art_chunks,
            regulatory_passages=reg_passages,
            ollama=ollama,
            applicable_articles=applicable_articles,
        )

        return score

    try:
        import time as _time

        chroma = request.app.state.chroma
        embeddings = request.app.state.embeddings

        # Parse → chunk (legal-unit) → detect language
        _set_status(AuditStatus.PARSING)
        _t0 = _time.time()
        raw_text = await parse_document(file_path, filename)
        chunks = await proposition_chunk_text(raw_text, source_file=filename)
        language = await detect_language(raw_text)
        _monitor.record_stage("parsing", _time.time() - _t0)
        for chunk in chunks:
            chunk.language = language

        # Phase 3 — legal decision hierarchy (must run before gap analysis)
        # Step 1: detect actor role (Article 3)
        # Step 2: applicability gate (Article 6 + Annex III) — determines if 9–15 apply
        _set_status(AuditStatus.CLASSIFYING)
        _t0 = _time.time()
        actor_result = await asyncio.to_thread(classify_actor, raw_text)
        applicability_result = await asyncio.to_thread(check_applicability, chunks)
        _monitor.record_stage("classifying", _time.time() - _t0)

        logger.info(
            "applicability_determined",
            audit_id=audit_id,
            actor=actor_result.actor_type.value,
            is_high_risk=applicability_result.is_high_risk,
            is_prohibited=applicability_result.is_prohibited,
            annex_iii_categories=[m.category.value for m in applicability_result.annex_iii_matches],
        )

        # Classify each chunk into an ArticleDomain
        chunks = await classify_chunks(chunks, ollama)
        classifier_backend = (
            "triton/gbert-base"
            if settings.use_triton
            else f"ollama/{settings.ollama_model}"
        )

        # NER enrichment — adds entity metadata to each chunk and corrects domain
        # for UNRELATED chunks that contain explicit Article references (9–15).
        chunks = await asyncio.to_thread(enrich_chunks_with_ner, chunks)

        # RAG retrieval + per-article gap analysis
        _set_status(AuditStatus.ANALYSING)
        _t0 = _time.time()

        # Group chunks by domain
        domain_chunks: dict[ArticleDomain, list] = {d: [] for d in ArticleDomain}
        for chunk in chunks:
            if chunk.domain:
                domain_chunks[chunk.domain].append(chunk)

        applicable_articles = applicability_result.applicable_articles

        tasks = []
        for article_num, domain in ARTICLE_DOMAINS.items():
            tasks.append(
                process_article(
                    article_num,
                    domain,
                    domain_chunks,
                    embeddings,
                    chroma,
                    ollama,
                    applicable_articles,
                )
            )

        article_scores = await asyncio.gather(*tasks)
        _monitor.record_stage("analysing", _time.time() - _t0)

        # Phase 3 — evidence mapping (deterministic, runs before LLM scoring)
        evidence_map = await asyncio.to_thread(
            map_evidence,
            chunks,
            actor_result.actor_type,
            applicable_articles,
        )

        # Aggregate article scores into a ComplianceReport
        _set_status(AuditStatus.SCORING)
        emotion_flag = await check_emotion_recognition(chunks)
        report = await score_audit(
            article_scores=article_scores,
            chunks=chunks,
            audit_id=audit_id,
            source_files=[filename],
            language=language,
            emotion_flag=emotion_flag,
            classifier_backend=classifier_backend,
            wizard_risk_tier=wizard_risk_tier,
            actor=actor_result,
            applicability=applicability_result,
            evidence_map=evidence_map,
            model_versions=getattr(request.app.state, "model_versions", {}),
        )

        _audits[audit_id] = AuditResponse(
            audit_id=audit_id,
            status=AuditStatus.COMPLETE,
            report=report,
        )
        _monitor.audit_completed(audit_id)
        logger.info("audit_complete", audit_id=audit_id, score=report.overall_score)

    except Exception as exc:
        logger.error("audit_failed", audit_id=audit_id, error=str(exc), exc_info=True)
        _monitor.audit_failed(audit_id)
        _set_status(AuditStatus.FAILED)

    finally:
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except OSError:
            pass
