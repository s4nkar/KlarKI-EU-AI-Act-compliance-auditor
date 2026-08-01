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
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile

from config import settings
from models.schemas import (
    APIResponse,
    ArticleDomain,
    AuditProgress,
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
from services.ner_service import apply_ner_domain_correction, extract_ner_entities_async
from services.ollama_client import OllamaClient
from services.rag_engine import retrieve_requirements, select_query_chunks
from services.monitoring_stats import stats as _monitor

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/audit", tags=["audit"])

# In-memory audit store — replace with Redis or a DB for multi-worker deployments.
_audits: dict[str, AuditResponse] = {}


@router.post("/upload", response_model=APIResponse)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    files: Annotated[list[UploadFile], File()] = [],
    raw_text: Annotated[str | None, Form()] = None,
    wizard_risk_tier: Annotated[str | None, Form()] = None,
) -> APIResponse:
    """Accept one or more document files, or raw text, and start the compliance audit pipeline.

    One of `files` or `raw_text` must be provided. Each uploaded file is parsed
    and chunked independently (so language detection and source-file
    provenance stay accurate per document) before all chunks are pooled for
    the rest of the pipeline.

    Args:
        files: One or more uploaded PDF, DOCX, TXT, or MD files (max 10 MB each,
               max settings.upload_max_files files).
        raw_text: Plain text pasted directly into the form.
        wizard_risk_tier: Optional risk tier from the Annex III wizard (pre-audit self-assessment).

    Returns:
        APIResponse with audit_id to poll for status.
    """
    files = [f for f in (files or []) if f.filename]

    if not files and not raw_text:
        raise HTTPException(status_code=400, detail="Provide one or more files, or raw_text.")

    if len(files) > settings.upload_max_files:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files ({len(files)}). Maximum is {settings.upload_max_files}.",
        )

    audit_id = str(uuid.uuid4())
    file_paths: list[str] = []
    filenames: list[str] = []

    if files:
        for idx, file in enumerate(files):
            ext = Path(file.filename or "").suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                raise HTTPException(
                    status_code=415,
                    detail=f"Unsupported file type '{ext}' ({file.filename}). "
                            f"Accepted: {', '.join(SUPPORTED_EXTENSIONS)}",
                )
            content = await file.read()
            if len(content) > settings.upload_max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"'{file.filename}' exceeds {settings.upload_max_size_mb} MB limit.",
                )

            upload_path = Path(settings.upload_dir) / f"{audit_id}_{idx}{ext}"
            async with aiofiles.open(upload_path, "wb") as f_out:
                await f_out.write(content)

            filenames.append(file.filename or f"upload_{idx}{ext}")
            file_paths.append(str(upload_path))
    else:
        # Raw text — save as .txt
        upload_path = Path(settings.upload_dir) / f"{audit_id}.txt"
        async with aiofiles.open(upload_path, "w", encoding="utf-8") as f_out:
            await f_out.write(raw_text)  # type: ignore[arg-type]
        filenames.append("paste.txt")
        file_paths.append(str(upload_path))

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
        file_paths=file_paths,
        filenames=filenames,
        request=request,
        ollama=ollama,
        wizard_risk_tier=parsed_wizard_tier,
    )

    logger.info("audit_started", audit_id=audit_id, filenames=filenames)
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
    return APIResponse(status="success", data={
        "status": audit.status.value,
        "progress": audit.progress.model_dump() if audit.progress else None,
    })


async def _run_pipeline(
    audit_id: str,
    file_paths: list[str],
    filenames: list[str],
    request: Request,
    ollama: OllamaClient,
    wizard_risk_tier: RiskTier | None = None,
) -> None:
    """Full compliance audit pipeline executed as a BackgroundTask.

    Stages: parse → chunk → detect language → classify → RAG → gap analysis → score

    Each file in file_paths/filenames is parsed, chunked, and language-detected
    independently (so per-document provenance and language stay accurate),
    then all chunks are pooled into one flat list before every downstream
    stage — those already operate on a flat list[DocumentChunk] regardless of
    how many source documents it came from.

    Updates _audits[audit_id].status at each stage.
    """

    def _set_status(status: AuditStatus) -> None:
        if audit_id in _audits:
            _audits[audit_id] = AuditResponse(
                audit_id=audit_id,
                status=status,
                report=_audits[audit_id].report,
                progress=_audits[audit_id].progress,
            )

    def _set_progress(**fields) -> None:
        """Update fine-grained progress within the current stage, without
        touching status/report. Safe under asyncio's cooperative scheduling —
        no true parallelism, so no lock needed even when called from multiple
        concurrently-running process_article() coroutines."""
        if audit_id in _audits:
            current = _audits[audit_id]
            _audits[audit_id] = AuditResponse(
                audit_id=audit_id,
                status=current.status,
                report=current.report,
                progress=AuditProgress(**fields),
            )

    # Mutable boxes so process_article's closure can update shared counters
    # across concurrently-scheduled tasks (asyncio.gather runs them
    # concurrently, but gap_analyser.py's _LLM_SEMAPHORE effectively
    # serializes the actual LLM calls, so "N of 7 done" is real progress,
    # not a fake animation). _llm_articles_done / _analysing_t0 feed a live
    # ETA computed from this run's own observed per-article timing — not a
    # static guess — since only articles with chunks AND applicability
    # actually go through the slow LLM path; the rest finish instantly.
    _articles_done = [0]
    _llm_articles_done = [0]
    _analysing_t0 = [0.0]

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
        is_applicable = not applicable_articles or article_num in applicable_articles
        used_llm = bool(art_chunks) and is_applicable
        query_chunks = await select_query_chunks(art_chunks, article_num, embeddings)

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

        _articles_done[0] += 1

        eta_seconds = None
        if used_llm:
            _llm_articles_done[0] += 1
            elapsed_in_stage = _time.time() - _analysing_t0[0]
            avg_per_llm_article = elapsed_in_stage / _llm_articles_done[0]
            remaining_llm = max(0, _llm_articles_total - _llm_articles_done[0])
            eta_seconds = round(avg_per_llm_article * remaining_llm)

        _set_progress(
            articles_done=_articles_done[0],
            articles_total=len(ARTICLE_DOMAINS),
            estimated_seconds_remaining=eta_seconds,
        )

        return score

    try:
        import time as _time

        chroma = request.app.state.chroma
        embeddings = request.app.state.embeddings

        # ── Stage 1: parse → chunk → language (per file, then pooled) ────────
        # Each file is parsed, chunked, and language-detected independently so
        # a multi-document upload with mixed languages (e.g. an English risk
        # policy + a German technical file) doesn't have one file's language
        # misapplied to another's chunks. Only the resulting chunk lists are
        # pooled — everything from here on operates on a flat list[DocumentChunk].
        _set_status(AuditStatus.PARSING)
        _set_progress(files_done=0, files_total=len(file_paths))
        _t0 = _time.time()
        raw_texts: list[str] = []
        chunks: list = []
        primary_language: str | None = None
        for idx, (path, name) in enumerate(zip(file_paths, filenames)):
            file_raw_text = await parse_document(path, name)
            file_chunks = await proposition_chunk_text(file_raw_text, source_file=name)
            file_language = await detect_language(file_raw_text)
            for chunk in file_chunks:
                chunk.language = file_language
            raw_texts.append(file_raw_text)
            chunks.extend(file_chunks)
            if primary_language is None:
                primary_language = file_language  # report-level language = first file's
            _set_progress(files_done=idx + 1, files_total=len(file_paths))
        # Actor classification takes one text blob for pattern matching (not
        # chunking/heading-sensitive like the chunker), so concatenating raw
        # texts here is safe — this reasoning is specific to classify_actor,
        # not a general license to concatenate raw text elsewhere.
        raw_text = "\n\n".join(raw_texts)
        _monitor.record_stage("parsing", _time.time() - _t0)

        # ── Stage 2: NER entity extraction ───────────────────────────────────
        # Runs before the legal gate so PROHIBITED_USE / RISK_TIER entities
        # are available to applicability_engine. Domain correction happens
        # after classify_chunks (Phase 2 below).
        _set_status(AuditStatus.EXTRACTING_ENTITIES)
        _t0 = _time.time()
        chunks = await extract_ner_entities_async(chunks)

        # ── Stage 3: actor + applicability gate ──────────────────────────────
        # Both are deterministic and use NER entity metadata written above.
        _set_status(AuditStatus.CLASSIFYING_RISK)
        actor_result, applicability_result = await asyncio.gather(
            asyncio.to_thread(classify_actor, raw_text, chunks),
            asyncio.to_thread(check_applicability, chunks),
        )

        logger.info(
            "applicability_determined",
            audit_id=audit_id,
            actor=actor_result.actor_type.value,
            is_high_risk=applicability_result.is_high_risk,
            is_prohibited=applicability_result.is_prohibited,
            annex_iii_categories=[m.category.value for m in applicability_result.annex_iii_matches],
        )

        # ── Stage 4: chunk classification (BERT/Ollama) ───────────────────────
        # Sequential Ollama classification is the biggest, previously-invisible
        # wait for large real documents (~150 chunks observed taking 5+ minutes
        # with zero progress shown). ETA starts as a rough guess from a
        # calibrated default rate the instant we know the chunk count, then
        # gets replaced by this run's own live-observed average after the
        # first chunk completes — same "start rough, refine live" pattern as
        # the ANALYSING stage below.
        _set_status(AuditStatus.CLASSIFYING_CHUNKS)
        _classify_t0 = _time.time()
        _DEFAULT_SEC_PER_CHUNK = 2.7  # calibrated from observed real-document runs
        _set_progress(
            chunks_done=0,
            chunks_total=len(chunks),
            estimated_seconds_remaining=round(len(chunks) * _DEFAULT_SEC_PER_CHUNK),
        )

        def _on_classify_progress(done: int, total: int) -> None:
            elapsed_in_stage = _time.time() - _classify_t0
            avg_per_chunk = elapsed_in_stage / done if done > 0 else _DEFAULT_SEC_PER_CHUNK
            eta = round(avg_per_chunk * max(0, total - done))
            _set_progress(chunks_done=done, chunks_total=total, estimated_seconds_remaining=eta)

        chunks, classifier_backend = await classify_chunks(chunks, ollama, on_progress=_on_classify_progress)

        # ── Stage 5: NER domain correction ───────────────────────────────────
        # Now that chunk.domain is set, correct UNRELATED chunks that NER
        # flagged as containing an unambiguous Article 9–15 reference.
        chunks = await asyncio.to_thread(apply_ner_domain_correction, chunks)
        _monitor.record_stage("classifying", _time.time() - _t0)

        # Ensure Ollama has the model loaded before firing 7 concurrent LangGraph calls
        await ollama.warmup()

        # RAG retrieval + per-article gap analysis
        _set_status(AuditStatus.ANALYSING)
        _t0 = _time.time()

        # Group chunks by domain
        domain_chunks: dict[ArticleDomain, list] = {d: [] for d in ArticleDomain}
        for chunk in chunks:
            if chunk.domain:
                domain_chunks[chunk.domain].append(chunk)

        applicable_articles = applicability_result.applicable_articles

        # Count articles that will actually need an LLM call (chunks present
        # AND applicable) — the rest finish near-instantly via gap_analyser.py's
        # own short-circuits, so only these drive real wait time. Used to turn
        # per-article completion timing into a live "N seconds remaining" ETA.
        _llm_articles_total = sum(
            1
            for article_num, domain in ARTICLE_DOMAINS.items()
            if domain_chunks.get(domain) and (not applicable_articles or article_num in applicable_articles)
        )
        _analysing_t0[0] = _time.time()
        _DEFAULT_SEC_PER_ARTICLE = 45  # calibrated from observed real gap-analysis runs
        _set_progress(
            articles_done=0,
            articles_total=len(ARTICLE_DOMAINS),
            estimated_seconds_remaining=round(_llm_articles_total * _DEFAULT_SEC_PER_ARTICLE),
        )

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

        # Phase 3 — evidence mapping (EU AI Act + GDPR, deterministic)
        _set_status(AuditStatus.MAPPING_EVIDENCE)
        evidence_map = await asyncio.to_thread(
            map_evidence,
            chunks,
            actor_result.actor_type,
            applicable_articles,
            applicability_result.gdpr_applicable_articles,
        )

        # Aggregate article scores into a ComplianceReport
        _set_status(AuditStatus.SCORING)
        emotion_flag = await check_emotion_recognition(chunks, applicability_result)
        report = await score_audit(
            article_scores=article_scores,
            chunks=chunks,
            audit_id=audit_id,
            source_files=filenames,
            language=primary_language,
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
        # Clean up all uploaded files
        for path in file_paths:
            try:
                os.remove(path)
            except OSError:
                pass
