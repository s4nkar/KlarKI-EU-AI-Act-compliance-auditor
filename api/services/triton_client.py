"""Triton Inference Server gRPC client.

Provides an async interface to the Triton ensemble:
BERT clause classifier + e5 embeddings + spaCy NER.
Only used when USE_TRITON=true in config.
"""

from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger()

# Label index → ArticleDomain value (must match training order in train_classifier.py)
_IDX_TO_LABEL = [
    "risk_management",
    "data_governance",
    "technical_documentation",
    "record_keeping",
    "transparency",
    "human_oversight",
    "security",
    "unrelated",
]

_TOKENIZER_CACHE: dict[str, Any] = {}


def _get_tokenizer(model_name: str = "deepset/gbert-base"):
    """Load and cache the tokenizer (loaded once per process)."""
    if model_name not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer
        _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _TOKENIZER_CACHE[model_name]


class TritonClient:
    """Async gRPC client for Triton Inference Server.

    Args:
        host: Triton server hostname (e.g. 'klarki-triton').
        grpc_port: Triton gRPC port (default 8001).
        tokenizer_model: HuggingFace model name for tokenization (client-side).
    """

    def __init__(
        self,
        host: str,
        grpc_port: int = 8001,
        tokenizer_model: str = "deepset/gbert-base",
    ) -> None:
        self._address = f"{host}:{grpc_port}"
        self._tokenizer_model = tokenizer_model
        self._client = None

    async def _get_client(self):
        """Lazily initialise the gRPC async client."""
        if self._client is None:
            import tritonclient.grpc.aio as grpcclient
            self._client = grpcclient.InferenceServerClient(url=self._address)
        return self._client

    async def classify(self, texts: list[str]) -> list[str]:
        """Classify texts using the BERT clause classifier.

        Tokenizes client-side, sends input_ids + attention_mask to Triton,
        returns argmax label strings per text.

        Args:
            texts: List of text strings to classify.

        Returns:
            List of ArticleDomain value strings (same order as input).
        """
        import tritonclient.grpc.aio as grpcclient

        tokenizer = _get_tokenizer(self._tokenizer_model)
        encoding = tokenizer(
            texts,
            return_tensors="np",
            max_length=128,
            padding="max_length",
            truncation=True,
        )

        input_ids      = encoding["input_ids"].astype(np.int64)
        attention_mask = encoding["attention_mask"].astype(np.int64)

        inputs = [
            grpcclient.InferInput("input_ids",      input_ids.shape,      "INT64"),
            grpcclient.InferInput("attention_mask",  attention_mask.shape,  "INT64"),
        ]
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)

        outputs = [grpcclient.InferRequestedOutput("logits")]

        client = await self._get_client()
        response = await client.infer(
            model_name="bert_clause_classifier",
            inputs=inputs,
            outputs=outputs,
        )

        logits = response.as_numpy("logits")           # (batch, 8)
        pred_indices = np.argmax(logits, axis=-1)      # (batch,)
        labels = [_IDX_TO_LABEL[int(i)] for i in pred_indices]

        logger.debug("triton_classify_done", batch_size=len(texts))
        return labels

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate 384-dim embeddings via Triton e5 model.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of 384-dim float vectors (mean-pooled last hidden state).
        """
        import tritonclient.grpc.aio as grpcclient

        tokenizer = _get_tokenizer("intfloat/multilingual-e5-small")
        encoding = tokenizer(
            texts,
            return_tensors="np",
            max_length=128,
            padding="max_length",
            truncation=True,
        )

        input_ids      = encoding["input_ids"].astype(np.int64)
        attention_mask = encoding["attention_mask"].astype(np.int64)

        inputs = [
            grpcclient.InferInput("input_ids",      input_ids.shape,      "INT64"),
            grpcclient.InferInput("attention_mask",  attention_mask.shape,  "INT64"),
        ]
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)

        outputs = [grpcclient.InferRequestedOutput("last_hidden_state")]

        client = await self._get_client()
        response = await client.infer(
            model_name="e5_embeddings",
            inputs=inputs,
            outputs=outputs,
        )

        hidden = response.as_numpy("last_hidden_state")  # (batch, seq, 384)
        # Mean pool over non-padding tokens
        mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1e-9)
        # L2 normalise
        norms = np.linalg.norm(pooled, axis=-1, keepdims=True).clip(min=1e-9)
        pooled = pooled / norms

        logger.debug("triton_embed_done", batch_size=len(texts))
        return pooled.tolist()

    async def ner(self, texts: list[str]) -> list[dict[str, list[str]]]:
        """Run spaCy NER via Triton Python backend.

        Args:
            texts: List of text strings (pre-truncated to model's preferred length).

        Returns:
            One dict per text: {label: [entity_text, ...]}
        """
        import json
        import tritonclient.grpc.aio as grpcclient

        np_texts = np.array([[t.encode("utf-8")] for t in texts], dtype=object)

        inputs = [grpcclient.InferInput("text", np_texts.shape, "BYTES")]
        inputs[0].set_data_from_numpy(np_texts)

        outputs = [grpcclient.InferRequestedOutput("entities")]

        client = await self._get_client()
        response = await client.infer(
            model_name="spacy_ner",
            inputs=inputs,
            outputs=outputs,
        )

        raw = response.as_numpy("entities").flatten()
        results: list[dict[str, list[str]]] = []
        for item in raw:
            decoded = item.decode("utf-8") if isinstance(item, bytes) else item
            entities_list = json.loads(decoded)
            grouped: dict[str, list[str]] = {}
            for ent in entities_list:
                grouped.setdefault(ent["label"], []).append(ent["text"])
            results.append(grouped)

        logger.debug("triton_ner_done", batch_size=len(texts))
        return results

    async def health_check(self) -> bool:
        """Check Triton server readiness via gRPC.

        Returns:
            True if server is ready, False otherwise.
        """
        try:
            client = await self._get_client()
            ready = await client.is_server_ready()
            return bool(ready)
        except Exception as exc:
            logger.warning("triton_health_check_failed", error=str(exc))
            return False
