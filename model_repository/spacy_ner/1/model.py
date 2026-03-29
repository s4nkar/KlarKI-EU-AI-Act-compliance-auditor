"""Triton Python backend for spaCy NER model.

Receives raw text strings, runs spaCy NER, returns JSON-serialised entities.
Deploy the trained spaCy model at the path set in SPACY_MODEL_PATH.
"""

import json
import os
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils

_SPACY_MODEL_PATH = os.environ.get(
    "SPACY_MODEL_PATH",
    "/models/spacy_ner/spacy_model",
)


class TritonPythonModel:
    """Triton Python backend lifecycle class."""

    def initialize(self, args: dict) -> None:
        """Load the spaCy pipeline once at startup."""
        import spacy

        model_path = Path(_SPACY_MODEL_PATH)
        if model_path.exists():
            self.nlp = spacy.load(str(model_path))
        else:
            # Fallback to base German model if fine-tuned model is not yet present
            self.nlp = spacy.load("de_core_news_sm")

        pb_utils.Logger.log_info(f"spaCy NER model loaded from {_SPACY_MODEL_PATH}")

    def execute(self, requests: list) -> list:
        """Run NER on each text in the batch."""
        responses = []

        for request in requests:
            texts_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            texts = [t.decode("utf-8") for t in texts_tensor.as_numpy().flatten()]

            results = []
            for text in texts:
                doc = self.nlp(text)
                entities = [
                    {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
                    for ent in doc.ents
                ]
                results.append(json.dumps(entities, ensure_ascii=False))

            out_tensor = pb_utils.Tensor(
                "entities",
                np.array(results, dtype=object),
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def finalize(self) -> None:
        """Cleanup on shutdown."""
        pb_utils.Logger.log_info("spaCy NER model unloaded")
