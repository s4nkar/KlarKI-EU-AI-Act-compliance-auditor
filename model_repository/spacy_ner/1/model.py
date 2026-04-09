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
    "/spacy_model/model-final",
)


class TritonPythonModel:
    """Triton Python backend lifecycle class."""

    def initialize(self, args: dict) -> None:
        """Load the spaCy pipeline once at startup."""
        import spacy

        model_path = Path(_SPACY_MODEL_PATH)
        if model_path.exists():
            self.nlp = spacy.load(str(model_path))
            pb_utils.Logger.log_info(f"Loaded fine-tuned spaCy NER from {model_path}")
        else:
            # Fallback: blank German model (no pre-trained weights needed)
            import spacy as _spacy
            self.nlp = _spacy.blank("de")
            pb_utils.Logger.log_warning(
                f"Fine-tuned model not found at {model_path}, using blank German model"
            )

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
