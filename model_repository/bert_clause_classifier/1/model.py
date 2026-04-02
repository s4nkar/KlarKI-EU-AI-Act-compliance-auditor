"""Triton Python backend for BERT clause classifier ONNX model.

Loads bert_clause_classifier/1/model.onnx via onnxruntime and serves
classification logits for 8 EU AI Act article domains.
"""

import json
import os
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args: dict) -> None:
        import onnxruntime as ort

        model_dir = Path(args["model_repository"]) / args["model_version"]
        onnx_path = str(model_dir / "model.onnx")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        active = self.session.get_providers()[0]
        pb_utils.Logger.log_info(f"BERT classifier loaded from {onnx_path} ({active})")

    def execute(self, requests: list) -> list:
        responses = []
        for request in requests:
            input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids").as_numpy()
            attention_mask = pb_utils.get_input_tensor_by_name(request, "attention_mask").as_numpy()

            ort_inputs = {
                "input_ids":      input_ids.astype(np.int64),
                "attention_mask": attention_mask.astype(np.int64),
            }
            logits = self.session.run(["logits"], ort_inputs)[0]  # (batch, 8)

            out_tensor = pb_utils.Tensor("logits", logits.astype(np.float32))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def finalize(self) -> None:
        pb_utils.Logger.log_info("BERT classifier unloaded")
