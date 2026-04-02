"""Triton Python backend for multilingual-e5-small ONNX model.

Loads e5_embeddings/1/model.onnx via onnxruntime and returns
mean-pooled, L2-normalised 384-dim embeddings.
"""

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
        pb_utils.Logger.log_info(f"e5-small embeddings loaded from {onnx_path} ({active})")

    def execute(self, requests: list) -> list:
        responses = []
        for request in requests:
            input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids").as_numpy()
            attention_mask = pb_utils.get_input_tensor_by_name(request, "attention_mask").as_numpy()

            ort_inputs = {
                "input_ids":      input_ids.astype(np.int64),
                "attention_mask": attention_mask.astype(np.int64),
            }
            hidden = self.session.run(["last_hidden_state"], ort_inputs)[0]  # (batch, seq, 384)

            # Mean-pool over non-padding tokens, then L2-normalise
            mask = attention_mask[:, :, np.newaxis].astype(np.float32)
            pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1e-9)
            norms = np.linalg.norm(pooled, axis=-1, keepdims=True).clip(min=1e-9)
            pooled = (pooled / norms).astype(np.float32)

            out_tensor = pb_utils.Tensor("last_hidden_state", pooled)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def finalize(self) -> None:
        pb_utils.Logger.log_info("e5-small embeddings unloaded")
