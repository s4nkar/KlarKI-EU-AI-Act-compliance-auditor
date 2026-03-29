#!/usr/bin/env python3
"""Export fine-tuned BERT classifier to ONNX format.

Converts the trained deepset/gbert-base model to ONNX and validates that
outputs match PyTorch inference within a tolerance of 1e-4.

Usage:
    python scripts/export_onnx.py \
        --model-path training/bert_classifier \
        --output-path model_repository/bert_clause_classifier/1/model.onnx

    # Also export e5 embeddings model:
    python scripts/export_onnx.py \
        --model-path intfloat/multilingual-e5-small \
        --output-path model_repository/e5_embeddings/1/model.onnx \
        --model-type embeddings
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer


def export_to_onnx(
    model_path: str,
    output_path: str,
    model_type: str = "classifier",
    max_length: int = 128,
    opset_version: int = 17,
) -> None:
    """Export PyTorch BERT model to ONNX format.

    Args:
        model_path: Path to fine-tuned model directory or HuggingFace model ID.
        output_path: Target path for the ONNX model file.
        model_type: 'classifier' for sequence classification, 'embeddings' for encoder.
        max_length: Maximum sequence length for dummy input.
        opset_version: ONNX opset version (17 = Triton 24.02 compatible).
    """
    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"Loading model ({model_type}) from {model_path}")
    if model_type == "classifier":
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        model = AutoModel.from_pretrained(model_path)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  Device: {device}")

    # Dummy input for tracing
    dummy_text = ["This is a sample compliance document for ONNX export."]
    encoding = tokenizer(
        dummy_text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\nExporting to ONNX: {output_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"] if model_type == "classifier" else ["last_hidden_state"],
            dynamic_axes={
                "input_ids":      {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits":         {0: "batch_size"} if model_type == "classifier"
                                  else {"last_hidden_state": {0: "batch_size"}},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )
    print(f"  Exported successfully ({Path(output_path).stat().st_size / 1e6:.1f} MB)")

    # Validate: compare ONNX output to PyTorch output
    print("\nValidating ONNX output vs PyTorch…")
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        onnx_inputs = {
            "input_ids":      input_ids.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy(),
        }
        onnx_out = sess.run(None, onnx_inputs)[0]

        with torch.no_grad():
            pt_out = model(input_ids, attention_mask)
            if model_type == "classifier":
                pt_arr = pt_out.logits.cpu().numpy()
            else:
                pt_arr = pt_out.last_hidden_state.cpu().numpy()

        max_diff = float(np.abs(onnx_out - pt_arr).max())
        print(f"  Max absolute difference: {max_diff:.2e}")
        if max_diff < 1e-4:
            print("  PASS — outputs match within tolerance 1e-4")
        else:
            print(f"  WARN — difference {max_diff:.2e} exceeds 1e-4 tolerance")

    except ImportError:
        print("  SKIP validation — onnxruntime not installed")
        print("  Install with: pip install onnxruntime")

    print(f"\nDone. Place {output_path} in the Triton model repository.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export BERT to ONNX for Triton deployment")
    parser.add_argument("--model-path",   required=True, help="HuggingFace model path or ID")
    parser.add_argument("--output-path",  required=True, help="Target .onnx file path")
    parser.add_argument("--model-type",   default="classifier", choices=["classifier", "embeddings"])
    parser.add_argument("--max-length",   type=int, default=128)
    parser.add_argument("--opset",        type=int, default=17)
    args = parser.parse_args()

    export_to_onnx(
        model_path=args.model_path,
        output_path=args.output_path,
        model_type=args.model_type,
        max_length=args.max_length,
        opset_version=args.opset,
    )
