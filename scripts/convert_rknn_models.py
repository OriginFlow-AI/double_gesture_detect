#!/usr/bin/env python3
"""Convert the official MediaPipe ONNX pair to RK3588 FP16 RKNN models.

This is a host-side build tool. The deployed camera application remains C++ only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

import onnx
from onnx import helper, TensorProto
from rknn.api import RKNN


ROOT = Path(__file__).resolve().parents[1]


CONTRACTS = (
    {
        "name": "palm_detection",
        "source": ROOT / "models/opencv_zoo/palm_detection_mediapipe_2023feb.onnx",
        "source_sha256": "78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c",
        "output": "palm_detection_mediapipe_2023feb_fp16.rknn",
        "input": {"name": "input_1", "shape": [1, 192, 192, 3]},
        "outputs": [
            {"index": 0, "name": "Identity", "shape": [1, 2016, 18]},
            {"index": 1, "name": "Identity_1", "shape": [1, 2016, 1]},
        ],
    },
    {
        "name": "handpose_estimation",
        "source": ROOT / "models/opencv_zoo/handpose_estimation_mediapipe_2023feb.onnx",
        "source_sha256": "db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c",
        "output": "handpose_estimation_mediapipe_2023feb_fp16.rknn",
        "input": {"name": "input_1", "shape": [1, 224, 224, 3]},
        "outputs": [
            {"index": 0, "name": "Identity", "shape": [1, 63]},
            {"index": 1, "name": "Identity_1", "shape": [1, 1]},
            {"index": 2, "name": "Identity_2", "shape": [1, 1]},
            {"index": 3, "name": "Identity_3", "shape": [1, 63]},
        ],
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_shape(value: onnx.ValueInfoProto) -> list[int]:
    shape: list[int] = []
    for dimension in value.type.tensor_type.shape.dim:
        if not dimension.HasField("dim_value"):
            raise RuntimeError(f"dynamic dimension is not supported: {value.name}")
        shape.append(dimension.dim_value)
    return shape


def validate_onnx(contract: dict) -> int:
    path = contract["source"]
    if not path.is_file():
        raise RuntimeError(f"missing ONNX source model: {path}")
    actual_sha = sha256(path)
    if actual_sha != contract["source_sha256"]:
        raise RuntimeError(
            f"unexpected SHA-256 for {path}: {actual_sha}; "
            f"expected {contract['source_sha256']}"
        )

    model = onnx.load(path)
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1:
        raise RuntimeError(f"{path} must contain exactly one input")
    model_input = model.graph.input[0]
    expected_input = contract["input"]
    if model_input.name != expected_input["name"]:
        raise RuntimeError(f"unexpected input name in {path}: {model_input.name}")
    if tensor_shape(model_input) != expected_input["shape"]:
        raise RuntimeError(f"unexpected input shape in {path}")
    if model_input.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise RuntimeError(f"{path} input must be float32")

    actual_outputs = list(model.graph.output)
    if len(actual_outputs) != len(contract["outputs"]):
        raise RuntimeError(f"unexpected output count in {path}")
    for actual, expected in zip(actual_outputs, contract["outputs"], strict=True):
        if actual.name != expected["name"] or tensor_shape(actual) != expected["shape"]:
            raise RuntimeError(
                f"unexpected output contract in {path}: "
                f"{actual.name} {tensor_shape(actual)}"
            )
        if actual.type.tensor_type.elem_type != TensorProto.FLOAT:
            raise RuntimeError(f"{path} output {actual.name} must be float32")

    versions = {item.domain: item.version for item in model.opset_import}
    return versions.get("", 0)


def require_success(status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"{operation} failed with RKNN status {status}")


def write_nchw_input_adapter(contract: dict, destination: Path) -> None:
    """Wrap an NHWC model in a standard NCHW input for RKNN preprocessing."""
    model = onnx.load(contract["source"])
    graph_input = model.graph.input[0]
    external_name = graph_input.name
    internal_name = f"{external_name}__nhwc"
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == external_name:
                node.input[index] = internal_name

    transpose = helper.make_node(
        "Transpose",
        inputs=[external_name],
        outputs=[internal_name],
        name="double_ok_nchw_to_nhwc",
        perm=[0, 2, 3, 1],
    )
    model.graph.node.insert(0, transpose)
    height = contract["input"]["shape"][1]
    width = contract["input"]["shape"][2]
    dimensions = graph_input.type.tensor_type.shape.dim
    for dimension, value in zip(dimensions, [1, 3, height, width], strict=True):
        dimension.ClearField("dim_param")
        dimension.dim_value = value
    onnx.checker.check_model(model)
    onnx.save(model, destination)


def convert(contract: dict, output_dir: Path) -> dict:
    opset = validate_onnx(contract)
    output_path = output_dir / contract["output"]
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_dir, delete=False
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    adapter = tempfile.NamedTemporaryFile(
        prefix=f".{contract['name']}.", suffix=".nchw.onnx", dir=output_dir, delete=False
    )
    adapter_path = Path(adapter.name)
    adapter.close()
    try:
        write_nchw_input_adapter(contract, adapter_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        adapter_path.unlink(missing_ok=True)
        raise

    try:
        rknn = RKNN(verbose=False)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        adapter_path.unlink(missing_ok=True)
        raise
    try:
        require_success(
            rknn.config(
                target_platform="rk3588",
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                float_dtype="float16",
                optimization_level=3,
            ),
            f"rknn.config({contract['name']})",
        )
        require_success(
            rknn.load_onnx(model=str(adapter_path)),
            f"rknn.load_onnx({contract['name']})",
        )
        require_success(
            rknn.build(do_quantization=False),
            f"rknn.build({contract['name']})",
        )
        require_success(
            rknn.export_rknn(str(temporary_path)),
            f"rknn.export_rknn({contract['name']})",
        )
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        rknn.release()
        adapter_path.unlink(missing_ok=True)

    if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"RKNN export is empty: {temporary_path}")
    os.replace(temporary_path, output_path)
    output_path.chmod(0o644)
    return {
        "name": contract["name"],
        "source": str(contract["source"].relative_to(ROOT)),
        "source_sha256": contract["source_sha256"],
        "source_opset": opset,
        "artifact": output_path.name,
        "artifact_sha256": sha256(output_path),
        "artifact_size": output_path.stat().st_size,
        "input": {
            **contract["input"],
            "runtime_type": "uint8",
            "layout": "NHWC",
            "color": "RGB",
            "normalization": "model performs value / 255",
            "rknn_logical_shape": [
                1,
                3,
                contract["input"]["shape"][1],
                contract["input"]["shape"][2],
            ],
            "adapter": "NCHW input followed by Transpose(0,2,3,1)",
        },
        "outputs": contract["outputs"],
    }


def main() -> int:
    if not hasattr(onnx, "mapping"):
        raise RuntimeError(
            "RKNN-Toolkit2 2.3.2 requires ONNX 1.16.2; install "
            "scripts/requirements-rknn.txt"
        )
    parser = argparse.ArgumentParser(
        description="Build the RK3588 FP16 MediaPipe RKNN model pair"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "models/rk3588",
        help="RKNN artifact directory (default: models/rk3588)",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    artifacts = [convert(contract, output_dir) for contract in CONTRACTS]
    manifest = {
        "schema": "double_ok_rknn_models_v1",
        "target_platform": "rk3588",
        "precision": "FP16",
        "quantized": False,
        "rknn_toolkit2": importlib.metadata.version("rknn-toolkit2"),
        "preprocessing": "RGB uint8 NHWC; mean=[0,0,0]; std=[255,255,255]",
        "models": artifacts,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"RKNN models ready: {output_dir}")
    for artifact in artifacts:
        print(f"  {artifact['artifact']}: {artifact['artifact_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
