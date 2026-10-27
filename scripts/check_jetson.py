"""Read-only Jetson/CUDA preflight; optionally lock already-working GPU dependencies."""

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import sys


def inspect_environment() -> tuple[dict, list[str]]:
    report = {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version()}
    errors = []
    for name, path in {"board": "/proc/device-tree/model", "l4t": "/etc/nv_tegra_release"}.items():
        source = Path(path)
        report[name] = source.read_text(errors="replace").strip("\x00\n") if source.exists() else None
    if report["system"] != "Linux" or report["machine"] != "aarch64" or not report["l4t"]:
        errors.append("Expected a Jetson running Linux ARM64 with NVIDIA L4T installed.")
    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        errors.append("This project requires Python 3.10-3.13.")
    try:
        import numpy as np
        import torch
        import torchvision

        report.update(torch=torch.__version__, torchvision=torchvision.__version__, numpy=np.__version__, cuda=torch.version.cuda)
        report["cuda_available"] = torch.cuda.is_available()
        if not report["cuda_available"]:
            errors.append("CUDA is unavailable; install JetPack-compatible NVIDIA PyTorch/torchvision.")
        else:
            report["gpu"] = torch.cuda.get_device_name(0)
            torch.from_numpy(np.zeros((2, 2), dtype=np.float32)).to("cuda").sum().item()
            boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=torch.float32, device="cuda")
            scores = torch.tensor([0.9, 0.8], device="cuda")
            kept = torchvision.ops.nms(boxes, scores, 0.5)
            if kept.cpu().tolist() != [0]:
                raise RuntimeError("Unexpected CUDA NMS result.")
            report["cuda_nms"] = "passed"
    except Exception as exc:
        errors.append(f"PyTorch/torchvision/NumPy compatibility check failed: {exc}")
    try:
        import tensorrt

        report["tensorrt"] = tensorrt.__version__
        major = int(tensorrt.__version__.split(".")[0])
        report["qdq_builder_version_supported"] = major == 10
        if not 8 <= major <= 10:
            errors.append("Original YOLOv5 runtime targets TensorRT 8-10; use a supported version supplied by JetPack.")
    except Exception as exc:
        errors.append(f"JetPack TensorRT Python bindings are unavailable: {exc}")
    report["ready_for_jetson_validation"] = not errors
    return report, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-constraints", type=Path, help="Pin the existing working torch/torchvision/numpy versions")
    parser.add_argument("--output", type=Path, help="Save the environment report")
    args = parser.parse_args()
    report, errors = inspect_environment()
    report["errors"] = errors
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if errors:
        return 1
    if args.write_constraints:
        pins = [f"{name}=={metadata.version(name)}" for name in ("torch", "torchvision", "numpy")]
        args.write_constraints.parent.mkdir(parents=True, exist_ok=True)
        args.write_constraints.write_text("# Generated on the target Jetson; preserve its working GPU stack.\n" + "\n".join(pins) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
