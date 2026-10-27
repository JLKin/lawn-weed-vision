"""Quantize exported YOLOv5 ONNX using representative camera images (Windows or Jetson)."""

import argparse
from pathlib import Path
from weed_vision.quantization import quantize_onnx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=300)
    parser.add_argument("--exclude-prefix", action="append", default=[])
    args = parser.parse_args()
    info = quantize_onnx(args.onnx, args.output, args.calibration_dir, args.max_images, args.exclude_prefix)
    print(f"Q/DQ graph ready: {info['quantize_node_count']} Q nodes, {info['calibration_image_count']} images.")
    print("This is a calibration artifact, not a verified accuracy or TensorRT speed result.")


if __name__ == "__main__":
    main()
