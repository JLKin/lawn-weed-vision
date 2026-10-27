"""Build FP16 or explicit Q/DQ INT8 engines on the target Jetson, using TensorRT 10.x."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

from weed_vision.config import Settings
from weed_vision.quantization import verify_onnx
from weed_vision.yolov5_runtime import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--precision", choices=["fp16", "int8"], required=True)
    parser.add_argument("--workspace-gib", type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument("--repo", default="vendor/yolov5")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64" or not Path("/etc/nv_tegra_release").exists():
        parser.error("Build engines on the target Jetson; this host cannot validate Jetson TensorRT engines")
    from scripts.check_jetson import inspect_environment
    report, errors = inspect_environment()
    if errors:
        raise SystemExit("\n".join(errors))
    import tensorrt as trt
    if trt.__version__.split(".")[0] != "10":
        parser.error("This Q/DQ builder targets TensorRT 10.x supplied by JetPack; do not replace the GPU stack")
    output = args.output.resolve()
    sidecar = output.with_suffix(".engine.json")
    if output.suffix != ".engine" or output.exists() or sidecar.exists():
        parser.error("Choose a new .engine path; preserve existing verified engines")
    info = json.loads(args.onnx.with_suffix(".onnx.json").read_text(encoding="utf-8"))
    graph = verify_onnx(args.onnx, info)
    qcount = sum(n.op_type == "QuantizeLinear" for n in graph.graph.node)
    if args.precision == "int8" and (qcount == 0 or info.get("precision_request") != "int8"):
        parser.error("INT8 requires a calibrated Q/DQ ONNX and matching metadata")
    if args.precision == "fp16" and qcount:
        parser.error("FP16 comparison must use the original FP32 ONNX")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    onnx_parser = trt.OnnxParser(network, logger)
    if not onnx_parser.parse_from_file(str(args.onnx.resolve())):
        raise RuntimeError("\n".join(str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gib << 30)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    # Explicit Q/DQ defines quantized precision. FP16 is allowed for floating fallback layers.
    config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build failed; inspect parser/build diagnostics")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("New engine failed to deserialize")
    inspection = engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    int8_observed = "int8" in inspection.lower()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".layers.json").write_text(inspection, encoding="utf-8")
    if args.precision == "int8" and not int8_observed:
        raise RuntimeError("Inspector found no INT8 tensor format; inspect .layers.json before treating this as accelerated INT8")
    output.write_bytes(bytes(serialized))
    info.update(engine_sha256=sha256_file(output), precision_request=args.precision,
                environment=report, int8_tensor_format_observed=int8_observed,
                built_at_utc=datetime.now(timezone.utc).isoformat(), workspace_gib=args.workspace_gib)
    sidecar.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    del engine, runtime, serialized, onnx_parser, network, config, builder
    import gc
    gc.collect()
    from weed_vision.yolov5_segmenter import Yolov5Segmenter
    candidate = Yolov5Segmenter(Settings(_env_file=None, model_path=str(output), backend="yolov5",
        yolov5_repo=args.repo, device="0", image_size=info["image_size"], half=True))
    candidate.load()
    print(f"Engine load and warmup passed: {output}")
    print("Inspect layer report; mixed precision is expected. Measure latency and validate task accuracy before use.")


if __name__ == "__main__":
    main()
