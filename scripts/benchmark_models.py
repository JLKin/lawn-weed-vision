"""Compare model latency and mask agreement on fixed images; not a labeled accuracy evaluation."""

import argparse
import gc
import json
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps

from weed_vision.config import Settings
from weed_vision.preprocessing import prepare_image
from weed_vision.yolov5_runtime import sha256_file
from weed_vision.yolov5_segmenter import Yolov5Segmenter


def percentiles(values):
    return {"p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)), "samples": len(values)}


def compare_masks(reference_names, reference, names, masks):
    # Maximum total IoU class-constrained pairing; reports misses separately.
    from scipy.optimize import linear_sum_assignment
    scores = np.zeros((len(reference), len(masks)))
    for i, first in enumerate(reference):
        for j, second in enumerate(masks):
            if reference_names[i] == names[j]:
                union = np.count_nonzero(first | second)
                scores[i, j] = np.count_nonzero(first & second) / union if union else 0
    if scores.size:
        rows, columns = linear_sum_assignment(-scores)
        matched = [(int(i), int(j), float(scores[i, j])) for i, j in zip(rows, columns) if scores[i, j] >= .5]
    else:
        matched = []
    return dict(reference_count=len(reference), candidate_count=len(masks), matched_iou_ge_0_5=len(matched),
                unmatched_reference=len(reference) - len(matched), unmatched_candidate=len(masks) - len(matched),
                mean_matched_mask_iou=float(np.mean([x[2] for x in matched])) if matched else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, action="append", help="Repeat: first model is comparison reference")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--http-url", help="Optional independent benchmark of currently running API")
    args = parser.parse_args()
    if not 1 <= args.max_images <= 1000 or not 1 <= args.repeats <= 100 or not 1 <= args.warmup <= 100:
        parser.error("Invalid image/repetition/warmup counts")
    paths = sorted(p for p in args.images_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})[:args.max_images]
    if not paths: parser.error("No images found")
    import torch
    report = dict(image_manifest=[dict(name=p.name, sha256=sha256_file(p)) for p in paths],
                  task_accuracy_validated=False, note="Output agreement is not ground-truth accuracy; latency excludes model load.", models=[])
    with tempfile.TemporaryDirectory(prefix="weed-benchmark-") as temp:
        for model_index, path in enumerate(args.model):
            model = Yolov5Segmenter(Settings(_env_file=None, backend="yolov5", model_path=str(path), device=args.device,
                                             half=False, image_size=args.image_size))
            model.load()
            device = model._model.device
            def sync():
                if device.type == "cuda": torch.cuda.synchronize(device)
            if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
            with Image.open(paths[0]) as source:
                probe = ImageOps.exif_transpose(source).convert("RGB")
            for _ in range(args.warmup): model.predict(probe)
            probe.close()
            network_times, pipeline_times, per_image = [], [], []
            for image_index, image_path in enumerate(paths):
                with Image.open(image_path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                tensor = torch.from_numpy(prepare_image(image, args.image_size)).to(device)
                tensor = tensor.half() if model._model.fp16 else tensor.float()
                with torch.inference_mode():
                    for _ in range(args.repeats):
                        sync(); start = perf_counter()
                        raw = model._model(tensor)
                        sync(); network_times.append((perf_counter() - start) * 1000)
                        del raw
                    for _ in range(args.repeats):
                        sync(); start = perf_counter()
                        detections, masks = model.predict_with_masks(image)
                        sync(); pipeline_times.append((perf_counter() - start) * 1000)
                names = [d.class_name for d in detections]
                cache = Path(temp) / f"{image_index}.npz"
                if model_index == 0:
                    np.savez_compressed(cache, names=np.array(names, dtype=str), masks=np.array(masks, dtype=bool))
                else:
                    with np.load(cache, allow_pickle=False) as reference:
                        per_image.append(dict(image=image_path.name, **compare_masks(reference["names"].tolist(), reference["masks"], names, masks)))
                image.close()
                del tensor, masks
            report["models"].append(dict(model=str(path), model_version=model.model_version,
                network=percentiles(network_times), preprocessing_inference_postprocessing=percentiles(pipeline_times),
                peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                mask_agreement=per_image))
            del model
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    if args.http_url:
        import httpx
        values, versions = [], set()
        with httpx.Client(timeout=180) as client:
            for p in paths:
                mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}[p.suffix.lower()]
                data = p.read_bytes()
                for iteration in range(args.warmup + args.repeats):
                    start = perf_counter()
                    response = client.post(args.http_url.rstrip("/") + "/v1/weed-segmentation",
                        data={"camera_id": "benchmark"}, files={"file": (p.name, data, mime)})
                    response.raise_for_status()
                    payload = response.json()
                    if iteration >= args.warmup: values.append((perf_counter() - start) * 1000)
                    versions.add(payload["model_version"])
        report["independent_http_service"] = dict(**percentiles(values), model_versions=sorted(versions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Benchmark written: {args.output}; labeled task accuracy still requires separate evaluation.")


if __name__ == "__main__":
    main()
