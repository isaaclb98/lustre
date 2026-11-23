"""
benchmarks/embed_batch_sweep.py — find the optimal batch size for
embedding throughput on this hardware.

Why: indexing 1M+ images is the hot path. Doubling batch size can
2-4x throughput on GPU (better SM occupancy, fewer kernel launches,
fewer Python overhead trips). But it can also blow VRAM (OOM at
some batch size) or even regress throughput (cache thrash, smaller
sweet spot than expected).

Sweep strategy:
  - For each batch_size in BATCH_SIZES:
      - Load the encoder (once, cached)
      - Warm up: encode one batch of the target size (first call
        triggers lazy CUDA kernel autotuning; results aren't
        representative)
      - Measure: encode N_MEASURE batches back-to-back, sum wall
        time, divide
      - Capture peak VRAM via nvidia-smi before/after
      - On OOM, mark and continue with smaller sizes
  - Print a table sorted by throughput (descending). Winner is the
    largest batch that fits AND has the highest throughput.

Usage:
    source .venv-indexer/bin/activate
    python benchmarks/embed_batch_sweep.py \
        --source /mnt/nas-main/images/kpop \
        --num-images 200 \
        --model C-RADIOv4-SO400M \
        --device cuda
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING"))
logger = logging.getLogger(__name__)


# Batch sizes to try. Spans small (CPU-friendly default) up to what
# might exhaust 10GB VRAM with ViT-L at 256x256. Sorted ascending;
# we OOM-skip anything that fails.
DEFAULT_BATCH_SIZES = (4, 8, 12, 16, 24, 32, 48, 64, 96, 128)


def _load_real_images(
    source: Path,
    n: int,
    target_resolution: int,
    *,
    max_depth: int = 0,
) -> list[Image.Image]:
    """Load up to `n` real images from `source`, scaled to the model's
    resolution. Pre-resizing isolates the embedding throughput from
    PIL decode (which the indexer also does, but at a fixed cost per
    image that doesn't change with batch size).

    `max_depth=0` (default) samples from the top-level directory only
    — fast on million-file SMB mounts because it doesn't recurse. The
    indexer's own scan IS recursive, but the embedding kernel cost
    is invariant to image location, so a flat sample is
    representative for this benchmark.
    """
    from indexer import scan as scan_mod
    if max_depth == 0:
        # Non-recursive top-level sample. orders of magnitude faster
        # than scan_mod.snapshot on deep SMB trees.
        paths: list[Path] = []
        with os.scandir(source) as it:
            for entry in it:
                if entry.is_file() and scan_mod.is_image_suffix(Path(entry.name).suffix):
                    paths.append(Path(entry.path))
                if len(paths) >= n * 4:  # over-sample in case some fail to load
                    break
    else:
        paths = scan_mod.snapshot(source, progress_every=100_000)
    if len(paths) > n:
        paths = paths[:n]
    if not paths:
        raise SystemExit(f"no images found under {source}")
    print(f"  loaded {len(paths)} candidate paths; pre-scaling to {target_resolution}x{target_resolution}")
    images: list[Image.Image] = []
    for p in paths:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im = im.resize((target_resolution, target_resolution), Image.Resampling.LANCZOS)
                images.append(im.copy())
        except Exception as exc:  # noqa: BLE001 — benchmark skips unreadable files; any error is logged and skipped
            logger.warning("skip %s: %s", p, exc)
        if len(images) >= n:
            break
    if not images:
        raise SystemExit(f"no loadable images found under {source}")
    return images


def _vram_mib() -> int | None:
    """Peak VRAM in MiB via torch.cuda, or None if not available."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() // (1024 * 1024)


def _reset_vram_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _encode_batches(
    encoder, images: list[Image.Image], batch_size: int
) -> tuple[float, int]:
    """Encode `images` in chunks of `batch_size`. Returns
    (elapsed_seconds, num_batches). On OOM, raises RuntimeError."""
    _reset_vram_peak()
    t0 = time.perf_counter()
    n_batches = 0
    for i in range(0, len(images), batch_size):
        chunk = images[i : i + batch_size]
        encoder.embed_batch(chunk)
        n_batches += 1
    torch.cuda.synchronize()  # ensure all kernels finished before timing
    return time.perf_counter() - t0, n_batches


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory containing real images to time against",
    )
    p.add_argument(
        "--num-images",
        type=int,
        default=200,
        help="How many images to load (default: 200, enough for ~3 "
             "batches at batch_size=64)",
    )
    p.add_argument(
        "--model",
        type=str,
        default="C-RADIOv4-SO400M",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--batch-sizes",
        type=int,
        nargs="*",
        default=list(DEFAULT_BATCH_SIZES),
    )
    p.add_argument(
        "--warmup-batches",
        type=int,
        default=2,
        help="Number of warmup batches per size (results discarded). "
             "First CUDA call autotunes kernels; second warmup picks up "
             "any compilation/lazy-init overhead.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Source: {args.source}")
    print(f"Model: {args.model} on {args.device}")
    print(f"Sweep batch sizes: {args.batch_sizes}")
    print()

    # Load encoder once. The first model load downloads the weights
    # if they're not cached (~3GB) — we surface this so users know
    # what to expect on a cold run.
    print("Loading encoder (downloads model on first run, ~3 GB)...")
    t0 = time.perf_counter()
    from indexer.vision_encoder import VisionEncoder
    encoder = VisionEncoder(arch=args.model, device=args.device)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    # Load and pre-resize images.
    print(f"Loading up to {args.num_images} images from {args.source}...")
    images = _load_real_images(args.source, args.num_images, encoder.resolution)
    print(f"  {len(images)} images ready\n")

    # Free CPU-side encode memory before sweep so it doesn't
    # contaminate VRAM measurements.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results: list[dict] = []
    for bs in args.batch_sizes:
        # Skip sizes larger than the dataset (no useful measurement).
        if bs > len(images):
            continue
        try:
            # Warmup.
            for _ in range(args.warmup_batches):
                _encode_batches(encoder, images, bs)
            # Measure.
            elapsed, n_batches = _encode_batches(encoder, images, bs)
        except torch.cuda.OutOfMemoryError as exc:
            print(f"  batch_size={bs:>4}  OOM ({exc})")
            torch.cuda.empty_cache()
            continue
        except RuntimeError as exc:
            # PyTorch wraps some OOMs in RuntimeError. Catch broadly.
            if "out of memory" in str(exc).lower() or "OOM" in str(exc):
                print(f"  batch_size={bs:>4}  OOM ({exc})")
                torch.cuda.empty_cache()
                continue
            raise

        throughput = len(images) / elapsed
        vram = _vram_mib()
        results.append({
            "batch_size": bs,
            "throughput": throughput,
            "ms_per_image": (elapsed / len(images)) * 1000,
            "n_batches": n_batches,
            "vram_mib": vram,
        })
        print(
            f"  batch_size={bs:>4}  {throughput:>7.2f} img/s  "
            f"{(elapsed / len(images)) * 1000:>7.1f} ms/img  "
            f"peak VRAM={vram} MiB  ({n_batches} batches)"
        )

    if not results:
        print("ERROR: no batch sizes succeeded.")
        return 1

    # Recommend the fastest batch size that fits in VRAM.
    results.sort(key=lambda r: r["throughput"], reverse=True)
    winner = results[0]

    print()
    print("=" * 64)
    print("Ranking (fastest first):")
    print(f"  {'bs':>4}  {'img/s':>8}  {'ms/img':>8}  {'VRAM MiB':>9}")
    for r in results:
        marker = "  <-- winner" if r is winner else ""
        print(
            f"  {r['batch_size']:>4}  {r['throughput']:>8.2f}  "
            f"{r['ms_per_image']:>8.1f}  {r['vram_mib'] or '-':>9}{marker}"
        )
    print()
    print(
        f"Recommended INDEXER_BATCH_SIZE = {winner['batch_size']}  "
        f"({winner['throughput']:.1f} img/s, {winner['vram_mib']} MiB VRAM)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
