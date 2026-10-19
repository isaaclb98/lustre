#!/usr/bin/env python3
"""
Read-only polling of the prod lustre Qdrant collection.

Polls /collections/images on the prod Qdrant (host port 6333) every
INTERVAL seconds and appends (timestamp_iso, points_count) to a CSV.
Stops when points_count is unchanged across two consecutive samples
(meaning the indexing run has ended) or when SIGTERM is received.

READ-ONLY. No writes to Qdrant. No API calls against the search app.
Just GET /collections/images.
"""
from __future__ import annotations

import csv
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

QDRANT_URL = "http://127.0.0.1:6333/collections/images"
INTERVAL = 5.0         # seconds between samples
QUIET_RUNS = 12        # ~60s of no movement before declaring done
OUT_PATH = Path("/tmp/prod_index_progress.csv")

stopping = False


def _on_sigterm(signum, frame):
    global stopping
    stopping = True


signal.signal(signal.SIGTERM, _on_sigterm)
signal.signal(signal.SIGINT, _on_sigterm)


def fetch_points() -> int | None:
    """Read prod collection point count. Returns None on read failure."""
    try:
        with urlopen(QDRANT_URL, timeout=5) as r:
            data = json.load(r)
        return int(data["result"]["points_count"])
    except (URLError, KeyError, ValueError, json.JSONDecodeError):
        return None


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not OUT_PATH.exists()
    with OUT_PATH.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp_iso", "epoch_s", "points_count", "status"])
        last_pts: int | None = None
        quiet = 0
        # Take an initial sample immediately so the first delta isn't lost.
        pts = fetch_points()
        ts = datetime.now(timezone.utc).isoformat()
        epoch = time.time()
        if pts is None:
            w.writerow([ts, f"{epoch:.1f}", "", "read_error"])
            f.flush()
            print(f"{ts} read_error")
        else:
            w.writerow([ts, f"{epoch:.1f}", pts, "ok"])
            f.flush()
            print(f"{ts} points={pts}")
            last_pts = pts
        while not stopping:
            time.sleep(INTERVAL)
            pts = fetch_points()
            ts = datetime.now(timezone.utc).isoformat()
            epoch = time.time()
            if pts is None:
                w.writerow([ts, f"{epoch:.1f}", "", "read_error"])
                f.flush()
                print(f"{ts} read_error")
                # Don't reset the quiet counter on transient errors;
                # if a real read keeps failing, the script will time out
                # at the supervisor level instead.
                continue
            if pts == last_pts:
                quiet += 1
            else:
                quiet = 0
                last_pts = pts
            w.writerow([ts, f"{epoch:.1f}", pts, "ok"])
            f.flush()
            print(f"{ts} points={pts} quiet={quiet}")
            if quiet >= QUIET_RUNS:
                print(f"{ts} done: {QUIET_RUNS} consecutive unchanged samples")
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
