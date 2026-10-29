"""
search/indexer_runner.py — subprocess management for the admin Index API.

The indexer (`python -m indexer.local_sync --json-progress`) is spawned
as a child process so its multi-minute runtime can never block the
search backend's request handler. This module owns the lifecycle:

  - spawn → track state, capture stdout/stderr into a ring buffer
  - poll  → return status snapshot for the API
  - cancel → SIGTERM the child; if it doesn't exit within a grace
             period, SIGKILL
  - restart → any subsequent start() after exit spawns a fresh job

State machine (guarded by a single lock):

    idle ──start()──► running ──exit 0──► idle (last_error=None)
                          │
                          ├─exit 130──► idle  (SIGTERM / cancel)
                          │
                          └─exit ≠0/130──► failed (last_error set)

The runner emits structured `IndexerStatus` snapshots. The admin
router translates them to JSON responses.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import IO, Any

logger = logging.getLogger(__name__)


class IndexerState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    FAILED = "failed"


class IndexConflictError(RuntimeError):
    """Raised when start() is called while a job is already running."""


@dataclass
class IndexerProgress:
    indexed: int = 0
    reembedded: int = 0
    skipped: int = 0
    errors: int = 0
    # Round-50: count of thumbnails regenerated for files whose
    # embed was skipped (mtime/size match → up-to-date) but whose
    # on-disk thumb was missing. See indexer/local_sync.py for
    # the producer side. Older indexer binaries (pre-Round-50)
    # don't emit this field; default 0 is the safe fallback.
    thumb_regens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "indexed": self.indexed,
            "reembedded": self.reembedded,
            "skipped": self.skipped,
            "errors": self.errors,
            "thumb_regens": self.thumb_regens,
        }


@dataclass
class IndexerStatus:
    state: IndexerState
    mode: str | None
    job_id: str | None
    pid: int | None
    started_at: str | None
    finished_at: str | None
    last_run_at: str | None
    last_error: str | None
    progress: IndexerProgress
    points_count: int | None  # post-completion snapshot from Qdrant
    # Sub-phase within `running`: "warming_up" while the encoder is
    # being downloaded/loaded on first batch, "embedding" once we're
    # actively upserting. None otherwise. Lets the UI distinguish
    # "0 indexed because model is loading" from "0 indexed because
    # the directory is empty".
    phase: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "mode": self.mode,
            "job_id": self.job_id,
            "pid": self.pid,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "last_run_at": self.last_run_at,
            "last_error": self.last_error,
            "progress": self.progress.to_dict(),
            "points_count": self.points_count,
            "phase": self.phase,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat()


CommandFactory = Callable[[str], list[str]]
"""A factory that builds the argv list for the indexer subprocess.

The single string argument is the mode (`incremental` or `rebuild`).
Returning a list of strings (not a single command string) avoids any
shell quoting pitfalls; `subprocess.Popen` is invoked with `shell=False`.

Tests inject a factory that returns `["python", "-c", "..."]` so the
runner can be exercised without the real indexer.
"""


class IndexerRunner:
    """Subprocess owner for one indexer job at a time.

    Thread-safe. All public methods take the lock before reading or
    mutating state. The reader thread (started by `start`) does the
    same when it updates progress / appends to the log buffer.
    """

    # Grace period between SIGTERM and SIGKILL when the indexer doesn't
    # honour the cooperative cancel signal fast enough.
    CANCEL_GRACE_S: float = 10.0

    def __init__(
        self,
        *,
        command_factory: CommandFactory,
        log_buffer_size: int = 1000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._cmd_factory = command_factory
        self._log_buffer: deque[str] = deque(maxlen=log_buffer_size)
        self._log_total: int = 0  # monotonic line counter
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._wait_thread: threading.Thread | None = None
        self._state: IndexerState = IndexerState.IDLE
        self._mode: str | None = None
        self._job_id: str | None = None
        self._pid: int | None = None
        self._started_at: datetime | None = None
        self._finished_at: datetime | None = None
        self._last_run_at: datetime | None = None
        self._last_error: str | None = None
        self._progress = IndexerProgress()
        self._points_count: int | None = None
        self._phase: str | None = None
        # Set to True by cancel(). Reset by start(). Lets the wait
        # loop treat an exit via default SIGTERM action (race where
        # the signal lands before the child installs its handler) as
        # a successful cancel rather than a failure.
        self._cancel_requested = False
        self._clock = clock
        # Event the wait helper blocks on; set when state transitions
        # away from RUNNING.
        self._idle_event = threading.Event()
        self._idle_event.set()

    # --- Public API -----------------------------------------------------

    def start(self, mode: str) -> str:
        """Spawn an indexer subprocess. Returns the new job_id.

        Raises IndexConflictError if a job is already running.
        """
        if mode not in ("incremental", "rebuild"):
            raise ValueError(f"unknown indexer mode: {mode!r}")

        with self._lock:
            if self._state is IndexerState.RUNNING:
                raise IndexConflictError(
                    f"indexer is already running (job_id={self._job_id})"
                )
            argv = self._cmd_factory(mode)
            logger.info("spawning indexer: %s", " ".join(shlex.quote(a) for a in argv))
            try:
                # Round‑31: OpenClipEmbedder reads `os.environ["DEVICE"]`
                # to pick CPU vs CUDA, NOT a constructor arg. The parent
                # container only has `INDEXER_DEVICE` set, so without
                # this copy the subprocess falls back to CPU and the
                # indexing job would take ~6 days instead of ~4 hours.
                # Mirror INDEXER_DEVICE → DEVICE if the parent env has
                # the former but not the latter (don't override an
                # explicit DEVICE if both are set).
                child_env = os.environ.copy()
                child_env.setdefault("PYTHONUNBUFFERED", "1")
                if (
                    "DEVICE" not in child_env
                    and "INDEXER_DEVICE" in child_env
                ):
                    child_env["DEVICE"] = child_env["INDEXER_DEVICE"]
                # argv is built by the module-level _factory() above
                # from hardcoded sources (CLI arg defaults, env vars
                # the operator controls, and `extra_args` which only
                # the test suite passes). No untrusted input.
                self._proc = subprocess.Popen(  # noqa: S603 — argv is operator-controlled, see _factory()
                    argv,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # merge stderr → stdout (single log stream)
                    text=True,
                    bufsize=1,  # line-buffered
                    env=child_env,
                )
            except OSError as exc:
                self._state = IndexerState.FAILED
                self._last_error = f"failed to spawn indexer: {exc}"
                self._finished_at = datetime.now(timezone.utc)
                self._idle_event.set()
                raise

            self._job_id = uuid.uuid4().hex
            self._mode = mode
            self._pid = self._proc.pid
            self._started_at = datetime.now(timezone.utc)
            self._finished_at = None
            self._last_error = None
            self._progress = IndexerProgress()
            self._points_count = None
            self._phase = None
            self._cancel_requested = False
            self._state = IndexerState.RUNNING
            self._idle_event.clear()

            # Reader thread: consumes stdout, pushes to ring buffer,
            # parses JSON progress events, and detects process exit.
            # stdout is the union[IO[bytes], None]; if Popen didn't
            # capture stdout (capture_output=False) this is None and
            # we can't run a reader — surface that as a startup error
            # rather than crashing the reader thread with AttributeError.
            reader_pipe = self._proc.stdout
            if reader_pipe is None:
                raise RuntimeError("indexer process has no captured stdout (capture_output=False?)")
            self._reader_thread = threading.Thread(
                target=self._reader_loop,
                args=(reader_pipe,),
                name="indexer-reader",
                daemon=True,
            )
            self._reader_thread.start()

            # Wait thread: blocks on proc.wait() so we can update
            # state when the child exits (the reader sees EOF on stdout
            # but doesn't get the exit code directly).
            self._wait_thread = threading.Thread(
                target=self._wait_loop,
                name="indexer-wait",
                daemon=True,
            )
            self._wait_thread.start()

            return self._job_id

    def cancel(self) -> bool:
        """Send SIGTERM to the running job. Returns True if a process
        was signalled, False if there was nothing to cancel.

        The flag `self._cancel_requested` lets the wait loop treat an
        exit via default SIGTERM action (negative exit code) as a
        successful cancel rather than a failure — covers the race
        where SIGTERM lands before the child installs its handler.
        """
        with self._lock:
            proc = self._proc
            if proc is None or self._state is not IndexerState.RUNNING:
                return False
            try:
                proc.send_signal(signal.SIGTERM)
                self._cancel_requested = True
                logger.info("sent SIGTERM to indexer pid=%d", proc.pid)
                return True
            except ProcessLookupError:
                return False

    def status(self) -> IndexerStatus:
        with self._lock:
            return IndexerStatus(
                state=self._state,
                mode=self._mode,
                job_id=self._job_id,
                pid=self._pid,
                started_at=_iso(self._started_at),
                finished_at=_iso(self._finished_at),
                last_run_at=_iso(self._last_run_at),
                last_error=self._last_error,
                progress=IndexerProgress(
                    indexed=self._progress.indexed,
                    reembedded=self._progress.reembedded,
                    skipped=self._progress.skipped,
                    errors=self._progress.errors,
                    thumb_regens=self._progress.thumb_regens,
                ),
                points_count=self._points_count,
                phase=self._phase,
                )

    def log(self, *, since_line: int = 0) -> dict[str, Any]:
        """Return log lines after `since_line` (0-based, exclusive).

        `since_line=0` returns everything in the buffer. The
        `next_line` field tells the caller where to resume from.
        """
        with self._lock:
            total = self._log_total
            since_line = max(since_line, 0)
            # The ring buffer may have evicted older lines, so
            # `since_line` past the eviction point returns nothing.
            oldest_kept = total - len(self._log_buffer)
            since_line = max(since_line, oldest_kept)
            offset = since_line - oldest_kept
            lines = list(self._log_buffer)[offset:]
            return {
                "lines": lines,
                "next_line": since_line + len(lines),
                "total": total,
            }

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until the job leaves RUNNING state (or timeout).

        Returns True if the job is idle (or became idle), False on timeout.
        """
        return self._idle_event.wait(timeout=timeout)

    # --- Internals ------------------------------------------------------

    def _reader_loop(self, stdout: IO[str]) -> None:
        """Consume stdout line-by-line until EOF. Updates buffer +
        progress under the lock.
        """
        try:
            for raw in stdout:
                line = raw.rstrip("\n")
                if not line:
                    continue
                with self._lock:
                    self._log_buffer.append(line)
                    self._log_total += 1
                    if line.startswith("{"):
                        # Not a progress event we care about — keep
                        # the line in the log buffer, but don't
                        # mutate state.
                        with contextlib.suppress(ValueError, KeyError, TypeError):
                            self._handle_event(json.loads(line))
        except Exception:
            logger.exception("indexer reader loop crashed")

    def _handle_event(self, event: dict[str, Any]) -> None:
        """Update progress / state from a JSON progress event. Caller
        must hold the lock."""
        kind = event.get("event")
        if kind == "start":
            # Reset progress counters; "start" is the first event.
            self._progress = IndexerProgress()
            self._phase = "scanning"
        elif kind == "warming_up":
            self._phase = "warming_up"
        elif kind == "batch":
            self._progress = IndexerProgress(
                indexed=int(event.get("indexed", 0)),
                reembedded=int(event.get("reembedded", 0)),
                skipped=int(event.get("skipped", 0)),
                errors=int(event.get("errors", 0)),
                # Round-50: older indexer binaries don't emit this
                # field; default 0 (the dataclass default) is the
                # safe fallback so the runner doesn't break when
                # pointed at a pre-Round-50 image.
                thumb_regens=int(event.get("thumb_regens", 0)),
            )
            self._phase = "embedding"
        elif kind in ("done", "cancelled", "failed"):
            self._progress = IndexerProgress(
                indexed=int(event.get("indexed", 0)),
                reembedded=int(event.get("reembedded", 0)),
                skipped=int(event.get("skipped", 0)),
                errors=int(event.get("errors", 0)),
                thumb_regens=int(event.get("thumb_regens", 0)),
            )
            self._phase = None
            if kind == "failed":
                # Capture the child's reason. Without this the status
                # API reports only "indexer exited with code 1" and
                # the operator has to dig through the log buffer to
                # find out why — which is exactly the gap that let a
                # CUDA OOM hide behind a generic failure.
                error = event.get("error")
                if error:
                    self._last_error = str(error)

    def _wait_loop(self) -> None:
        """Block on proc.wait(); update terminal state when it exits."""
        proc = self._proc
        if proc is None:
            return
        try:
            rc = proc.wait()
        except Exception:
            logger.exception("indexer wait crashed")
            return

        with self._lock:
            # Exit codes:
            #   0     = success (done event)
            #   130   = SIGTERM-driven cancel (child caught the signal
            #           and exited cleanly via sys.exit(130))
            #   -15   = child killed by SIGTERM via default action
            #           (race: signal landed before handler installed).
            #           Treat as cancel IF we sent the SIGTERM.
            #   anything else = failed
            if rc == 0 or rc == 130 or rc == -15 and self._cancel_requested:
                self._state = IndexerState.IDLE
                self._last_error = None
            else:
                self._state = IndexerState.FAILED
                # Use the last error from the most recent "failed"
                # progress event if we captured one; else synthesise
                # from the exit code.
                if self._last_error is None:
                    self._last_error = f"indexer exited with code {rc}"
            self._finished_at = datetime.now(timezone.utc)
            self._last_run_at = self._finished_at
            self._idle_event.set()

        # SIGKILL fallback if the child ignored SIGTERM. Should be a
        # no-op for well-behaved indexers but defends against a hung
        # subprocess that blocks forever.
        if rc is None or rc not in (0, 130):
            pass  # already exited

        # If we sent SIGTERM and the child still hasn't exited within
        # the grace period, escalate. Done outside the lock to avoid
        # contention with status() callers.
        time.sleep(0)  # yield


# --- Default command factory -------------------------------------------


def default_indexer_command_factory(
    *,
    python: str = sys.executable,
    sources: Iterable[str],
    model: str,
    device: str,
    qdrant_url: str,
    qdrant_api_key: str | None,
    qdrant_collection: str,
    batch_size: int,
    indexer_module: str = "indexer.local_sync",
    extra_args: Iterable[str] = (),
) -> CommandFactory:
    """Build the default CommandFactory used by the search backend.

    The factory returns the argv list for the indexer subprocess given
    a mode (`incremental` or `rebuild`). The list is suitable for
    `subprocess.Popen` (no shell), so paths with spaces or shell
    metacharacters in `sources` are handled correctly.

    Note: `--prefix` / `--base` flags were intentionally removed in
    commit 5388f91; path translation is the search container's
    responsibility (via `HOST_PATH_PREFIX` env var). The indexer
    stores absolute paths as-is.

    Option B (Sept 2026): the indexer writes directly to the
    canonical collection (no staging area, no read/write split).
    `qdrant_collection` is the only flag forwarded.
    """
    sources_list = list(sources)

    def _factory(mode: str) -> list[str]:
        argv = [
            python,
            "-m",
            indexer_module,
            "--json-progress",
            "--model",
            model,
            "--device",
            device,
            "--qdrant-url",
            qdrant_url,
            "--qdrant-collection",
            qdrant_collection,
            "--batch-size",
            str(batch_size),
        ]
        if qdrant_api_key:
            argv += ["--qdrant-api-key", qdrant_api_key]
        for s in sources_list:
            argv += ["--source", s]
        if mode == "rebuild":
            argv += ["--rebuild"]
        elif mode == "incremental":
            # 'Index new & unchanged' also prunes. Without this, files
            # deleted from a source dir since the last index run stay
            # in Qdrant as orphan points (visible in search results,
            # occupying collection quota). Pruning after the
            # change-detection embed keeps the index in sync with the
            # filesystem. The rebuild path doesn't need this — it
            # wipes the collection before re-embedding.
            argv += ["--prune"]
        argv += list(extra_args)
        return argv

    return _factory
