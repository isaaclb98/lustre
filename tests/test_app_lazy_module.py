"""The module-level app must not be built at import time.

`search/app.py` used to run `app = create_app()` as a module-level
statement. Importing the module therefore built a whole app: it read
the config, opened the photo cache, and — through the startup
reconciler — CONTACTED QDRANT at whatever QDRANT_URL the environment
held.

That last part is the hazard. The default URL is localhost:6333, which
on any host that also runs the production stack is a REAL production
Qdrant. So an incidental import — the demo server, the OpenAPI
dumper, a test module — inspected the production collection and, on a
dim mismatch, refused to start. The reconciler no longer drops
collections, but it still refuses, and it still reads production.

The app is now built lazily, on first attribute access. Production
servers load it by name (`gunicorn search.app:app`), which resolves
through a getattr, so the lazy build still serves them.

These tests pin both halves of that: importing must not touch
Qdrant, and the attribute must still resolve to a working app.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap


def _run_python(body: str, env_overrides: dict[str, str]) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(env_overrides)
    env["PYTHONPATH"] = os.getcwd()
    # The config validates that PHOTOS_DIR exists, so every run gets
    # a real (empty) directory. A bogus one would fail the config
    # load rather than the behaviour under test.
    env.setdefault("PHOTOS_DIR", tempfile.mkdtemp(prefix="lustre-lazy-app-"))
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        env=env,
        cwd=os.getcwd(),
        timeout=120,
        check=False,
    )


def test_importing_search_app_does_not_build_an_app() -> None:
    """`import search.app` must not construct the app (and so must not
    contact Qdrant or read the config)."""
    proc = _run_python(
        """
        import search.app as m

        # The module imported cleanly. If the app had been built at
        # import time, `app` would already be in the module's
        # __dict__; with the lazy build it is not there until someone
        # asks for it.
        assert "app" not in vars(m), "the app was built at import time"
        print("OK")
        """,
        {
            # The reconciler is pointed at a port nothing listens on,
            # and PHOTOS_DIR at a path that does not exist. If the
            # import still loaded the config, the run would fail on
            # the missing directory — so the config was NOT loaded,
            # which is the point of the test. (The reconciler, for
            # its part, only logs a connection failure; the missing
            # directory is what turns "reached the config" into a
            # hard failure.)
            "QDRANT_URL": "http://127.0.0.1:1",
            "PHOTOS_DIR": "/nonexistent-photos-dir",
            "INDEX_DB_PATH": ":memory:",
            "SEARCH_NO_MODEL": "1",
            "SEARCH_TEST_MODE": "1",
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_the_app_attribute_still_resolves() -> None:
    """`search.app:app` — the form gunicorn and uvicorn load — must
    still resolve to a working FastAPI app."""
    proc = _run_python(
        """
        import tempfile

        import search.app as m

        app = getattr(m, "app")
        assert type(app).__name__ == "FastAPI", type(app).__name__
        # And it is cached: a second access returns the same app, so
        # the module does not build one per request.
        assert getattr(m, "app") is app
        print("OK")
        """,
        {
            "QDRANT_URL": "http://127.0.0.1:1",
            "INDEX_DB_PATH": ":memory:",
            "SEARCH_NO_MODEL": "1",
            "SEARCH_TEST_MODE": "1",
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_gunicorn_can_load_the_app_by_name() -> None:
    """The production entry point must keep working."""
    proc = _run_python(
        """
        from gunicorn.util import import_app

        app = import_app("search.app:app")
        assert type(app).__name__ == "FastAPI", type(app).__name__
        print("OK")
        """,
        {
            "QDRANT_URL": "http://127.0.0.1:1",
            "INDEX_DB_PATH": ":memory:",
            "SEARCH_NO_MODEL": "1",
            "SEARCH_TEST_MODE": "1",
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_an_unknown_attribute_still_raises() -> None:
    """The module __getattr__ must not swallow typos."""
    proc = _run_python(
        """
        import search.app as m

        try:
            m.definitely_not_a_real_attribute
        except AttributeError:
            print("OK")
        else:
            raise AssertionError("expected an AttributeError")
        """,
        {
            "QDRANT_URL": "http://127.0.0.1:1",
            "PHOTOS_DIR": "/nonexistent-photos-dir",
            "INDEX_DB_PATH": ":memory:",
            "SEARCH_NO_MODEL": "1",
            "SEARCH_TEST_MODE": "1",
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
