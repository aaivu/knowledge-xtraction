import os
import importlib
from pathlib import Path
from typing import Optional

find_dotenv = None
load_dotenv = None
try:
    dotenv_module = importlib.import_module("dotenv")
    find_dotenv = getattr(dotenv_module, "find_dotenv", None)
    load_dotenv = getattr(dotenv_module, "load_dotenv", None)
except ImportError:  # pragma: no cover
    pass


_ENV_LOADED_FLAG = "KGANALYTICA_ENV_LOADED"


def load_environment() -> Optional[str]:
    """Load .env variables once, preferring the current working directory.

    Returns the resolved .env path when found, otherwise ``None``.
    """
    if os.environ.get(_ENV_LOADED_FLAG) == "1":
        return None

    candidate_paths = []

    # 1) Most common expectation: .env in the directory where the user runs Python.
    if find_dotenv is not None:
        cwd_dotenv = find_dotenv(usecwd=True)
        if cwd_dotenv:
            candidate_paths.append(Path(cwd_dotenv).resolve())

    # 2) Fallback: search upward from this file (editable installs / source checkouts).
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        env_file = parent / ".env"
        if env_file.exists():
            candidate_paths.append(env_file.resolve())

    # De-duplicate while preserving order.
    seen = set()
    for env_path in candidate_paths:
        key = str(env_path)
        if key in seen:
            continue
        seen.add(key)
        if load_dotenv is not None:
            loaded = load_dotenv(dotenv_path=env_path, override=False)
        else:
            loaded = _fallback_load_dotenv(env_path)

        if loaded:
            os.environ[_ENV_LOADED_FLAG] = "1"
            return key

    # Still mark as attempted to avoid repeated filesystem scans.
    if load_dotenv is not None:
        load_dotenv(override=False)
    os.environ[_ENV_LOADED_FLAG] = "1"
    return None


def _fallback_load_dotenv(path: Path) -> bool:
    """Minimal .env loader when python-dotenv is unavailable."""
    loaded_any = False
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
                loaded_any = True
    except OSError:
        return False
    return loaded_any
