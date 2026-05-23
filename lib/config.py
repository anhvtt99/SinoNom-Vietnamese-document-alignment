"""
Project configuration and environment variable helpers.

Priority:
1. Real environment variables
2. .env file
3. Provided default values
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


_ENV_LOADED = False


def load_project_env(env_path: Optional[str] = None, override: bool = False) -> None:
    """
    Load environment variables from a .env file.

    Args:
        env_path:
            Optional path to a .env file. If None, python-dotenv searches
            from the current working directory upward.
        override:
            If False, real environment variables are not overwritten.
    """
    global _ENV_LOADED

    if _ENV_LOADED:
        return

    if env_path:
        load_dotenv(dotenv_path=Path(env_path), override=override)
    else:
        load_dotenv(override=override)

    _ENV_LOADED = True


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get an environment variable with an optional default.
    """
    return os.getenv(name, default)


def require_env(name: str) -> str:
    """
    Get a required environment variable.

    Raises:
        ValueError if the variable is missing or empty.
    """
    value = os.getenv(name)

    if value is None or value.strip() == "":
        raise ValueError(
            f"Missing required environment variable: {name}. "
            f"Set it in your shell or .env file."
        )

    return value