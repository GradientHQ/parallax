from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    # Search for the project root by looking for pyproject.toml in parent directories
    current_dir = Path(__file__).parent
    while current_dir != current_dir.parent:
        if (current_dir / "pyproject.toml").exists():
            return current_dir
        current_dir = current_dir.parent

    # If not found, fallback to current working directory
    return Path.cwd()
