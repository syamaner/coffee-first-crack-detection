"""Filesystem preflight for fresh, non-destructive data preparation outputs."""

from pathlib import Path


def require_absent_output_dir(output_dir: Path) -> None:
    """Require an absent output entry and an existing directory parent.

    Args:
        output_dir: Proposed new output directory.

    Raises:
        FileExistsError: If any entry already occupies the output path.
        FileNotFoundError: If the parent is missing.
        NotADirectoryError: If the parent is not a directory.
        OSError: If absence cannot be established by lstat.
    """
    try:
        output_dir.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(f"Output already exists: {output_dir}")
    if not output_dir.parent.is_dir():
        if not output_dir.parent.exists():
            raise FileNotFoundError(f"Output parent must already exist: {output_dir.parent}")
        raise NotADirectoryError(output_dir.parent)
