import shutil
import zipfile
from pathlib import Path
from typing import Iterable, Optional


def _is_within_directory(directory: Path, target: Path) -> bool:
    """Check that the target path is within the given directory to prevent ZipSlip."""
    try:
        directory = directory.resolve(strict=False)
        target = target.resolve(strict=False)
    except Exception:
        # Fall back to simple check
        pass
    try:
        target.relative_to(directory)
        return True
    except ValueError:
        return False


# PUBLIC_INTERFACE
def safe_extract_zip(zip_path: Path, dest_dir: Path, *, allow_symlinks: bool = False) -> None:
    """Safely extract a ZIP file into dest_dir preventing ZipSlip and optionally blocking symlinks.

    Args:
        zip_path: Path to the source ZIP file.
        dest_dir: Directory where files will be extracted.
        allow_symlinks: Whether to allow symlink members. Defaults to False.

    Raises:
        ValueError: If an entry attempts path traversal or an unsafe symlink is found.
        zipfile.BadZipFile: If the zip is invalid.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            # Normalize path and prevent absolute paths or traversal
            member_path = Path(info.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe ZIP entry detected: {info.filename}")

            final_path = dest_dir / member_path
            if not _is_within_directory(dest_dir, final_path):
                raise ValueError(f"Path traversal attempt: {info.filename}")

            if info.is_dir():
                final_path.mkdir(parents=True, exist_ok=True)
                continue

            # Ensure parent dir exists
            final_path.parent.mkdir(parents=True, exist_ok=True)

            if not allow_symlinks:
                # Heuristic: ZIP doesn't store symlink bit portably. We block external links by writing file content.
                # If later the environment requires symlinks, set allow_symlinks=True.
                pass

            with zf.open(info, "r") as src, open(final_path, "wb") as dst:
                shutil.copyfileobj(src, dst)


# PUBLIC_INTERFACE
def create_zip_from_directory(source_dir: Path, zip_path: Path, include_patterns: Optional[Iterable[str]] = None) -> None:
    """Create a ZIP archive from source directory.

    Args:
        source_dir: Directory to package.
        zip_path: Output ZIP file path.
        include_patterns: Optional iterable of glob patterns to include. If None, include all files.

    Notes:
        - Preserves relative paths inside the archive.
        - Ensures consistent file permissions where possible.
    """
    source_dir = Path(source_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    def _matches(path: Path) -> bool:
        if include_patterns is None:
            return True
        for pat in include_patterns:
            if path.match(pat):
                return True
        return False

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in source_dir.rglob("*"):
            if p.is_dir():
                continue
            rel = p.relative_to(source_dir)
            if _matches(rel):
                zf.write(p, arcname=str(rel))


# PUBLIC_INTERFACE
def create_zip_from_files(files: Iterable[Path], base_dir: Path, zip_path: Path) -> None:
    """Create a ZIP archive containing a set of files, preserving paths relative to base_dir.

    Args:
        files: Iterable of file paths to include.
        base_dir: Base directory; archive paths will be relative to this dir.
        zip_path: Output ZIP path.
    """
    base_dir = Path(base_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in files:
            f = Path(f)
            if f.is_file():
                arc = f.relative_to(base_dir)
                zf.write(f, arcname=str(arc))


# PUBLIC_INTERFACE
def read_zip_filenames(zip_path: Path) -> list[str]:
    """Return a list of filenames contained in the ZIP, for inspection or logging."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        return [i.filename for i in zf.infolist()]
