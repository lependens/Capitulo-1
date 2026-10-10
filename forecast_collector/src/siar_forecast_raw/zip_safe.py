from pathlib import PurePosixPath
import zipfile

from .errors import UnsafeZipError


def extract_single_csv(zip_bytes: bytes, max_uncompressed_bytes: int = 20_000_000) -> tuple[str, bytes]:
    """Return the sole CSV member without extracting paths to the filesystem."""
    import io
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except (zipfile.BadZipFile, OSError) as exc:
        raise UnsafeZipError(f"Invalid ZIP: {exc}") from exc
    with archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1:
            raise UnsafeZipError("Expected exactly one non-directory ZIP member")
        member = members[0]
        path = PurePosixPath(member.filename.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or (path.parts and ":" in path.parts[0]):
            raise UnsafeZipError("ZIP member path is unsafe")
        if path.suffix.lower() != ".csv" or member.file_size > max_uncompressed_bytes:
            raise UnsafeZipError("ZIP must contain one bounded CSV file")
        try:
            chunks = bytearray()
            with archive.open(member, "r") as source:
                while len(chunks) <= max_uncompressed_bytes:
                    chunk = source.read(min(64 * 1024, max_uncompressed_bytes + 1 - len(chunks)))
                    if not chunk:
                        break
                    chunks.extend(chunk)
            data = bytes(chunks)
        except (zipfile.BadZipFile, RuntimeError) as exc:
            raise UnsafeZipError(f"Could not read ZIP member: {exc}") from exc
        if len(data) > max_uncompressed_bytes:
            raise UnsafeZipError("ZIP member expands beyond the configured size limit")
        if len(data) != member.file_size:
            raise UnsafeZipError("ZIP member size does not match its directory entry")
        return member.filename, data
