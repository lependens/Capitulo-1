from pathlib import PurePosixPath
import zipfile
import zlib

from .errors import UnsafeZipError

MAX_ARCHIVE_BYTES = 20_000_000
MAX_UNCOMPRESSED_BYTES = 20_000_000


def extract_single_csv(
    zip_bytes: bytes,
    max_uncompressed_bytes: int = MAX_UNCOMPRESSED_BYTES,
    max_archive_bytes: int = MAX_ARCHIVE_BYTES,
) -> tuple[str, bytes]:
    """Return the sole CSV member without extracting paths to the filesystem."""
    import io
    if len(zip_bytes) > max_archive_bytes:
        raise UnsafeZipError("ZIP exceeds the configured archive size limit")
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except (zipfile.BadZipFile, OSError, zlib.error) as exc:
        raise UnsafeZipError(f"Invalid ZIP: {exc}") from exc
    with archive:
        entries = archive.infolist()
        for entry in entries:
            path = PurePosixPath(entry.filename.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts or (path.parts and ":" in path.parts[0]):
                raise UnsafeZipError("ZIP member path is unsafe")
        members = [item for item in entries if not item.is_dir()]
        if len(members) != 1:
            raise UnsafeZipError("Expected exactly one non-directory ZIP member")
        member = members[0]
        path = PurePosixPath(member.filename.replace("\\", "/"))
        if path.suffix.lower() != ".csv" or member.file_size > max_uncompressed_bytes:
            raise UnsafeZipError("ZIP must contain one bounded CSV file")
        try:
            if archive.testzip() is not None:
                raise UnsafeZipError("ZIP member failed CRC validation")
            chunks = bytearray()
            with archive.open(member, "r") as source:
                while len(chunks) <= max_uncompressed_bytes:
                    chunk = source.read(min(64 * 1024, max_uncompressed_bytes + 1 - len(chunks)))
                    if not chunk:
                        break
                    chunks.extend(chunk)
            data = bytes(chunks)
        except (zipfile.BadZipFile, RuntimeError, zlib.error, EOFError) as exc:
            raise UnsafeZipError(f"Could not read ZIP member: {exc}") from exc
        if len(data) > max_uncompressed_bytes:
            raise UnsafeZipError("ZIP member expands beyond the configured size limit")
        if len(data) != member.file_size:
            raise UnsafeZipError("ZIP member size does not match its directory entry")
        return member.filename, data
