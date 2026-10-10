class CaptureError(Exception):
    """Expected station-level capture failure."""

    def __init__(self, message: str, artifacts: dict[str, bytes] | None = None, http: dict | None = None):
        super().__init__(message)
        self.artifacts = artifacts or {}
        self.http = http or {}


class SensitiveHtmlError(CaptureError):
    """HTML appears to contain a reusable session secret; it is not persisted."""


class UnsafeZipError(CaptureError):
    """ZIP structure is unsafe or ambiguous."""
