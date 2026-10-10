import io
import zipfile

import pytest

from siar_forecast_raw.errors import UnsafeZipError
from siar_forecast_raw.zip_safe import extract_single_csv


def make_zip(name="forecast.csv", body=b"Fecha;ET0\n10/10/2026;2,30\n"):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, body)
    return stream.getvalue()


def test_safe_csv_read_stays_in_memory():
    assert extract_single_csv(make_zip()) == ("forecast.csv", b"Fecha;ET0\n10/10/2026;2,30\n")


@pytest.mark.parametrize("name", ["../escape.csv", "/tmp/escape.csv", "C:/escape.csv", "nested/../../escape.csv"])
def test_rejects_path_traversal(name):
    with pytest.raises(UnsafeZipError):
        extract_single_csv(make_zip(name))


def test_rejects_multiple_files_and_non_csv():
    with pytest.raises(UnsafeZipError):
        extract_single_csv(make_zip("forecast.txt"))
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("a.csv", "a")
        archive.writestr("b.csv", "b")
    with pytest.raises(UnsafeZipError):
        extract_single_csv(stream.getvalue())
