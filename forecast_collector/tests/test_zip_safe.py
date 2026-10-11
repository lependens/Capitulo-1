import io
import csv
import struct
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


def test_accepts_observed_csv_encoding_delimiter_headers_and_rows():
    rows = [
        "11/10/2026;0,80;2,30;1,84;0,00;1,84",
        "12/10/2026;0,81;2,40;1,94;0,00;1,94",
        "13/10/2026;0,82;2,50;2,05;0,00;2,05",
        "14/10/2026;0,83;2,60;2,16;0,00;2,16",
        "15/10/2026;0,84;2,70;2,27;0,00;2,27",
        "16/10/2026;0,85;2,80;2,38;0,00;2,38",
        "17/10/2026;0,86;2,90;2,49;0,00;2,49",
    ]
    body = b"\xef\xbb\xbf" + ("Fecha;Kc;ET0;ETc;Pe;ETc - Pe\n" + "\n".join(rows) + "\n").encode("utf-8")
    filename, extracted = extract_single_csv(make_zip("Pronostico_IB04_Son_Ferriol.csv", body))

    assert filename == "Pronostico_IB04_Son_Ferriol.csv"
    assert extracted == body
    parsed = list(csv.reader(extracted.decode("utf-8-sig").splitlines(), delimiter=";"))
    assert parsed[0] == ["Fecha", "Kc", "ET0", "ETc", "Pe", "ETc - Pe"]
    assert len(parsed[1:]) == 7
    assert all(len(row) == 6 for row in parsed[1:])


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


def test_rejects_unsafe_directory_entries():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("../", b"")
        archive.writestr("forecast.csv", "Fecha;ET0\n")
    with pytest.raises(UnsafeZipError, match="path is unsafe"):
        extract_single_csv(stream.getvalue())


def test_rejects_archive_and_uncompressed_size_limits():
    payload = make_zip(body=b"x" * 128)
    with pytest.raises(UnsafeZipError, match="archive size limit"):
        extract_single_csv(payload, max_archive_bytes=len(payload) - 1)
    with pytest.raises(UnsafeZipError, match="bounded CSV"):
        extract_single_csv(payload, max_uncompressed_bytes=32)


def test_rejects_crc_corruption():
    corrupted = bytearray(make_zip(body=b"Fecha;ET0\n10/10/2026;2,30\n"))
    central_directory = corrupted.find(b"PK\x01\x02")
    assert central_directory >= 0
    crc_offset = central_directory + 16
    original_crc = struct.unpack_from("<I", corrupted, crc_offset)[0]
    struct.pack_into("<I", corrupted, crc_offset, original_crc ^ 1)

    with pytest.raises(UnsafeZipError):
        extract_single_csv(bytes(corrupted))
