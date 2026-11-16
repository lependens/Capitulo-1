"""Contractual checkout bytes, tested with synthetic Git objects only."""
import hashlib
from pathlib import Path
import subprocess
import unittest

from tests_support import temporary_directory


class CheckoutBytesTests(unittest.TestCase):
    def test_contractual_bytes_survive_autocrlf_checkout(self):
        attributes = Path(__file__).resolve().parents[2] / ".gitattributes"
        fixtures = {
            "estaciones_baleares.csv": b"station,latitude\nSYNTH,39.5\n",
            "datos_siar_baleares/IB01_datos_completos.csv": b"date,value\n2020-01-01,1\n",
            "data_quality/siar/post/SIAR-QA-POST-20261018T233901Z/report.csv": b"key,value\r\nsynthetic,1\r\n",
        }
        with temporary_directory() as tmp:
            root = Path(tmp)
            repo, checkout = root / "repo", root / "checkout"
            repo.mkdir()
            checkout.mkdir()

            def git(*args):
                return subprocess.run(
                    ["git", "-C", str(repo), "-c", "core.autocrlf=true", *args],
                    check=True, capture_output=True,
                ).stdout

            git("init")
            (repo / ".gitattributes").write_bytes(attributes.read_bytes())
            for name, content in fixtures.items():
                target = repo / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            git("add", ".")
            # checkout-index applies Git's checkout filters without changing sources.
            git("checkout-index", "--all", "--prefix=" + checkout.as_posix() + "/")
            for name, expected in fixtures.items():
                with self.subTest(path=name):
                    blob = git("show", ":" + name)
                    actual = (checkout / name).read_bytes()
                    self.assertEqual(blob, expected)
                    self.assertEqual(actual, expected)
                    self.assertEqual(hashlib.sha256(actual).digest(), hashlib.sha256(blob).digest())
