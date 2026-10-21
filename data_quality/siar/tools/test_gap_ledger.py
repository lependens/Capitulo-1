import unittest

from gap_ledger import merge_gap_ledger

COLUMNS = [
    "audit_id", "repo_git_sha", "station_code", "observation_date",
    "gap_status", "last_query_result", "first_detected_at_utc",
    "last_attempt_at_utc", "attempt_count", "recovered_at_utc",
    "siar_records_returned", "evidence_ref", "error_code", "notes",
]

def norm(value):
    return str(value)[:10]

class GapLedgerTests(unittest.TestCase):
    def setUp(self):
        self.recovered = dict(zip(COLUMNS, [
            "PRE", "repo-pre", "IB07", "2012-11-13", "RECOVERED",
            "DATA_RETURNED", "2026-10-12T07:37:40Z", "", "0",
            "2026-10-12T21:22:05.511553+00:00", "1",
            "FORENSIC-IB07-001/F2", "",
            "PRE gap detected from canonical CSV; no SIAR query attempted",
        ]))

    def test_recovered_row_is_retained_from_persisted_ledger(self):
        rows = merge_gap_ledger([], [self.recovered], COLUMNS, norm, "POST-new", "repo-new")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["audit_id"], "POST-new")
        self.assertEqual(rows[0]["repo_git_sha"], "repo-new")
        for field in COLUMNS[2:]:
            self.assertEqual(rows[0][field], self.recovered[field], field)

    def test_unresolved_rows_keep_persisted_history_and_r2_state(self):
        pending = dict(self.recovered, observation_date="2012-11-14",
                       gap_status="CONFIRMED_NO_DATA", last_query_result="ZERO_ROWS",
                       last_attempt_at_utc="2026-10-18T12:00:00Z", attempt_count="2",
                       recovered_at_utc="", siar_records_returned="0",
                       evidence_ref="trace:IB07-2012-11-14")
        audited = dict(pending, first_detected_at_utc="2026-10-20T19:00:00Z")
        rows = merge_gap_ledger([audited], [self.recovered, pending], COLUMNS, norm, "POST-new", "repo-new")
        self.assertEqual([r["observation_date"] for r in rows], ["2012-11-13", "2012-11-14"])
        self.assertEqual(rows[1]["first_detected_at_utc"], pending["first_detected_at_utc"])
        self.assertNotEqual(rows[1]["first_detected_at_utc"], audited["first_detected_at_utc"])
        self.assertEqual(rows[1]["attempt_count"], "2")
        self.assertEqual(rows[1]["gap_status"], "CONFIRMED_NO_DATA")

    def test_rejects_unexplained_missing_or_duplicate_rows(self):
        with self.assertRaises(ValueError):
            merge_gap_ledger([], [dict(self.recovered, gap_status="PENDING_RETRY")], COLUMNS, norm, "POST-new", "repo-new")
        with self.assertRaises(ValueError):
            merge_gap_ledger([], [self.recovered, self.recovered], COLUMNS, norm, "POST-new", "repo-new")
        pending = dict(self.recovered, observation_date="2012-11-14", gap_status="CONFIRMED_NO_DATA")
        with self.assertRaises(ValueError):
            merge_gap_ledger([], [pending], COLUMNS, norm, "POST-new", "repo-new")

if __name__ == "__main__":
    unittest.main()
