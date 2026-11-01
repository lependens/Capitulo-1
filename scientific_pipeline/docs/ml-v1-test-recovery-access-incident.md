# ML v1 TEST recovery access incident

## First Kaggle recovery attempt

Run: `ml-v1-test-recovery-20261101T072045Z`

Classification: `TEST_ACCESS_OCCURRED`; `TEST_EVALUATION_NOT_PERFORMED`.

The recovery validated the frozen selection and DEVELOPMENT refit, then called `_context_for_split(..., "TEST")`. That loader reads the complete Dataset v1 before the attempt failed on the missing AHC dependency `ahc_coefficients.csv`. The durable journal did not reach `TEST_OPENED`; the manifest's `test_inspected=false` therefore understates technical access and is superseded by this incident record.

No ML TEST predictions or metrics were produced. No selection, fitting, refit, feature, candidate, or hyperparameter decisions used TEST.

Preserved evidence:

- `manifest.json` SHA256: `9bb4c7bc1f605b121e0d8797743094883a54e773b6275258f0a9fec1769e70f1`
- `recovery_journal.jsonl` SHA256: `307c065336f3be736cf19f1121222e4b499642ca98db8bcef87ef6122a1336df`
- Failure: an AHC output declared by the official AHC TEST manifest was absent from the Kaggle package.

The original campaign remains classified `TEST_ACCESS_UNKNOWN`; the server recovery remains `TEST_NOT_OPENED`. The frozen 35 model configurations remain valid for retrospective recovery evaluation because all model selection and DEVELOPMENT refits preceded this technical TEST access.

## Recovery runtime safeguards

The recovery runtime verifies every file declared under `ahc_test_manifest.json: outputs` against its pinned SHA256 before it reads Dataset v1. It durably appends `TEST_OPENED` before hashing or parsing Dataset v1, so an interruption during loading cannot leave the journal claiming that TEST was unopened. The final recovery must be labeled `retrospective_recovery_evaluation`; do not describe it as the first TEST opening, an exactly-once opening, or a pristine unseen holdout.

The recovery is evaluation-only: it consumes the frozen selection and serialized DEVELOPMENT models and performs no training, selection, or refit.
