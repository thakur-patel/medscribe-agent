# Golden cases

One case per JSONL line; lines starting with `#` are comments.

```json
{"case_id": "dev-001", "split": "dev", "review_status": "draft", "tags": ["routine"],
 "transcript": [{"id": "u01", "speaker": "doctor", "text": "..."}],
 "expected_prescription": { ... Prescription JSON ... }}
```

In `expected_prescription`, any field left out means "not stated": the value is null with flag `missing`. So a medication with no `dose` key expects the model to return the dose as null and flagged, not filled.

- `dev/`: development cases. The 10 seed cases are synthetic and marked **DRAFT: needs clinical review by the team's healthcare member**. Set `review_status` to `reviewed` once checked.
- `heldout/`: written by people only, never read during development. See its README.

Seed case mix: 4 routine visits (dev-001 to dev-004), dose or frequency never stated (dev-005, dev-006), a sound-alike drug name (dev-007), an allergy mentioned by the patient (dev-008), a patient asking for a drug the doctor does not prescribe (dev-009), a patient guessing a diagnosis the doctor does not confirm (dev-010).
