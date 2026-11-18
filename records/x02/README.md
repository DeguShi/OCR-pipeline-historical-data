# X02 — PA with the search budget of E17's baseline A (181 configurations)

Written before the run (2026-11-18).

**Question.** In E20 the PA of every recognizer came from 92 configurations (id 0 + 60 random, center + 30 local).
E17's baseline A, on R1, used 181 (120 + 60) and reached tuning WER 20.09 against 20.54 for E20's R1·PA. Does
R1·P6's lead over R4·PA and R5·PA (test10, Gemini reference: −2.34 and −2.52 WER points) survive a PA search of the
E17 size, and how large is the decoder share with the recognizer fixed (R4·P6 vs R4·PA: −1.24 in E20)?

**Decision it changes.** How the ICDAR version states the decoder comparison: "under the recorded budgets" (E20) or
"with the PA search at least as large as E17's", and which number the decoder share gets.

**Design.** `tools/run_x02.sh` (select / eval / report), same tools, seeds (0 for the search, 42 for the bootstrap),
character map and scoring as E20; only `--n` changes (A 120, A2 60). Recognizers R1, R4, R5. Selection on the 9
tuning pages only (official GT); frozen (`select/frozen_x02.json`) before any held-out or test page is decoded. The
held-out and test10 posteriors are the E20 exports (Colab T4, commit 82607a6), reused, not recomputed. P6 is not
re-tuned: it keeps its E20 budget (16 variants), so after X02 the PA gets more tuning configurations than P6 on
R4/R5.

**Success criterion.** (1) Reproduction: every configuration shared with E20's 92 gives the same tuning WER
(`REPRO_OK`). (2) Frozen before evaluation. (3) Reported whatever the outcome: dWER with the word-block bootstrap CI
and the page-level sign-flip p and page-bootstrap CI, for the pairs in `tools/x02_report.py`.

**Cost estimate.** CPU only, free Colab account (0 units). E20 logged ~15 s per configuration with 2 decode
workers; 3 recognizers × 181 configurations ≈ 2.3 h on 2 vCPU at that rate, plus ~10 min setup and ~5 min eval.
Smoke test first (A 2 + A2 1 on R4) on the same VM type.

**Inputs not in git.** `OUT/tune/post/{r1_st_c1m,r4_traprinq,r5_mccatmus}` and `OUT/eval/post/{heldout,test10}/...`
from `outputs/e20_colab_e20ev_ev/outputs/e20/` (sha256 of the tarball recorded at launch); KenLM files from the HF
model repo (`lm/`, sha256 checked in the job).

### 2026-11-18 — smoke test failed (no X02 result)

Free Colab CPU VM, code 03dc4a4, mode smoke (R4, A 2 + A2 1). Setup 225 s (venvs, HF weights, KenLM sha256 OK);
`PART select` exited 1 after 58.7 s, peak RSS 529 MiB: `tools/charmap.py` could not read its default
`--labels records/training/st_c1m/labels_r1/lines.jsonl`. Cause: the shipped `code.tgz` was a trimmed
`git archive` (1,488 of 2,486 files, to keep the upload small) and the trim left out `records/training/`. The two
decoded configurations were never scored, so neither the reproduction check nor the freeze ran. The full search
was not launched. The session was then preserved and stopped (archive sha256 c1069449…, kept off git: local
`hackathon_ocr_historico/outputs/colab_preservation_20261118/x02_remote.tar.gz`, copy on Google Drive
`OCR-backup-20261118/`); that archive also holds `post.tgz`, the E20 posteriors this run reuses.

Fix before relaunch: ship the untrimmed `git archive HEAD` (≈207 MB, upload in parts) or keep `records/training/`
in the trim, and assert on the VM that `records/training/st_c1m/labels_r1/lines.jsonl` exists before `PART select`.
Then repeat the smoke test.
