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

### 2026-11-19 — smoke test passed; full run launched

Free Colab CPU VM (2 vCPU, 0 units), code 441c646 (untrimmed `git archive HEAD`, sha256 deca09c9…; post.tgz
sha256 e5b80ffc…, same as the preserved copy). Launcher `records/colab_jobs/x02_launch.py`; VM sha256 of
`tools/run_x02.sh` and `tools/charmap.py` equal to git. Smoke (R4, A 2 + A2 1): setup 263 s, select 119 s, eval 44 s,
peak RSS 528 MiB; `REPRO r4_traprinq A shared 3 of E20 61 differ 0`, `REPRO_OK`, frozen before eval, `JOB_OK`. The
smoke's frozen configuration and its held-out/test decodes (outputs/x02_smoke) are a pipeline check only and are not
reported.

Note for reading the full run: A2 is centred on the best A configuration. With 121 A configurations that best may
differ from E20's (61), and then A2 shares no configuration with E20 (as in the smoke: `A2 shared 0`); the
reproduction check is then carried by A alone.

Full-run estimate: 543 configurations (3 recognizers × 181); at the smoke's ≈20–40 s per configuration on 2 vCPU,
about 3–6 h, peak memory < 1 GB. One shard (2 vCPU).

### 2026-11-19 — results (full run, `JOB_OK`)

Free Colab CPU VM (2 vCPU, 0 units), code 441c646, one shard. Select 4 h 35 min (16,532 s), eval 158 s, report 2 s;
peak RSS 575 MiB. Pulled with `colab_pull.py` (298 files, 894 MB, `outputs/x02_colab_full/`, not in git) before the
session was stopped; small files in `run_20261119/` (frozen configuration, all scored configurations and logs,
`report/grid_pa181.json`, `report/paired.json`, `report/paired_pages.json`, 9 `eval/score/.../result.json`, job log,
env, machine, tool hashes).

**Gates.** (1) Reproduction: `REPRO <rec> A shared 61 of E20 61 differ 0` for R1, R4 and R5, `REPRO_OK`. A2 shared 0
of E20's 31 for every recognizer: the larger A moved the best A configuration (R1 id 112, R4 id 92, R5 id 71, all
outside E20's 61), so A2 explored a different neighbourhood; the reproduction check rests on A. (2) Frozen
(`select/frozen_x02.json`) before any held-out or test page was decoded. (3) Reported below whatever the outcome.

**Tuning WER of the frozen configuration** (official GT, 9 tuning pages; E20's PA in brackets): R1 19.92 (20.54),
R4 20.94 (20.71), R5 22.29 (22.86). Because A2 is centred on A's best, the 181 configurations are not a superset of
E20's 92, and on R4 the larger search ends with a worse tuning WER than E20's. P6's tuning WER on R1 is 19.64.

**Evaluation** (WER / CER %, normalized):

| system | held-out official | held-out Gemini | test10 Gemini |
|---|---|---|---|
| R1·pa181 | 19.69 / 4.53 | 8.58 / 2.68 | 4.33 / 1.66 |
| R4·pa181 | 19.21 / 4.59 | 7.23 / 2.55 | 6.06 / 2.05 |
| R5·pa181 | 19.84 / 5.07 | 10.24 / 3.63 | 6.63 / 2.31 |
| R1·P6 (E20, for reference) | 18.74 / 4.35 | 6.33 / 2.26 | 3.49 / 1.50 |

**Paired comparisons** (dWER = first minus second; word-block bootstrap CI95; pages won/tied/lost; page-level
sign-flip p; `run_20261119/report/paired*.json`):

| pair | held-out official | held-out Gemini | test10 Gemini |
|---|---|---|---|
| R1·P6 vs R1·pa181 | −0.94 [−2.42, 0.31] 2/1/0 | −2.26 [−3.90, −0.75] 3/0/0 | −0.84 [−1.41, −0.26] 6/3/1, p 0.047 |
| R1·P6 vs R4·pa181 | −0.47 [−2.23, 1.26] 1/1/1 | −0.90 [−3.19, 1.21] 2/0/1 | −2.57 [−3.65, −1.50] 8/1/1, p 0.008 |
| R1·P6 vs R5·pa181 | −1.10 [−3.49, 1.27] 2/0/1 | −3.92 [−6.04, −1.82] 3/0/0 | −3.14 [−4.27, −2.02] 10/0/0, p 0.002 |
| R4·P6 vs R4·pa181 | −0.47 [−1.55, 0.47] | −1.81 [−3.32, −0.45] | −1.46 [−2.21, −0.75] 9/0/1, p 0.020 |
| R5·P6 vs R5·pa181 | −0.16 [−2.19, 1.90] | −3.16 [−4.95, −1.64] | −1.86 [−2.64, −1.09] 9/0/1, p 0.006 |
| R1·pa181 vs R1·pa | −0.16 [−1.55, 1.24] | −1.81 [−3.31, −0.45] | −1.11 [−1.68, −0.58] 7/3/0, p 0.016 |
| R4·pa181 vs R4·pa | +0.00 [−0.80, 0.94] | +0.60 [−0.45, 1.95] | +0.22 [−0.18, 0.62] 2/3/5, p 0.28 |
| R5·pa181 vs R5·pa | +0.47 [−1.09, 2.03] | +2.26 [0.76, 4.06] | +0.62 [0.26, 1.03] 1/2/7, p 0.031 |

With 3 held-out pages the smallest attainable sign-flip p is 0.25, so the page-level test can only speak on test10.

**Reading by the E20 rule** ("beats" only if the test CI excludes 0 and held-out is not unfavourable): R1·P6 beats
R1·pa181, R4·pa181 and R5·pa181 (test CIs exclude 0, held-out official CIs include 0, no held-out point estimate
favours PA). The answer to X02's question: R1·P6's lead over R4·PA / R5·PA on test10 does not shrink when PA gets
E17's search size (R4: −2.34 → −2.57, R5: −2.52 → −3.14), and P6 still beats PA with the recognizer fixed on test10
(R4 −1.46, R5 −1.86). Against R1·PA the lead shrinks from −1.95 to −0.84 and its CI still excludes 0. A larger PA
search helped on R1 only; on R5 the configuration with the better tuning WER is worse on test10 (+0.62, CI excludes
0), so a bigger random/local search does not by itself buy test accuracy here.
