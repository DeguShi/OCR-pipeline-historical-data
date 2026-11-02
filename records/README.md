# Development records

Everything the searches and training runs produced. Kept so that whoever continues the work starts from what was
measured instead of starting over. Recognizer weights are not stored here: they are on the Hugging Face Hub and are
downloaded by `setup_pipeline.py`.

## Contents

| folder                                                     | what it is                                                                                                                   |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `searches/v1_medium`                                       | greedy conditional search of v1 (PP-OCRv6 medium, CPU): 330 evaluations, 3 passes                                            |
| `searches/v3_medium_3passes`, `searches/v3_medium_4passes` | v3 search: converged on pass 3 (465 evaluations); pass 4 confirmed it (518)                                                  |
| `searches/v4_medium`                                       | v4 search, still with the medium recognizer: 767 evaluations                                                                 |
| `searches/v4_st_v3`                                        | v4 search with the self-trained recognizer st_v3 (GPU L4): 306 evaluations                                                   |
| `searches/g1_interrupted`                                  | search with st_g1, interrupted (72 evaluations)                                                                              |
| `searches/v5_st_g2`                                        | v5 search with st_g2 (GPU L4): 313 evaluations                                                                               |
| `searches/random`                                          | random searches used in the random-vs-greedy comparison (two processes per version)                                          |
| `searches/v6_sweep_st_c1m`                                 | final sweep with the clean-chain recognizer: 20 isolated changes, none accepted                                              |
| `searches/e01_e06`                                         | sweeps of experiments E01–E06, before the greedy search existed                                                              |
| `searches/warm_start`                                      | starting choices used to resume searches (`--init`)                                                                          |
| `searches/logs`                                            | logs of the searches and final evaluations                                                                                   |
| `final_scores`                                             | frozen evaluation of each version (v3–v6) against both references                                                            |
| `training/st_c1m`                                          | the final recognizer's run: training log, metrics, line selection and labels                                                 |
| `training/other_runs`                                      | logs and metrics of every other training and evaluation run (ft_v1, st_v1–v4, st_g1–g3, st_c_s…), no weights                 |
| `corpus/sources.tgz`                                       | raw period corpus: Project Gutenberg books, São Paulo laws (ALESP, 1889–1910) and the training pages' text                   |
| `analysis`                                                 | small result files: reference comparison, ablations, v5 candidates                                                           |
| `misc`                                                     | loose logs, outputs of each version's configuration (`runs/`), learned edit costs (E11), line-ensemble test                  |
| `reference_review`                                         | first check of the official transcription against the images                                                                 |
| `colab_jobs`                                               | the exact job scripts that ran on Colab (paths updated to this layout; some build steps referred to the original repository) |

Each search folder holds:

- `evaluations.jsonl`: one line per evaluated configuration, with the full configuration and per-page WER/CER;
- `decisions.json`: the search path (each pass, each stage, what entered and what left) and the final choice;
- `best_config.json`: the best configuration found;
- `run_config.json`: the arguments and specification used (paths as they were at the time).

Search specifications are in `sweeps/`, and the frozen configuration of each version is in `configs/`. All search
measurements use only the 9 tuning pages; the held-out and test pages do not appear in any search file.

## Resuming from what was done

To run the final pipeline, use `configs/final_v6.json`. To continue searching, the search tool has two options:

- `--init` starts from the final choices of an earlier search (a `decisions.json`);
- `--reuse` reads earlier `evaluations.jsonl` files and does not recompute any configuration already scored on the same pages.

```bash
python -m ocr_pipeline.tuning.search sweeps/search_g2.json --pages tuning \
    --init records/searches/v5_st_g2/decisions.json \
    --reuse records/searches/*/evaluations.jsonl
```

Reuse applies only to identical configurations with the same recognizer. The only evaluations with the final
recognizer (st_c1m) are the 20 of the v6 sweep (`searches/v6_sweep_st_c1m/sweep.json`, in another format). A new
search with st_c1m therefore recomputes its configurations, but it can start from the v5 choices. The segmentation and
recognizer-output cache was not kept; it is rebuilt on the first run.

## Rebuilding the dictionary and the context model

Extract `corpus/sources.tgz` into `models/lexicon/` and run `tools/build_period_lexicon.py`. The `*.manifest.json`
files in `corpus/` record how each file was built, and the `build_*.log` files hold the original runs.
