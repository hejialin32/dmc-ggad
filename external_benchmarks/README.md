# External Benchmark Artifacts

This directory tracks external baseline work that is separate from the
DMC-GGAD method implementation. It is intended to keep benchmark scripts,
status summaries, and large-GPU follow-up plans visible without mixing them
into the method code.

Current scope:

- GADBench-style datasets: Amazon, Yelp, T-Finance, Elliptic, Reddit,
  Questions, Tolokers, and Weibo.
- Representative external methods: XGBGraph, RFGraph, BWGNN, GAD-NR, RHO,
  Flex-GAD, and later UniGAD where protocol alignment is feasible.
- DMC-GGAD itself is intentionally not run here because the method is still
  being adjusted.

Important files:

- `scripts/run_flexgad_gadbench.py`: GADBench-compatible Flex-GAD runner.
- `scripts/start_flexgad_queue.sh`: queue launcher used for the 48GB run.
- `results/flexgad_current_summary_20260710.csv`: completed Flex-GAD runs and
  OOM statuses on the 48GB server.
- `results/oom_status_20260710.csv`: datasets that require an 80GB GPU under
  the current Flex-GAD configuration.
- `results/tam_legacy_summary_20260710.csv`: TAM legacy baseline completion
  summary, including the NaN-guard rerun of Citeseer.
- `plans/representative_80gb_benchmark_plan.md`: reduced benchmark plan for an
  80GB GPU machine.

Raw long-form CSVs and logs remain on the experiment server and should be
copied into this directory after SSH access is stable.
