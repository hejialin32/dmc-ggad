# Representative 80GB Benchmark Plan

Goal: build a credible external benchmark without spending 3-7 days
reimplementing methods that do not have confirmed official code.

## Main Datasets

Use six representative datasets in the main table:

- Amazon: e-commerce/review fraud.
- Yelp: business review fraud.
- T-Finance: financial fraud and large-scale stress test.
- Elliptic: blockchain transaction anomaly detection.
- Questions: Q&A/social graph setting.
- Weibo or Tolokers: social/crowdsourcing anomaly setting.

Reddit can be moved to a supplementary table because its separation is weaker
in the current runs.

## Main Methods

Use these representative baselines:

- XGBGraph: strong GADBench tree model with neighborhood aggregation.
- RFGraph: strong tree model with neighborhood aggregation.
- BWGNN: representative spectral/wavelet GNN baseline.
- GAD-NR: useful normal-reference baseline, but not a recent-two-year method.
- RHO: recent 2025 method; complete missing datasets/seeds.
- Flex-GAD: recent 2025 method; rerun 48GB OOM datasets on 80GB GPU.

Optional after the main table:

- UniGAD: 2024 method; include only after node-level protocol alignment.
- GraphNC and CAGAD: do not include unless official code is confirmed or a
  separate reimplementation budget is approved.

## First 80GB Queue

1. Flex-GAD on `amazon,yelp,tfinance,elliptic` with the original dimension 128
   and 5 seeds first.
2. RHO on the six representative datasets, targeting 5 seeds per dataset.
3. GAD-NR only for missing representative datasets if it does not block the
   recent-method runs.

Expected wall-clock time:

- Without UniGAD: roughly 12-24 hours.
- With UniGAD protocol adaptation: roughly 1-2 days.

DMC-GGAD is intentionally excluded from this queue because the method is still
being adjusted.
