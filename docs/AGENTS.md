# Project Agent Rules

This project uses the global Superpowers skill bundle installed at:

```text
~/.ai-superpowers/all.md
```

Before making non-trivial changes, read and follow the relevant Superpowers workflow. For this project, the most relevant skills are:

1. `brainstorming` for unclear requirements.
2. `writing-plans` before multi-step implementation.
3. `test-driven-development` for behavioral changes.
4. `systematic-debugging` for failures.
5. `verification-before-completion` before reporting success.

## Project Goal

Detect whether both hands are making an OK gesture before allowing GLASSES-side data capture.

The core runtime decision is:

```text
left hand is OK
and right hand is OK
and both hands are inside the camera FOV center
and glasses pose is acceptable
=> ready to capture
```

## First-Principles Working Mode

Do not assume the requested implementation path is the shortest path.

When intent is unclear, stop and clarify the goal. When the goal is clear but the path is wasteful, explain the shorter path and proceed with it.

Every technical decision should answer:

```text
Why is this necessary for the capture gate?
```

## Data And Model Rules

Current model path:

```text
models/ok_hand_numpy_logreg.pkl
```

Current training data path:

```text
data/processed/hagrid_ok_features.csv
```

The project trains on HaGRID landmark annotations, not raw images. Raw images are only needed if the project switches to an image detector/classifier.

Use this pipeline:

```text
HaGRID JSON annotations
-> hand_landmarks
-> feature CSV
-> single-hand OK classifier
-> runtime two-hand gate
```

## Commands

Run tests:

```bash
scripts/test.sh
```

Prepare HaGRID landmarks:

```bash
PYTHONPATH=src python -m double_ok_gesture.prepare_hagrid \
  --annotations-dir data/raw/hagrid/annotations \
  --output data/processed/hagrid_ok_features.csv
```

Train NumPy fallback model:

```bash
PYTHONPATH=src python -m double_ok_gesture.train \
  --input data/processed/hagrid_ok_features.csv \
  --output models/ok_hand_numpy_logreg.pkl \
  --model numpy_logreg
```

Run demo with capture gate:

```bash
PYTHONPATH=src python -m double_ok_gesture.demo \
  --camera 0 \
  --model models/ok_hand_numpy_logreg.pkl \
  --capture-gate
```

## Verification Standard

Do not report completion until the relevant command has been run and its result is stated.

For code changes, minimum verification is:

```bash
PYTHONPATH=src pytest -q
```

For data/model changes, also report:

1. CSV row count.
2. Positive and negative sample counts.
3. Model output path.
4. Evaluation metrics.
