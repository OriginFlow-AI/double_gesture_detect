# HaGRID Workflow

This project uses HaGRID/HaGRIDv2 as a single-hand gesture source.

## Why single-hand training works

The target event is "both hands are OK". Instead of searching for a dataset
that labels the whole frame as double OK, train a single-hand OK classifier and
compose two detections at runtime:

```text
hand_1 == ok and hand_2 == ok -> double_ok
```

This gives better reuse of open datasets and makes the runtime easier to debug.

## Data mapping

Positive:

```text
ok
```

Negative:

```text
no_gesture
palm
fist
stop
like
peace
one
two_up
three
rock
```

The prepared CSV contains normalized 21-point hand landmarks and geometric
features. The model does not need full images unless you later switch to an
image detector.

`--max-negative-per-class` is applied independently to each dataset split and
gesture label. This preserves negative examples in train, val, and test.

## Commands

```bash
python -m double_ok_gesture.prepare_hagrid \
  --annotations-dir /path/to/hagrid_annotations \
  --output data/processed/hagrid_ok_features.csv

python -m double_ok_gesture.train \
  --input data/processed/hagrid_ok_features.csv \
  --output models/ok_hand_mlp.joblib \
  --model mlp

python -m double_ok_gesture.evaluate \
  --input data/processed/hagrid_ok_features.csv \
  --model models/ok_hand_mlp.joblib \
  --split auto
```

`--split auto` prefers the independent test split, then val. Use `--split all`
only when intentionally inspecting the complete dataset.

## Local raw data

HaGRID reference images from the upstream `images/` directory are stored in:

```text
data/raw/hagrid/images/
```

The useful training input for this project is the official landmark annotation
archive, not the full image archives:

```text
data/raw/hagrid/annotations_zip/annotations.zip
data/raw/hagrid/annotations/
```

The full per-gesture image archives are tens of GB each. Download them only if
you later switch from landmark features to an image detector/classifier.

## Local validation

Capture a small local validation set after training:

```bash
python -m double_ok_gesture.capture_samples --label double_ok
python -m double_ok_gesture.capture_samples --label not_double_ok
```

Use those images to check lighting, camera angle, distance, and false triggers.
