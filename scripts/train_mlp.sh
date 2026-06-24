#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "C++ build no longer provides the scikit-learn MLP trainer."
echo "Use scripts/train_numpy_logreg.sh for the built-in C++ logistic model."
exit 2
