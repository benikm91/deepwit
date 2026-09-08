#!/bin/bash

# Downloads the MNIST dataset that the mnistClassification, autoencoder and variationalAutoencoder
# examples train on (~55 MB).
#
#   ./setup-mnist.sh            # into data/ next to this script
#   ./setup-mnist.sh <dir>      # into <dir>
#
# The target defaults to `data` next to this script, which is where MNISTLoader looks: a forked
# `examples/runMain` runs with the examples directory as its working directory.

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGET_DIR="${1:-$SCRIPT_DIR/data}"
BASE_URL="https://ossci-datasets.s3.amazonaws.com/mnist"

# Create the directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# List of files to download
FILES=(
  "train-images-idx3-ubyte"
  "train-labels-idx1-ubyte"
  "t10k-images-idx3-ubyte"
  "t10k-labels-idx1-ubyte"
)

echo "Starting MNIST dataset setup in $TARGET_DIR..."

failed=0
for FILE in "${FILES[@]}"; do
  # Check if the uncompressed file already exists
  if [ -f "$TARGET_DIR/$FILE" ]; then
    echo "Skipping $FILE (already exists)"
  else
    echo "Downloading and decompressing $FILE..."
    # Download via curl and pipe directly into gunzip to save to the target directory
    # A failed download must not leave a truncated file behind that the next run would skip.
    if curl -fL --retry 3 "$BASE_URL/$FILE.gz" | gunzip > "$TARGET_DIR/$FILE.part"; then
      mv "$TARGET_DIR/$FILE.part" "$TARGET_DIR/$FILE"
      echo "Successfully saved to $TARGET_DIR/$FILE"
    else
      rm -f "$TARGET_DIR/$FILE.part"
      echo "Error: Failed to process $FILE" >&2
      failed=1
    fi
  fi
done

if [ "$failed" -ne 0 ]; then
  echo "MNIST setup incomplete; re-run to retry." >&2
  exit 1
fi

echo "MNIST setup complete."
