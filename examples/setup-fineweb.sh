#!/bin/bash

# Downloads the GPT-2-tokenized FineWeb-10B shards that the `gpt` example trains on.
#
# The shards are the ones modded-nanogpt/llm.c produce, re-hosted on the Hugging Face Hub so a new
# machine does not have to re-tokenize. Each shard is 100M tokens (~200 MB): a 1024-byte header
# followed by the tokens as little-endian uint16, which is the layout FineWebDataset.loadShard reads.
#
#   ./setup-fineweb.sh          # validation shard + 1 train shard (~400 MB), enough to smoke-test
#   ./setup-fineweb.sh 8        # validation shard + train shards 1..8 (~1.6 GB)
#   ./setup-fineweb.sh all      # the full 10B tokens (103 train shards, ~21 GB)
#
# The target directory defaults to `data/fineweb10B` next to this script, which is where
# FineWebDataset looks when FINEWEB_DIR is not set. Override it with FINEWEB_DIR or a second
# argument. Interrupted downloads resume, and completed shards are skipped.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_URL="https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main"
MAGIC=20240520
TOTAL_TRAIN_SHARDS=103

NUM_TRAIN_SHARDS="${1:-1}"
TARGET_DIR="${2:-${FINEWEB_DIR:-$SCRIPT_DIR/data/fineweb10B}}"

if [ "$NUM_TRAIN_SHARDS" = "all" ]; then
  NUM_TRAIN_SHARDS=$TOTAL_TRAIN_SHARDS
fi

if ! [[ "$NUM_TRAIN_SHARDS" =~ ^[0-9]+$ ]] || [ "$NUM_TRAIN_SHARDS" -lt 1 ] || [ "$NUM_TRAIN_SHARDS" -gt "$TOTAL_TRAIN_SHARDS" ]; then
  echo "Usage: $0 [1-$TOTAL_TRAIN_SHARDS|all] [target_dir]" >&2
  exit 1
fi

# Reads a uint32 from FILE at byte OFFSET. The shards are little-endian, as is every host we run on.
read_uint32() {
  od -An -tu4 -j "$2" -N4 "$1" | tr -d ' '
}

file_size() {
  wc -c < "$1" | tr -d ' '
}

# A shard is complete when its header says what it should and the token bytes are all there.
verify_shard() {
  local file="$1"
  [ -f "$file" ] || return 1
  [ "$(file_size "$file")" -ge 1024 ] || return 1

  local magic tokens
  magic=$(read_uint32 "$file" 0)
  tokens=$(read_uint32 "$file" 8)
  [ "$magic" = "$MAGIC" ] || return 1
  [ "$(file_size "$file")" = "$((1024 + 2 * tokens))" ]
}

download_shard() {
  local name="$1"
  local file="$TARGET_DIR/$name"

  if verify_shard "$file"; then
    echo "Skipping $name (already complete)"
    return 0
  fi

  echo "Downloading $name..."
  # `-C -` resumes a partial download; the partial file only becomes $name once it verifies.
  if ! curl -fL --retry 3 -C - -o "$file.part" "$BASE_URL/$name"; then
    echo "Error: failed to download $name" >&2
    return 1
  fi

  if ! verify_shard "$file.part"; then
    echo "Error: $name failed its header check; delete $file.part and retry" >&2
    return 1
  fi

  mv "$file.part" "$file"
  echo "Saved $file"
}

mkdir -p "$TARGET_DIR"

echo "Starting FineWeb-10B setup in $TARGET_DIR ($NUM_TRAIN_SHARDS of $TOTAL_TRAIN_SHARDS train shards, ~$((200 * (NUM_TRAIN_SHARDS + 1))) MB)..."

failed=0
download_shard "fineweb_val_000000.bin" || failed=1
for i in $(seq 1 "$NUM_TRAIN_SHARDS"); do
  download_shard "$(printf 'fineweb_train_%06d.bin' "$i")" || failed=1
done

if [ "$failed" -ne 0 ]; then
  echo "FineWeb setup incomplete; re-run to resume." >&2
  exit 1
fi

echo "FineWeb setup complete."
