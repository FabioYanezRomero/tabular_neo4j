#!/bin/bash
# Script to remove all folders inside samples/, prompt_samples/, and state/
# but not the directories themselves

set -e

for dir in "samples" "prompt_samples" "state"; do
  if [ -d "$dir" ]; then
    echo "Cleaning $dir/ ..."
    find "$dir" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
  else
    echo "$dir/ does not exist, skipping."
  fi
done

echo "Cleanup complete."
