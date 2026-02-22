#!/usr/bin/env bash
set -euo pipefail
salloc -p normal --gres=gpu:1 --cpus-per-task=4 -t 01:00:00
