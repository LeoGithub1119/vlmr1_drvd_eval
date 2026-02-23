#!/usr/bin/env bash
set -euo pipefail
salloc -p normal --gres=gpu:2 --cpus-per-task=12 -t 08:00:00
