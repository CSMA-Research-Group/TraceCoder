#!/bin/sh
set -eu
exec sh "$(dirname "$0")/../start.sh" "$@"
