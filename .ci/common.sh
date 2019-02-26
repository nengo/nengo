#!/usr/bin/env bash
# Small snippets common to all CI scripts.
# All CI scripts should source this script.

STATUS=0  # used to exit with non-zero status if any command fails

exe() {
    echo "\$ $*";
    # shellcheck disable=SC2034
    "$@" || STATUS=1;
}
