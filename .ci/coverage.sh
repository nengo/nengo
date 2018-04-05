#!/usr/bin/env bash

NAME=$0
COMMAND=$1

function usage {
    echo "usage: $NAME <command>"
    echo
    echo "  install  Install dependencies for collecting coverage"
    echo "  run      Run pytest and collect coverage"
    echo "  upload   Upload coverage to codecov.io"
    exit 1
}

if [[ "$COMMAND" == "install" ]]; then
    pip install coverage
elif [[ "$COMMAND" == "run" ]]; then
    coverage run -m pytest nengo -v --duration 20 && coverage report
elif [[ "$COMMAND" == "upload" ]]; then
    eval "bash <(curl -s https://codecov.io/bash)"
else
    if [[ -z "$COMMAND" ]]; then
        echo "Command required"
    else
        echo "Command $COMMAND not recognized"
    fi
    echo
    usage
fi
