#!/usr/bin/env bash

NAME=$0
COMMAND=$1

function usage {
    echo "usage: $NAME <command>"
    echo
    echo "  install  Install dependencies for running static checks"
    echo "  run      Run static checks"
    exit 1
}

if [[ "$COMMAND" == "install" ]]; then
    pip install codespell flake8 pylint
elif [[ "$COMMAND" == "run" ]]; then
    flake8 -v nengo && codespell -q 3 && pylint nengo
else
    if [[ -z "$COMMAND" ]]; then
        echo "Command required"
    else
        echo "Command $COMMAND not recognized"
    fi
    echo
    usage
fi
