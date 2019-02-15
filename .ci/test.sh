#!/usr/bin/env bash

NAME=$0
COMMAND=$1

function usage {
    echo "usage: $NAME <command>"
    echo
    echo "  install  Install dependencies for running test"
    echo "  run      Run tests"
    exit 1
}

if [[ "$COMMAND" == "install" ]]; then
    conda install --quiet jupyter matplotlib numpy="$NUMPY"
    if [[ "$SCIPY" == "true" ]]; then
        conda install --quiet scipy
        pip install scikit-learn
    fi
    pip install pytest
    pip install -e .
elif [[ "$COMMAND" == "run" ]]; then
    python -c "import numpy; numpy.show_config()"
    pytest nengo -v --duration 20
else
    if [[ -z "$COMMAND" ]]; then
        echo "Command required"
    else
        echo "Command $COMMAND not recognized"
    fi
    echo
    usage
fi
