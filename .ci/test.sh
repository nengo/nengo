#!/usr/bin/env bash
if [[ ! -e .ci/common.sh || ! -e nengo ]]; then
    echo "Run this script from the root directory of this repository"
    exit 1
fi
source .ci/common.sh

# This script runs the test suite

NAME=$0
COMMAND=$1

if [[ "$COMMAND" == "install" ]]; then
    exe conda install --quiet jupyter matplotlib numpy="$NUMPY"
    if [[ "$SCIPY" == "true" ]]; then
        exe conda install --quiet scipy
        exe pip install scikit-learn
    fi
    exe pip install pytest
    if [[ "$COVERAGE" == "true" ]]; then
        exe pip install coverage
    fi
    exe pip install -e . --no-deps
elif [[ "$COMMAND" == "script" ]]; then
    exe python -c "import numpy; numpy.show_config()"
    if [[ "$COVERAGE" == "true" ]]; then
        exe coverage run -m pytest nengo -v --duration 20 --color=yes --plots
        exe coverage report
    else
        exe pytest nengo -v --duration 20 --color=yes
    fi
elif [[ "$COMMAND" == "after_script" ]]; then
    if [[ "$COVERAGE" == "true" ]]; then
        eval "bash <(curl -s https://codecov.io/bash)"
    fi
elif [[ -z "$COMMAND" ]]; then
    echo "$NAME requires a command like 'install' or 'script'"
else
    echo "$NAME does not define $COMMAND"
fi

exit "$STATUS"
