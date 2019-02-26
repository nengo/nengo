#!/usr/bin/env bash
if [[ ! -e .ci/common.sh || ! -e nengo ]]; then
    echo "Run this script from the root directory of this repository"
    exit 1
fi
source .ci/common.sh

# This script sets up the conda environment for all the other scripts

NAME=$0
COMMAND=$1
MINICONDA="http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh"

if [[ "$COMMAND" == "install" ]]; then
    wget "$MINICONDA" --quiet -O miniconda.sh
    bash miniconda.sh -b -p "$HOME/miniconda"
    export PATH="$HOME/miniconda/bin:$PATH"
    exe conda config --set always_yes yes --set changeps1 no
    exe conda update --quiet conda
    exe conda info -a
    exe conda create --quiet -n test python="$PYTHON" pip
    exe source activate test
elif [[ -z "$COMMAND" ]]; then
    echo "$NAME requires a command like 'install' or 'script'"
else
    echo "$NAME does not define $COMMAND"
fi
