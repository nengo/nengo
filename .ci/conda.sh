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
    export PATH="$HOME/miniconda/bin:$PATH"
    if ! [[ -d $HOME/miniconda/envs/test ]]; then
        exe rm -rf "$HOME/miniconda"
        exe wget "$MINICONDA" --quiet -O miniconda.sh
        exe bash miniconda.sh -b -p "$HOME/miniconda"
        exe conda create --quiet -y -n test python="$PYTHON" pip
    fi
    exe conda config --set always_yes yes --set changeps1 no
    exe conda update --quiet conda
    exe conda info -a
    exe source activate test
    export PIP_UPGRADE=true  # always upgrade packages to latest version
    exe pip install pip
elif [[ -z "$COMMAND" ]]; then
    echo "$NAME requires a command like 'install' or 'script'"
else
    echo "$NAME does not define $COMMAND"
fi
