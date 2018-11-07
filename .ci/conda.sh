#!/usr/bin/env bash

NAME=$0
COMMAND=$1
MINICONDA="http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh"

function usage {
    echo "usage: $NAME <command>"
    echo
    echo "  install  Install miniconda and a simple test environment"
    exit 1
}

if [[ "$COMMAND" == "install" ]]; then
    export PATH="$HOME/miniconda/bin:$PATH"
    if ! [[ -d $HOME/miniconda/envs/test ]]; then
      rm -rf $HOME/miniconda
      wget "$MINICONDA" --quiet -O miniconda.sh
      bash miniconda.sh -b -p "$HOME/miniconda"
      conda create --quiet -y -n test python="$PYTHON" pip
    fi
    conda config --set always_yes yes --set changeps1 no
    conda update --quiet conda
    conda info -a
    source activate test
    export PIP_UPGRADE=true  # always upgrade packages to latest version
    pip install pip
else
    if [[ -z "$COMMAND" ]]; then
        echo "Command required"
    else
        echo "Command $COMMAND not recognized"
    fi
    echo
    usage
fi
