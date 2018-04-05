#!/usr/bin/env bash

NAME=$0
COMMAND=$1

function usage {
    echo "usage: $NAME <command>"
    echo
    echo "  install  Install dependencies for building docs"
    echo "  run      Build documentation"
    echo "  upload   Upload documentation to gh-pages branch"
    exit 1
}

if [[ "$COMMAND" == "install" ]]; then
    conda install matplotlib scipy
    pip install "ipython[all]==2.4.1" sphinx guzzle_sphinx_theme ghp-import
elif [[ "$COMMAND" == "run" ]]; then
    rm "$HOME/.ipython/profile_default/ipython_config.py"
    sphinx-build -W docs docs/_build
elif [[ "$COMMAND" == "upload" ]]; then
    DATE=$(date '+%Y-%m-%d %T')
    git config --global user.email "travis@travis-ci.org"
    git config --global user.name "TravisCI"
    ghp-import -m "Last update at $DATE" -b gh-pages docs/_build
    git push -fq "https://$GH_TOKEN@github.com/nengo/nengo.git" gh-pages
else
    if [[ -z "$COMMAND" ]]; then
        echo "Command required"
    else
        echo "Command $COMMAND not recognized"
    fi
    echo
    usage
fi
