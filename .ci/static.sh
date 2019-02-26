#!/usr/bin/env bash
if [[ ! -e .ci/common.sh || ! -e nengo ]]; then
    echo "Run this script from the root directory of this repository"
    exit 1
fi
source .ci/common.sh

# This script runs the static style checks

shopt -s globstar

NAME=$0
COMMAND=$1
STATUS=0  # Used to exit with non-zero status if any check fails

if [[ "$COMMAND" == "install" ]]; then
    # pip installs a more recent entrypoints version than conda
    exe pip install entrypoints
    exe conda install --quiet jupyter
    exe pip install codespell flake8 pylint gitlint
elif [[ "$COMMAND" == "script" ]]; then
    # Convert notebooks to Python scripts
    jupyter-nbconvert \
        --log-level WARN \
        --to python \
        --TemplateExporter.exclude_input_prompt=True \
        -- docs/examples/**/*.ipynb
    # Remove style issues introduced in the conversion:
    #   s/# $/#/g replaces lines with just '# ' with '#'
    #   /get_ipython()/d removes lines containing 'get_ipython()'
    sed -i -e 's/# $/#/g' -e '/get_ipython()/d' -- docs/examples/**/*.py
    exe flake8 nengo
    exe flake8 --ignore=E703,W391 docs
    exe pylint docs nengo
    rm docs/examples/**/*.py
    exe codespell -q 3 --skip="./build,./docs/_build,*-checkpoint.ipynb"
    exe shellcheck -e SC2087 .ci/*.sh
    # undo single-branch cloning
    git config --replace-all remote.origin.fetch +refs/heads/*:refs/remotes/origin/*
    git fetch origin master
    N_COMMITS=$(git rev-list --count HEAD ^origin/master)
    for ((i=0; i<N_COMMITS; i++)) do
        git log -n 1 --skip $i --pretty=%B | grep -v '^Co-authored-by:' | exe gitlint -vvv
    done
elif [[ -z "$COMMAND" ]]; then
    echo "$NAME requires a command like 'install' or 'script'"
else
    echo "$NAME does not define $COMMAND"
fi

exit "$STATUS"
