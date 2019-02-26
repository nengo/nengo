#!/usr/bin/env bash
if [[ ! -e .ci/common.sh || ! -e nengo ]]; then
    echo "Run this script from the root directory of this repository"
    exit 1
fi
source .ci/common.sh

# This script deploys a release to PyPI

NAME=$0
COMMAND=$1

if [[ "$COMMAND" == "install" ]]; then
    pip install check-manifest collective.checkdocs
elif [[ "$COMMAND" == "script" ]]; then
    check-manifest
    python setup.py checkdocs
    if [[ "$TRAVIS_TAG" == "" ]]; then
        # shellcheck disable=SC2086
        TAG="v$(cut -d'-' -f3 <<<$TRAVIS_BRANCH)"
    else
        TAG=$TRAVIS_TAG
    fi
    python -c "from nengo import version; assert version.dev is None, 'this is a dev version'"
    python -c "from nengo import version; assert 'v' + version.version == '$TAG', 'version does not match tag'"
    python -c "from nengo import version; assert any(line.startswith(version.version) and 'unreleased' not in line for line in open('CHANGES.rst').readlines()), 'changelog not updated'"
elif [[ -z "$COMMAND" ]]; then
    echo "$NAME requires a command like 'install' or 'script'"
else
    echo "$NAME does not define $COMMAND"
fi
