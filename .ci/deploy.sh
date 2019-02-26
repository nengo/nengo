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
    exe conda install --quiet "$NUMPY"
    exe pip install check-manifest collective.checkdocs
elif [[ "$COMMAND" == "script" ]]; then
    exe check-manifest
    exe python setup.py checkdocs
    if [[ "$TRAVIS_TAG" == "" ]]; then
        TAG=v$(cut -d'-' -f3 <<<"$TRAVIS_BRANCH")
    else
        TAG=$TRAVIS_TAG
    fi
    exe python -c "from nengo import version; \
        assert version.dev is None, 'this is a dev version'"
    exe python -c "from nengo import version; \
        assert 'v' + version.version == '$TAG', 'version does not match tag'"
    exe python -c "from nengo import version; \
        assert any(line.startswith(version.version) \
        and 'unreleased' not in line \
        for line in open('CHANGES.rst').readlines()), \
        'changelog not updated'"
elif [[ -z "$COMMAND" ]]; then
    echo "$NAME requires a command like 'install' or 'script'"
else
    echo "$NAME does not define $COMMAND"
fi

exit "$STATUS"
