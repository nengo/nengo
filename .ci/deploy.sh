#!/usr/bin/env bash

NAME=$0
COMMAND=$1

function usage {
    echo "usage: $NAME <command>"
    echo
    echo "  install  Install dependencies for deployment"
    echo "  run      Verify branch ready for deployment"
    exit 1
}

if [[ "$COMMAND" == "install" ]]; then
    pip install check-manifest collective.checkdocs
elif [[ "$COMMAND" == "run" ]]; then
  check-manifest
  python setup.py checkdocs
  if [[ "$TRAVIS_TAG" == "" ]]; then
    TAG="v$(cut -d'-' -f3 <<<$TRAVIS_BRANCH)"
  else
    TAG=$TRAVIS_TAG
  fi
  python -c "from nengo import version; assert version.dev is None, 'this is a dev version'"
  python -c "from nengo import version; assert 'v' + version.version == '$TAG', 'version does not match tag'"
  python -c "from nengo import version; assert any(line.startswith(version.version) and 'unreleased' not in line for line in open('CHANGES.rst').readlines()), 'changelog not updated'"
else
    if [[ -z "$COMMAND" ]]; then
        echo "Command required"
    else
        echo "Command $COMMAND not recognized"
    fi
    echo
    usage
fi
