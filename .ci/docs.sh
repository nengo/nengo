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
    conda install --quiet jupyter matplotlib scipy
    pip install -e .[docs]
elif [[ "$COMMAND" == "check" ]]; then
    sphinx-build -b linkcheck -vW -D nbsphinx_execute=never docs docs/_build
elif [[ "$COMMAND" == "run" ]]; then
    git clone -b gh-pages-release https://github.com/nengo/nengo.git ../nengo-docs
    RELEASES=$(find ../nengo-docs -maxdepth 1 -type d -name "v[0-9].*" -printf "%f,")

    if [[ "$TRAVIS_BRANCH" == "$TRAVIS_TAG" ]]; then
        RELEASES="$RELEASES$TRAVIS_TAG"
        sphinx-build -b html docs ../nengo-docs/"$TRAVIS_TAG" -vW -A building_version="$TRAVIS_TAG" -A releases="$RELEASES"
    else
        sphinx-build -b html docs ../nengo-docs -vW -A building_version=latest -A releases="$RELEASES"
    fi
elif [[ "$COMMAND" == "upload" ]]; then
    cd ../nengo-docs
    git config --global user.email "travis@travis-ci.org"
    git config --global user.name "TravisCI"
    git add --all

    if [[ "$TRAVIS_BRANCH" == "$TRAVIS_TAG" ]]; then
        git commit -m "Documentation for release $TRAVIS_TAG"
        git push -q "https://$GH_TOKEN@github.com/nengo/nengo.git" gh-pages-release
    elif [[ "$TRAVIS_BRANCH" == "master" ]]; then
        git commit -m "Last update at $(date '+%Y-%m-%d %T')"
        git push -fq "https://$GH_TOKEN@github.com/nengo/nengo.git" gh-pages-release:gh-pages
    else
        git commit -m "Documentation for branch $TRAVIS_BRANCH"
        git push -fq "https://$GH_TOKEN@github.com/nengo/nengo.git" gh-pages-release:gh-pages-test
    fi
else
    if [[ -z "$COMMAND" ]]; then
        echo "Command required"
    else
        echo "Command $COMMAND not recognized"
    fi
    echo
    usage
fi
