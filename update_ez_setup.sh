#!/usr/bin/env bash
# update_ez_setup.sh: Replace the current ez_setup.py with the most recent one

if hash wget 2>/dev/null; then
    wget -N https://bootstrap.pypa.io/ez_setup.py
else
    curl -O https://bootstrap.pypa.io/ez_setup.py
fi
