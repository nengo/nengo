#!/bin/bash
set -e

# the miniconda directory may exist if it has been restored from cache
if [ -d "$HOME/miniconda" ] && [ -e "$HOME/miniconda/bin/conda" ]; then
    echo "Miniconda install already present from cache: $HOME/miniconda"
else # if it does not exist, we need to install miniconda
    rm -rf "$HOME/miniconda" # remove the directory in case we have an empty cached directory
    
  wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p "$HOME/miniconda"
  export PATH="$HOME/miniconda/bin:$PATH"
  conda config --set always_yes yes --set changeps1 no
  conda update -q conda
  conda info -a
  conda create -q -n test python="$PYTHON" pip
fi