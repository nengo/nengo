***********************
Vendorized dependencies
***********************

This directory contains Nengo dependencies
that have been vendorized.
A vendorized dependency is shipped with Nengo
to allow for easy offline install.

To add a new vendorized dependency,
add it to ``nengo/_vendor/requirements.txt`` and run

.. code:: bash

   pip install --target nengo/_vendor -r nengo/_vendor/requirements.txt

from the Nengo root directory.

To update a vendorized dependency,
change the version number associated with that package
in ``nengo/_vendor/requirements.txt``
and rerun the above command
from the Nengo root directory.
