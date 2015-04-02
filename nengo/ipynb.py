"""IPython extension that activates special IPython notebook features of Nengo.

At the moment this only activates the improved progress bar.

Use ``%load_ext nengo.ipynb`` in an IPython notebook to load the extension.

Note
----

This IPython extension cannot be unloaded.
"""

from nengo.rc import rc
from nengo.utils.progress import IPython2ProgressBar, IPythonProgressWidget


def load_ipython_extension(ipython):
    IPythonProgressWidget.load_frontend()
    if rc.get('progress', 'progress_bar') == 'auto':
        rc.set('progress', 'progress_bar', '.'.join((
            'nengo.utils.progress',
            IPython2ProgressBar.__name__)))
