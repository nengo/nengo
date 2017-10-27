"""IPython extension that activates special Jupyter notebook features of Nengo.

At the moment this only activates the improved progress bar.

Use ``%load_ext nengo.ipynb`` in a Jupyter notebook to load the extension.

Note
----

This IPython extension cannot be unloaded.
"""

try:
    from html import escape
except ImportError:
    from cgi import escape as cgi_escape
    escape = lambda s, quote=True: cgi_escape(s, quote=quote)
import warnings

import IPython

from nengo.rc import rc
from nengo.utils.ipython import has_ipynb_widgets
from nengo.utils.progress import ProgressBar, timestamp2timedelta

if has_ipynb_widgets():
    if IPython.version_info[0] <= 3:
        from IPython.html.widgets import DOMWidget
        import IPython.utils.traitlets as traitlets
    else:
        import ipywidgets
        from ipywidgets import DOMWidget
        import traitlets
    from IPython.display import display
else:
    raise ImportError(
        "Required dependency could not be loaded. Please install ipywidgets.")


try:
    import notebook
    notebook_version = notebook.version_info
except ImportError:
    notebook_version = IPython.version_info


def load_ipython_extension(ipython):
    # Not a deprecation warning as this are hidden by default and we want to
    # make sure the user sees this message.
    warnings.warn(
        "Loading the nengo.ipynb notebook extension will break with current "
        "Jupyter notebook versions and is deprecated. All features provided "
        "by the old extension are automatically activated for IPython>=5.")
    if has_ipynb_widgets() and rc.get('progress', 'progress_bar') == 'auto':
        IPythonProgressWidget.load_frontend(ipython)
        rc.set('progress', 'progress_bar', '.'.join((
            __name__, IPython2ProgressBar.__name__)))


class IPythonProgressWidget(DOMWidget):
    """IPython widget for displaying a progress bar."""

    # pylint: disable=too-many-public-methods
    _view_name = traitlets.Unicode('NengoProgressBar', sync=True)
    if notebook_version[0] >= 4:
        _view_module = traitlets.Unicode('nengo', sync=True)
    progress = traitlets.Float(0., sync=True)
    text = traitlets.Unicode(u'', sync=True)

    WIDGET = '''
      var NengoProgressBar = widgets.DOMWidgetView.extend({
        render: function() {
          // Work-around for messed up CSS in IPython 4
          $('.widget-subarea').css({flex: '2 1 0%'});
          // $el is the DOM of the widget
          this.$el.css({width: '100%', marginBottom: '0.5em'});
          this.$el.html([
            '<div style="',
                'width: 100%;',
                'border: 1px solid #cfcfcf;',
                'border-radius: 4px;',
                'text-align: center;',
                'position: relative;">',
              '<div class="pb-text" style="',
                  'position: absolute;',
                  'width: 100%;">',
                '0%',
              '</div>',
              '<div class="pb-bar" style="',
                  'background-color: #bdd2e6;',
                  'width: 0%;',
                  'transition: width 0.1s linear;">',
                '&nbsp;',
              '</div>',
            '</div>'].join(''));
        },

        update: function() {
          this.$el.css({width: '100%', marginBottom: '0.5em'});
          var progress = 100 * this.model.get('progress');
          var text = this.model.get('text');
          this.$el.find('div.pb-bar').width(progress.toString() + '%');
          this.$el.find('div.pb-text').html(text);
        },
      });
    '''

    FRONTEND = '''
    define('nengo', ["jupyter-js-widgets"], function(widgets) {{
        {widget}

      return {{
        NengoProgressBar: NengoProgressBar
      }};
    }});'''.format(widget=WIDGET)

    LEGACY_FRONTEND = '''
    require(["widgets/js/widget", "widgets/js/manager"],
        function(widgets, manager) {{
      if (typeof widgets.DOMWidgetView == 'undefined') {{
        widgets = IPython;
      }}
      if (typeof manager.WidgetManager == 'undefined') {{
        manager = IPython;
      }}

      {widget}

      manager.WidgetManager.register_widget_view(
        'NengoProgressBar', NengoProgressBar);
    }});'''.format(widget=WIDGET)

    LEGACY_4_FRONTEND = '''
    define('nengo', ["widgets/js/widget"], function(widgets) {{
        {widget}

      return {{
        NengoProgressBar: NengoProgressBar
      }};
    }});'''.format(widget=WIDGET)

    @classmethod
    def load_frontend(cls, ipython):
        """Loads the JavaScript front-end code required by then widget."""
        warnings.warn(
            "The IPythonProgressWidget is deprecated and will break current "
            "Jupyter notebook versions.", DeprecationWarning)
        if notebook_version[0] < 4:
            ipython.run_cell_magic('javascript', '', cls.LEGACY_FRONTEND)
        elif ipywidgets.version_info[0] < 5:
            nb_ver_4x = (notebook_version[0] == 4 and notebook_version[1] > 1)
            if notebook_version[0] > 4 or nb_ver_4x:
                warnings.warn(
                    "Incompatible versions of notebook and ipywidgets "
                    "detected. Please update your ipywidgets package to "
                    "version 5 or above.")
            ipython.run_cell_magic('javascript', '', cls.LEGACY_4_FRONTEND)
        else:
            ipython.run_cell_magic('javascript', '', cls.FRONTEND)


class IPython2ProgressBar(ProgressBar):
    """IPython progress bar based on widgets."""

    supports_fast_ipynb_updates = True

    def __init__(self, task):
        warnings.warn(
            "IPython2ProgressBar is deprecated and will be removed. "
            "Use nengo.progress.IPython5ProgressBar for IPython>=5.",
            DeprecationWarning)
        super(IPython2ProgressBar, self).__init__(task)
        self._escaped_task = escape(task)
        self._widget = IPythonProgressWidget()
        self._initialized = False

    def update(self, progress):
        if not self._initialized:
            display(self._widget)
            self._initialized = True

        self._widget.progress = progress.progress
        if progress.finished:
            self._widget.text = "{} finished in {}.".format(
                self._escaped_task,
                timestamp2timedelta(progress.elapsed_seconds()))
        else:
            self._widget.text = (
                "{task}&hellip; {progress:.0f}%, ETA: {eta}".format(
                    task=self._escaped_task, progress=100 * progress.progress,
                    eta=timestamp2timedelta(progress.eta())))
