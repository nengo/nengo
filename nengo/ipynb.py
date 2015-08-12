"""IPython extension that activates special IPython notebook features of Nengo.

At the moment this only activates the improved progress bar.

Use ``%load_ext nengo.ipynb`` in an IPython notebook to load the extension.

Note
----

This IPython extension cannot be unloaded.
"""

from nengo.rc import rc
from nengo.utils.ipython import has_ipynb_widgets
from nengo.utils.progress import ProgressBar, timestamp2timedelta

if has_ipynb_widgets():
    import IPython
    if IPython.version_info[0] <= 3:
        from IPython.html.widgets import DOMWidget
        import IPython.utils.traitlets as traitlets
    else:
        from ipywidgets import DOMWidget
        import traitlets
    from IPython.display import display
else:
    DOMWidget = object


def load_ipython_extension(ipython):
    if has_ipynb_widgets() and rc.get('progress', 'progress_bar') == 'auto':
        IPythonProgressWidget.load_frontend(ipython)
        rc.set('progress', 'progress_bar', '.'.join((
            __name__, IPython2ProgressBar.__name__)))


class IPythonProgressWidget(DOMWidget):
    """IPython widget for displaying a progress bar."""

    # pylint: disable=too-many-public-methods
    _view_name = traitlets.Unicode('NengoProgressBar', sync=True)
    progress = traitlets.Float(0., sync=True)
    text = traitlets.Unicode(u'', sync=True)

    FRONTEND = '''
    require(["widgets/js/widget", "widgets/js/manager"],
        function(widget, manager) {
      if (typeof widget.DOMWidgetView == 'undefined') {
        widget = IPython;
      }
      if (typeof manager.WidgetManager == 'undefined') {
        manager = IPython;
      }

      var NengoProgressBar = widget.DOMWidgetView.extend({
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
          this.$el.find('div.pb-text').text(text);
        },
      });

      manager.WidgetManager.register_widget_view(
        'NengoProgressBar', NengoProgressBar);
    });'''

    @classmethod
    def load_frontend(cls, ipython):
        """Loads the JavaScript front-end code required by then widget."""
        ipython.run_cell_magic('javascript', '', cls.FRONTEND)


class IPython2ProgressBar(ProgressBar):
    """IPython progress bar based on widgets."""

    supports_fast_ipynb_updates = True

    def __init__(self, task="Simulation"):
        super(IPython2ProgressBar, self).__init__(task)
        self._widget = IPythonProgressWidget()
        self._initialized = False

    def update(self, progress):
        if not self._initialized:
            display(self._widget)
            self._initialized = True

        self._widget.progress = progress.progress
        if progress.finished:
            self._widget.text = "{0} finished in {1}.".format(
                self.task,
                timestamp2timedelta(progress.elapsed_seconds()))
        else:
            self._widget.text = "{progress:.0f}%, ETA: {eta}".format(
                progress=100 * progress.progress,
                eta=timestamp2timedelta(progress.eta()))
