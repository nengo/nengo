"""Utilities for progress tracking and display to the user."""

import importlib
import os
import sys
import threading
import time
import uuid
import warnings
from datetime import timedelta
from html import escape
from shutil import get_terminal_size

import numpy as np

from ..exceptions import ValidationError
from ..rc import rc
from .ipython import check_ipy_version, get_ipython

if get_ipython() is not None:
    from IPython.display import Javascript, display  # pragma: no cover


class MemoryLeakWarning(UserWarning):
    pass


warnings.filterwarnings("once", category=MemoryLeakWarning)


def timestamp2timedelta(timestamp):
    """Convert timestamp to timedelta."""

    if timestamp == -1:
        return "Unknown"
    return timedelta(seconds=np.ceil(timestamp))


def _load_class(name):
    mod_name, cls_name = name.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


class Progress:
    """Stores and tracks information about the progress of some process.

    This class is to be used as part of a ``with`` statement. Use ``step()`` to
    update the progress.

    Parameters
    ----------
    max_steps : int
        The total number of calculation steps of the process.
    name_during : str, optional
        Short description of the task to be used while it is running.
    name_after : str, optional
        Short description of the task to be used after it has
        finished. Defaults to ``name_during``.

    Attributes
    ----------
    max_steps : int, optional
        The total number of calculation steps of the process, if known.
    name_after : str
        Name of the task to be used after it has finished.
    name_during : str
        Name of the task to be used while it is running.
    steps : int
        Number of completed steps.
    success : bool or None
        Whether the process finished successfully. ``None`` if the process
        did not finish yet.
    time_end : float
        Time stamp of the time the process was finished or aborted.
    time_start : float
        Time stamp of the time the process was started.

    Examples
    --------

    .. testcode::

       from nengo.utils.progress import Progress

       max_steps = 10
       with Progress(max_steps=max_steps) as progress:
           for i in range(max_steps):
               # do something
               progress.step()

    """

    def __init__(self, name_during="", name_after=None, max_steps=None):
        if max_steps is not None and max_steps <= 0:
            raise ValidationError(
                "must be at least 1 (got %d)" % (max_steps,), attr="max_steps"
            )
        self.n_steps = 0
        self.max_steps = max_steps
        self.name_during = name_during
        if name_after is None:
            name_after = name_during
        self.name_after = name_after
        self.time_start = self.time_end = time.time()
        self.finished = False
        self.success = None

    @property
    def progress(self):
        """The current progress as a number from 0 to 1 (inclusive).

        Returns
        -------
        float
        """
        if self.max_steps is None:
            return 0.0
        return min(1.0, self.n_steps / self.max_steps)

    def elapsed_seconds(self):
        """The number of seconds passed since entering the ``with`` statement.

        Returns
        -------
        float
        """
        if self.finished:
            return self.time_end - self.time_start
        else:
            return time.time() - self.time_start

    def eta(self):
        """The estimated number of seconds until the process is finished.

        Stands for estimated time of arrival (ETA).
        If no estimate is available -1 will be returned.

        Returns
        -------
        float
        """
        if self.progress > 0.0:
            return (1.0 - self.progress) * self.elapsed_seconds() / self.progress
        else:
            return -1

    def __enter__(self):
        self.finished = False
        self.success = None
        self.n_steps = 0
        self.time_start = time.time()
        return self

    def __exit__(self, exc_type, dummy_exc_value, dummy_traceback):
        self.success = exc_type is None
        if self.success and self.max_steps is not None:
            self.n_steps = self.max_steps
        self.time_end = time.time()
        self.finished = True

    def step(self, n=1):
        """Advances the progress.

        Parameters
        ----------
        n : int
            Number of steps to advance the progress by.
        """
        self.n_steps += n


class ProgressBar:
    """Visualizes the progress of a process.

    This is an abstract base class that progress bar classes some inherit from.
    Progress bars should visually displaying the progress in some way.
    """

    def update(self, progress):
        """Updates the displayed progress.

        Parameters
        ----------
        progress : Progress
            The progress information to display.
        """
        raise NotImplementedError()

    def close(self):
        """Closes the progress bar.

        Indicates that not further updates will be made.
        """


class NoProgressBar(ProgressBar):
    """A progress bar that does not display anything.

    Helpful in headless situations or when using Nengo as a library.
    """

    def update(self, progress):
        pass


class TerminalProgressBar(ProgressBar):
    """A progress bar that is displayed as ASCII output on ``stdout``."""

    def update(self, progress):
        if progress.finished:
            line = self._get_finished_line(progress)
        elif progress.max_steps is None:
            line = self._get_unknown_progress_line(progress)
        else:
            line = self._get_in_progress_line(progress)
        sys.stdout.write(line)
        sys.stdout.flush()

    def _get_in_progress_line(self, progress):
        line = "[{{}}] ETA: {eta}".format(eta=timestamp2timedelta(progress.eta()))
        percent_str = " {}... {}% ".format(
            progress.name_during, int(100 * progress.progress)
        )
        width, _ = get_terminal_size()
        progress_width = max(0, width - len(line))
        progress_str = (int(progress_width * progress.progress) * "#").ljust(
            progress_width
        )

        percent_pos = (len(progress_str) - len(percent_str)) // 2
        if percent_pos > 0:
            progress_str = (
                progress_str[:percent_pos]
                + percent_str
                + progress_str[percent_pos + len(percent_str) :]
            )

        return "\r" + line.format(progress_str)

    def _get_unknown_progress_line(self, progress):
        """Generates a progress line with continuously moving marker.

        This is to indicate processing while not knowing how far along we
        progressed with the processing.
        """
        duration = progress.elapsed_seconds()
        line = "[{{}}] duration: {duration}".format(
            duration=timestamp2timedelta(duration)
        )
        text = " {}... ".format(progress.name_during)
        width, _ = get_terminal_size()
        marker = ">>>>"
        progress_width = max(0, width - len(line) + 2)
        index_width = progress_width + len(marker)
        i = int(10.0 * duration) % (index_width + 1)
        progress_str = (" " * i) + marker + (" " * (index_width - i))
        progress_str = progress_str[len(marker) : -len(marker)]
        text_pos = (len(progress_str) - len(text)) // 2
        progress_str = (
            progress_str[:text_pos] + text + progress_str[text_pos + len(text) :]
        )
        return "\r" + line.format(progress_str)

    def _get_finished_line(self, progress):
        width, _ = get_terminal_size()
        line = "{} finished in {}.".format(
            progress.name_after, timestamp2timedelta(progress.elapsed_seconds())
        ).ljust(width)
        return "\r" + line

    def close(self):
        sys.stdout.write(os.linesep)
        sys.stdout.flush()


class VdomProgressBar(ProgressBar):  # pragma: no cover
    """A progress bar using a virtual DOM representation.

    This HTML representation can be used in Jupyter lab (>=0.32) environments.

    .. versionadded: 3.0.0
    """

    def __init__(self):
        super().__init__()
        self._uuid = uuid.uuid4()
        self._handle = None
        self.progress = None

    def update(self, progress):
        self.progress = progress
        if self._handle is None:
            self._handle = display(self, display_id=True)
        else:
            self._handle.update(self)

    def _repr_mimebundle_(self, include, exclude, **kwargs):
        return {"application/vdom.v1+json": self._get_vdom(self.progress)}

    def _get_vdom(self, progress):
        return {
            "tagName": "div",
            "attributes": {
                "id": str(self._uuid),
                "style": {
                    "width": "100%",
                    "boxSizing": "border-box",
                    "border": "1px solid #cfcfcf",
                    "borderRadius": "4px",
                    "textAlign": "center",
                    "position": "relative",
                },
            },
            "children": [
                {
                    "tagName": "div",
                    "attributes": {
                        "class": "pb-text",
                        "style": {"position": "absolute", "width": "100%"},
                    },
                    "children": [self._get_text(self.progress)],
                },
                {
                    "tagName": "div",
                    "attributes": {
                        "class": "pb-fill",
                        "style": self._get_fill_style(self.progress),
                    },
                    "children": [
                        {
                            "tagName": "style",
                            "attributes": {"type": "text/css", "scoped": "scoped"},
                            "children": [
                                """
                        @keyframes pb-fill-anim {
                            0% { background-position: 0 0; }
                            100% { background-position: 100px 0; }
                        }}"""
                            ],
                        },
                        "\u00A0",  # non-breaking space
                    ],
                },
            ],
        }

    def _get_text(self, progress):
        if progress is None:
            text = ""
        elif progress.finished:
            text = "{} finished in {}.".format(
                escape(progress.name_after),
                timestamp2timedelta(progress.elapsed_seconds()),
            )
        elif progress.max_steps is None:
            text = "{task}\u2026 duration: {duration}".format(
                task=escape(progress.name_during),
                duration=timestamp2timedelta(progress.elapsed_seconds()),
            )
        else:
            text = "{task}\u2026 {progress:.0f}%, ETA: {eta}".format(
                task=escape(progress.name_during),
                progress=100.0 * progress.progress,
                eta=timestamp2timedelta(progress.eta()),
            )
        return text

    def _get_fill_style(self, progress):
        if progress.max_steps is None:
            style = self._get_unknown_steps_fill_style(progress)
        else:
            style = self._get_known_steps_fill_style(progress)

        if progress.finished:
            style["animation"] = "none"
            style["backgroundImage"] = "none"

        return style

    def _get_known_steps_fill_style(self, progress):
        return {
            "width": "{:.0f}%".format(100.0 * progress.progress),
            "animation": "none",
            "backgroundColor": "#bdd2e6",
            "backgroundImage": "none",
            "transition": "width 0.1s linear" if progress.progress > 0.0 else "none",
        }

    def _get_unknown_steps_fill_style(self, progress):
        return {
            "width": "100%",
            "animation": "pb-fill-anim 2s linear infinite",
            "backgroundColor": "#bdd2e6",
            "backgroundSize": "100px 100%",
            "backgroundImage": (
                "repeating-linear-gradient("
                "90deg, #bdd2e6, #edf2f8 40%, #bdd2e6 80%, #bdd2e6)"
            ),
        }


class HtmlProgressBar(ProgressBar):  # pragma: no cover
    """A progress bar using a HTML representation.

    This HTML representation can be used in Jupyter notebook environments
    and is provided by the *_repr_html_* method that will be automatically
    used by IPython interpreters.

    If the kernel frontend does not support HTML (e.g., in Jupyter qtconsole),
    a warning message will be issued as the ASCII representation.
    """

    def __init__(self):
        super().__init__()
        self._uuid = uuid.uuid4()
        self._handle = None

    def update(self, progress):
        if self._handle is None:
            display(self._HtmlBase(self._uuid))
            self._handle = display(self._js_update(progress), display_id=True)
        else:
            self._handle.update(self._js_update(progress))

    class _HtmlBase:
        def __init__(self, my_uuid):
            self.uuid = my_uuid

        def __repr__(self):
            return (
                "HtmlProgressBar cannot be displayed. Please use the "
                "TerminalProgressBar. It can be enabled with `nengo.rc['progress']"
                "['progress_bar'] = 'nengo.utils.progress.TerminalProgressBar'`."
            )

        def _repr_html_(self):
            return """
                <script>
                    if (Jupyter.version.split(".")[0] < 5) {{
                        var pb = document.getElementById("{uuid}");
                        var text = document.createTextNode(
                            "HMTL progress bar requires Jupyter Notebook >= " +
                            "5.0 or Jupyter Lab. Alternatively, you can use " +
                            "TerminalProgressBar().");
                        pb.parentNode.insertBefore(text, pb);
                    }}
                </script>
                <div id="{uuid}" style="
                    width: 100%;
                    border: 1px solid #cfcfcf;
                    border-radius: 4px;
                    text-align: center;
                    position: relative;">
                  <div class="pb-text" style="
                      position: absolute;
                      width: 100%;">
                    0%
                  </div>
                  <div class="pb-fill" style="
                      background-color: #bdd2e6;
                      width: 0%;">
                    <style type="text/css" scoped="scoped">
                        @keyframes pb-fill-anim {{
                            0% {{ background-position: 0 0; }}
                            100% {{ background-position: 100px 0; }}
                        }}
                    </style>
                    &nbsp;
                  </div>
                </div>""".format(
                uuid=self.uuid
            )

    def _js_update(self, progress):
        if progress is None:
            text = ""
        elif progress.finished:
            text = "{} finished in {}.".format(
                escape(progress.name_after),
                timestamp2timedelta(progress.elapsed_seconds()),
            )
        elif progress.max_steps is None:
            text = "{task}&hellip; duration: {duration}".format(
                task=escape(progress.name_during),
                duration=timestamp2timedelta(progress.elapsed_seconds()),
            )
        else:
            text = "{task}&hellip; {progress:.0f}%, ETA: {eta}".format(
                task=escape(progress.name_during),
                progress=100.0 * progress.progress,
                eta=timestamp2timedelta(progress.eta()),
            )

        if progress.max_steps is None:
            update = self._update_unknown_steps(progress)
        else:
            update = self._update_known_steps(progress)

        if progress.finished:
            finish = """
                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';
            """
        else:
            finish = ""

        return Javascript(
            """
              (function () {{
                  var root = document.getElementById('{uuid}');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = '{text}';
                  {update}
                  {finish}
              }})();
        """.format(
                uuid=self._uuid, text=text, update=update, finish=finish
            )
        )

    def _update_known_steps(self, progress):
        return """
            if ({progress} > 0.) {{
                fill.style.transition = 'width 0.1s linear';
            }} else {{
                fill.style.transition = 'none';
            }}

            fill.style.width = '{progress}%';
            fill.style.animation = 'none';
            fill.style.backgroundImage = 'none'
        """.format(
            progress=100.0 * progress.progress
        )

    def _update_unknown_steps(self, progress):
        return """
            fill.style.width = '100%';
            fill.style.animation = 'pb-fill-anim 2s linear infinite';
            fill.style.backgroundSize = '100px 100%';
            fill.style.backgroundImage = 'repeating-linear-gradient(' +
                '90deg, #bdd2e6, #edf2f8 40%, #bdd2e6 80%, #bdd2e6)';
        """


class VdomOrHtmlProgressBar(ProgressBar):  # pragma: no cover
    """Progress bar using the VDOM or HTML progress bar.

    This progress bar will transmit both representations as part of a MIME
    bundle and it is up to the Jupyter client to pick the preferred version.
    Usually this will be the VDOM if supported, and the HMTL version where VDOM
    is not supported.

    .. versionadded: 3.0.0
    """

    def __init__(self):
        super().__init__()
        self._handle = None
        self._vdom = VdomProgressBar()
        self._html = HtmlProgressBar()

    def update(self, progress):
        self._vdom.progress = progress
        if self._handle is None:
            display(self._get_initial_bundle(progress), raw=True)
            self._handle = display(
                self._get_update_bundle(progress), raw=True, display_id=True
            )
        self._handle.update(self._get_update_bundle(progress), raw=True)

    def _get_initial_bundle(self, progress):
        return {
            "application/vdom.v1+json": {"tagName": "div", "attributes": {}},
            "text/html": self._html._HtmlBase(self._html._uuid)._repr_html_(),
            "text/plain": repr(self._html._HtmlBase(self._html._uuid)),
        }

    def _get_update_bundle(self, progress):
        bundle = self._vdom._repr_mimebundle_([], [])
        bundle["text/html"] = (
            "<script>"
            + self._html._js_update(progress)._repr_javascript_()
            + "</script>"
        )
        return bundle


class IPython5ProgressBar(ProgressBar):  # pragma: no cover
    """ProgressBar for IPython>=5 environments.

    Provides a VDOM/HTML representation, except for in a pure terminal IPython
    (i.e. not an IPython kernel that was connected to via ZMQ), where a
    ASCII progress bar will be used.

    Note that some Jupyter environments (like qtconsole) will try to use the
    VDOM/HTML version, but do not support HTML and will show a warning instead
    of an actual progress bar.
    """

    def __init__(self):
        super().__init__()

        class Displayable:
            def __init__(self):
                self.display_requested = False

            def _ipython_display_(self):
                self.display_requested = True

        d = Displayable()
        display(d, exclude=["text/plain"])

        if d.display_requested:
            self._progress_bar = VdomOrHtmlProgressBar()
        else:
            self._progress_bar = TerminalProgressBar()

    def update(self, progress):
        self._progress_bar.update(progress)


class WriteProgressToFile(ProgressBar):
    """Writes progress to a file.

    This is useful for remotely and intermittently monitoring progress.
    Note that this file will be overwritten on each update of the progress!

    Parameters
    ----------
    filename : str
        Path to the file to write the progress to.
    """

    def __init__(self, filename):
        self.filename = filename
        super().__init__()

    def update(self, progress):
        if progress.finished:
            text = "{} finished in {}.".format(
                progress.name_after, timestamp2timedelta(progress.elapsed_seconds())
            )
        else:
            text = "{progress:.0f}%, ETA: {eta}".format(
                progress=100 * progress.progress,
                eta=timestamp2timedelta(progress.eta()),
            )

        with open(self.filename, "w") as f:
            f.write(text + os.linesep)


class AutoProgressBar(ProgressBar):
    """Suppresses the progress bar unless the ETA exceeds a threshold.

    Parameters
    ----------
    delegate : ProgressBar
        The actual progress bar to display, if ETA is high enough.
    min_eta : float, optional
        The minimum ETA threshold for displaying the progress bar.
    """

    def __init__(self, delegate, min_eta=1.0):
        self.delegate = delegate

        super().__init__()

        self.min_eta = min_eta
        self._visible = False

    def update(self, progress):
        min_delay = progress.time_start + 0.1
        long_eta = (
            progress.elapsed_seconds() + progress.eta() > self.min_eta
            and min_delay < time.time()
        )
        if self._visible:
            self.delegate.update(progress)
        elif long_eta or progress.finished:
            self._visible = True
            self.delegate.update(progress)

    def close(self):
        self.delegate.close()


class ProgressTracker:
    """Tracks the progress of some process with a progress bar.

    Parameters
    ----------
    progress_bar : ProgressBar or bool or None
        The progress bar to display the progress (or True to use the default
        progress bar, False/None to disable progress bar).
    total_progress : int
        Maximum number of steps of the process.
    update_interval : float, optional
        Time to wait (in seconds) between updates to progress bar display.
    """

    def __init__(self, progress_bar, total_progress, update_interval=0.1):
        self.progress_bar = to_progressbar(progress_bar)
        self.total_progress = total_progress
        self.update_interval = update_interval
        self.update_thread = threading.Thread(target=self.update_loop)
        self.update_thread.daemon = True
        self._closing = False
        self.sub_progress = None

    def next_stage(self, name_during="", name_after=None, max_steps=None):
        """Begin tracking progress of a new stage.

        Parameters
        ----------
        max_steps : int, optional
            The total number of calculation steps of the process.
        name_during : str, optional
            Short description of the task to be used while it is running.
        name_after : str, optional
            Short description of the task to be used after it has
            finished. Defaults to *name_during*.
        """
        if self.sub_progress is not None:
            self.total_progress.step()
        self.sub_progress = Progress(name_during, name_after, max_steps)
        return self.sub_progress

    def __enter__(self):
        self._closing = False
        self.total_progress.__enter__()
        if not isinstance(self.progress_bar, NoProgressBar):
            self.update_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._closing = True
        self.total_progress.__exit__(exc_type, exc_value, traceback)
        if not isinstance(self.progress_bar, NoProgressBar):
            self.update_thread.join()
        self.progress_bar.update(self.total_progress)
        self.progress_bar.close()

    def update_loop(self):
        """Update the progress bar display (will run in a separate thread)."""

        while not self._closing:
            if self.sub_progress is not None and not self.sub_progress.finished:
                self.progress_bar.update(self.sub_progress)
            else:
                self.progress_bar.update(self.total_progress)
            time.sleep(self.update_interval)


def get_default_progressbar():
    """The default progress bar to use depending on the execution environment.

    Returns
    -------
    ``ProgressBar``
    """
    try:
        pbar = rc["progress"].getboolean("progress_bar")
        if pbar:
            pbar = "auto"
        else:
            pbar = "none"
    except ValueError:
        pbar = rc["progress"]["progress_bar"]

    if pbar.lower() == "auto":
        if get_ipython() is not None and check_ipy_version((5, 0)):  # pragma: no cover
            return AutoProgressBar(IPython5ProgressBar())
        else:
            return AutoProgressBar(TerminalProgressBar())
    if pbar.lower() == "none":
        return NoProgressBar()

    return _load_class(pbar)()


def to_progressbar(progress_bar):
    """Converts to a ``ProgressBar`` instance.

    Parameters
    ----------
    progress_bar : None, bool, or ProgressBar
        Object to be converted to a ``ProgressBar``.

    Returns
    -------
    ProgressBar
        Return ``progress_bar`` if it is already a progress bar, the default
        progress bar if ``progress_bar`` is ``True``, and ``NoProgressBar`` if
        it is ``None`` or ``False``.
    """
    if progress_bar is False or progress_bar is None:
        progress_bar = NoProgressBar()
    if progress_bar is True:
        progress_bar = get_default_progressbar()
    return progress_bar
