import os
import sys


if sys.platform.startswith('win'):
    config_dir = os.path.expanduser(os.path.join("~", ".nengo"))
    cache_dir = os.path.join(config_dir, "cache")
else:
    cache_dir = os.path.expanduser(os.path.join("~", ".cache", "nengo"))

install_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
examples_dir = os.path.join(install_dir, "examples")
