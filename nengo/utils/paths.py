import os
import sys


if sys.platform.startswith('win'):
    config_dir = os.path.expanduser(os.path.join("~", ".nengo"))
    cache_dir = os.path.join(config_dir, "cache")
else:
    cache_dir = os.path.expanduser(os.path.join("~", ".cache", "nengo"))
