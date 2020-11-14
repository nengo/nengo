import os
import sys

if sys.platform.startswith("win"):  # pragma: no cover
    config_dir = os.path.expanduser(os.path.join("~", ".nengo"))
    cache_dir = os.path.join(config_dir, "cache")
else:
    config_dir = os.path.expanduser(os.path.join("~", ".config", "nengo"))
    cache_dir = os.path.expanduser(os.path.join("~", ".cache", "nengo"))

decoder_cache_dir = os.path.join(cache_dir, "decoders")
install_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
examples_dir = os.path.join(install_dir, "docs", "examples")

nengorc = {
    "system": os.path.join(install_dir, "nengo-data", "nengorc"),
    "user": os.path.join(config_dir, "nengorc"),
    "project": os.path.abspath(os.path.join(os.curdir, "nengorc")),
}
