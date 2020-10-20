import pathlib
import sys

if sys.platform.startswith("win"):  # pragma: no cover
    config_dir = pathlib.Path.home() / ".nengo"
    cache_dir = config_dir / "cache"
else:
    config_dir = pathlib.Path.home() / ".config" / "nengo"
    cache_dir = pathlib.Path.home() / ".cache" / "nengo"

decoder_cache_dir = cache_dir / "decoders"
install_dir = pathlib.Path(__file__).parents[2]
data_dir = install_dir / "nengo" / "data"
examples_dir = install_dir / "docs" / "examples"

nengorc = {
    "system": install_dir / "nengo-data" / "nengorc",
    "user": config_dir / "nengorc",
    "project": pathlib.Path.cwd() / "nengorc",
}
