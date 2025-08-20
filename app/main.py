import importlib.util
import pathlib
import sys

_root = pathlib.Path(__file__).resolve().parent.parent / "app.py"
spec = importlib.util.spec_from_file_location("app_root", _root)
mod = importlib.util.module_from_spec(spec)
sys.modules["app_root"] = mod
spec.loader.exec_module(mod)

for _name in dir(mod):
    if not _name.startswith("_"):
        globals()[_name] = getattr(mod, _name)
