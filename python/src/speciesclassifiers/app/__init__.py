"""The speciesClassifiers app (needs ``pip install "speciesclassifiers[app]"``): ``speciesclassifiers app``."""


def main(port=None, show=True):
    from .main import main as _main
    _main(port=port, show=show)
