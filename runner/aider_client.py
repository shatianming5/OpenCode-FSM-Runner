from __future__ import annotations

def create_coder(*, model_name: str, fnames: list[str]):
    try:
        from aider.coders import Coder
        from aider.io import InputOutput
        from aider.models import Model
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Failed to import aider. Install deps with `pip install -r requirements.txt`.") from e

    io = InputOutput(yes=True)
    model = Model(model_name)
    return Coder.create(main_model=model, fnames=fnames, io=io)
