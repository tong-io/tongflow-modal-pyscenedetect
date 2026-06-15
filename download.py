"""Modal download entry for pyscenedetect.

Run:
  modal run download.py::download

Self-contained. This plugin installs deps in the image build; no separate
download step is needed.
"""

from __future__ import annotations

import modal

app = modal.App("pyscenedetect-download")


@app.local_entrypoint()
def download() -> None:
    print("No download step required for pyscenedetect.")
