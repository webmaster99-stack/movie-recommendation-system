"""The serving API: FastAPI on top of the exported model bundle.

Deliberately independent of the `recsys` training package, which needs pandas, implicit and
torch. This package runs on numpy and scipy alone, so it fits a 512 MB host.
"""
