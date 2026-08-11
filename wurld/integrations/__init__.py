"""Integrations with third-party training/visualization stacks.

Each module here is import-guarded on its host package; wurld itself never
depends on them. Current: ``nerfstudio_parser`` (a DataParser reading .wurld.webm
directly). Foxglove needs no module — ``wurld extract --format mcap``.
"""
