# Provide a stable import path for the fork under a unique top-level package name
# and re-export the original LibEER package API for backward compatibility.

from importlib import import_module

# Re-export original structure if available
try:
    LibEER = import_module('LibEER')
except Exception:
    LibEER = None

__all__ = ['LibEER']
