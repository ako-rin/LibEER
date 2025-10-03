"""
Local fork of LibEER. This __init__ ensures LibEER is a regular package when
installed in editable mode from this repository, shadowing any namespace
package that may remain in site-packages.

Exports modules from the local tree.
"""

# Optionally expose version or fork metadata
__version__ = "0.1.0-ako"

# Nothing else required; presence of this file converts LibEER into a regular
# package. Subpackages (data_utils, models, utils, etc.) are discovered
# normally from the local source tree.
