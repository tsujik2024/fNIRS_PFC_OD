"""
Data loading and input/output operations for fNIRS data.
"""

from .loaders import (
    read_txt_file,           # Main public function
    _read_metadata,          # Internal but needed
    _read_data               # Internal but needed
)

# Explicit exports
__all__ = [
    'read_txt_file'          # Only expose this to public API
]

# Internal imports for cross-module use
__internals__ = [
    '_read_metadata',
    '_read_data'
]
