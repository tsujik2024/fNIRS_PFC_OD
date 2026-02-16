"""
fNIRS preprocessing modules including filtering, artifact correction, and signal processing.
"""

from .average_channels import average_channels
from .baseline_correction import baseline_subtraction
from .fir_filter import fir_filter
from .short_channel_regression import scr_regression
from .tddr import tddr

__all__ = [
    'average_channels',
    'baseline_subtraction',
    'fir_filter',
    'scr_regression',
    'tddr'
]