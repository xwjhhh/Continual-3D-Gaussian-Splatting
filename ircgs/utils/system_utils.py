"""
System utility functions.
"""

import os


def mkdir_p(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
