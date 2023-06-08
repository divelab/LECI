r"""
This module includes GNNs used in our leaderboard. It includes: GINs, GINvirtualnodes, and GCNs, in which GCNs are only
for node classifications.
"""

import glob
from os.path import dirname, basename, isfile, join

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

from . import *
