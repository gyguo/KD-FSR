"""Set up paths for PFN2"""
import sys
import os


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = os.path.dirname(__file__)

# Add to PYTHONPATH
add_path(os.path.join(this_dir, '..', 'lib'))
add_path(os.path.join(this_dir, '..', 'scripts'))

