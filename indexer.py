"""Convenience entry point: python indexer.py [--rebuild]"""

import sys
from helioai.indexer import build_index

if __name__ == "__main__":
    build_index(rebuild="--rebuild" in sys.argv)
