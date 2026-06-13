"""PyTest configuration.

This file adjusts the Python path so that the package under ``src``
can be imported without installation.  The tests live in
``sucnr1-metaflex/tests`` and the source code resides in
``sucnr1-metaflex/src``.  Inserting this directory into
``sys.path`` allows `import sucnr1_metaflex` to succeed during
testing.
"""

import sys
from pathlib import Path

# Add the src directory to sys.path
current_dir = Path(__file__).resolve()
project_root = current_dir.parents[1]
src_path = project_root / "src"
sys.path.insert(0, str(src_path))