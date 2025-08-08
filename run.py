#!/usr/bin/env python
"""
Redirect script to the new location
This file is kept for backward compatibility
"""

import sys
import os
from pathlib import Path

# Add the scripts directory to the path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# Import and run the actual script
print("[INFO] Scripts have been moved to scripts/ directory")
print()

# Run the actual script
script_path = scripts_dir / "run.py"
with open(script_path, 'r', encoding='utf-8') as f:
    exec(f.read())