"""Put the repo root on sys.path so `app` and `training` import in tests."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
