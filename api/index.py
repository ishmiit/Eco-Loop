import os
import sys

# Add the src directory to the path so the ecoloop module can be found
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ecoloop.server.app import app
