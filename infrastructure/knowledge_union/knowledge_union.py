"""
Knowledge Union — merges Fact Core, Third Brain, Master Overlay
"""
import json
from pathlib import Path

BASE = Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')

def get_unified_view(query):
    # Merge sources into unified view
    view = {
        'query': query,
        'fact_core': None,
        'third_brain': None,
        'master_overlay': None
    }
    return view
