#!/usr/bin/env python3
"""
Fix imports in mm_utils modules.
"""
import os
import re

MM_UTILS_DIR = "/Users/sunchangzhi/TeleAI/Projects/telemem/telemem/mm_utils"

def fix_imports(filepath):
    """Fix imports in a Python file."""
    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content

    # Fix: from build_database -> from .build_database
    content = re.sub(r'\nfrom build_database import', '\nfrom .build_database import', content)

    # Fix: from core -> from .core
    content = re.sub(r'\nfrom core import', '\nfrom .core import', content)

    # Fix: from frame_caption -> from .frame_caption
    content = re.sub(r'\nfrom frame_caption import', '\nfrom .frame_caption import', content)

    # Fix: from video_utils -> from .video_utils
    content = re.sub(r'\nfrom video_utils import', '\nfrom .video_utils import', content)

    # Fix: from memory_utils -> from .memory_utils (already done, but ensure)
    content = re.sub(r'\nfrom memory_utils import', '\nfrom .memory_utils import', content)

    # Fix: from func_call_shema -> from .func_call_schema (typo in original)
    content = re.sub(r'\nfrom func_call_shema import', '\nfrom .func_call_schema import', content)

    # Only write if changed
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    files = [
        'build_database.py',
        'core.py',
        'frame_caption.py',
        'func_call_schema.py',  # Note: filename was typo in patch
        'memory_utils.py',
        'video_utils.py',
    ]

    for filename in files:
        filepath = os.path.join(MM_UTILS_DIR, filename)
        if os.path.exists(filepath):
            changed = fix_imports(filepath)
            if changed:
                print(f"✓ Fixed imports in {filename}")
            else:
                print(f"  No changes needed in {filename}")
        else:
            print(f"  Warning: {filename} not found")

if __name__ == '__main__':
    main()
