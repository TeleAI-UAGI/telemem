#!/usr/bin/env python3
"""
Extract TeleMem code from patch file and transform imports.
"""
import re
import os

PATCH_FILE = "/Users/sunchangzhi/TeleAI/Projects/telemem/overlay/patches/add-TeleMemory.patch"
OUTPUT_DIR = "/Users/sunchangzhi/TeleAI/Projects/telemem/telemem"

def extract_file_from_patch(lines, target_file):
    """Extract content for a specific file from the patch."""
    content = []
    in_target = False
    for line in lines:
        if f'+++ b/vendor/TeleMem/{target_file}' in line:
            in_target = True
            continue
        if in_target:
            if line.startswith('diff --git') or line.startswith('---'):
                break
            if line.startswith('+') and not line.startswith('+++'):
                content.append(line[1:])  # Remove '+' prefix
    return ''.join(content)

def transform_imports(content, is_mm_utils=False):
    """Transform imports from vendor/TeleMem to relative imports."""
    replacements = [
        ('from TeleMem.configs import', 'from ..config import' if is_mm_utils else 'from .config import'),
        ('from TeleMem.utils import', 'from ..utils import' if is_mm_utils else 'from .utils import'),
        ('from TeleMem.mm_utils.', 'from .' if is_mm_utils else 'from .mm_utils.'),
        ('import TeleMem.configs', 'import ..config' if is_mm_utils else 'import .config'),
        ('import TeleMem.utils', 'import ..utils' if is_mm_utils else 'import .utils'),
    ]

    for old, new in replacements:
        content = content.replace(old, new)

    # Remove sys.path manipulation for mm_utils
    if is_mm_utils:
        content = re.sub(
            r"sys\.path\.insert\(0, os\.path\.join\(BASE_DIR, 'mm_utils'\)\)\n",
            '',
            content
        )
        content = re.sub(
            r"BASE_DIR = os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\n",
            '',
            content
        )

    return content

def main():
    with open(PATCH_FILE, 'r') as f:
        lines = f.readlines()

    files_to_extract = [
        ('__init__.py', False),
        ('main.py', False),
        ('utils.py', False),
        ('mm_utils/__init__.py', True),
        ('mm_utils/build_database.py', True),
        ('mm_utils/core.py', True),
        ('mm_utils/frame_caption.py', True),
        ('mm_utils/func_call_shema.py', True),
        ('mm_utils/memory_utils.py', True),
        ('mm_utils/video_utils.py', True),
    ]

    for filename, is_mm_utils in files_to_extract:
        print(f"Extracting {filename}...")
        content = extract_file_from_patch(lines, filename)

        if not content:
            print(f"  Warning: No content found for {filename}")
            continue

        # Transform imports
        content = transform_imports(content, is_mm_utils)

        # Rename main.py to memory.py
        output_filename = filename
        if filename == 'main.py':
            output_filename = 'memory.py'

        output_path = os.path.join(OUTPUT_DIR, output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(content)

        print(f"  ✓ Written to {output_path}")

    print("\nAll files extracted successfully!")

if __name__ == '__main__':
    main()
