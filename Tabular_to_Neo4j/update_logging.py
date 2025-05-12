#!/usr/bin/env python3
"""
Script to update logging configuration across all files in the Tabular_to_Neo4j project.
This script ensures consistent logging patterns throughout the codebase.
"""

import os
import re
import glob
from pathlib import Path

def update_logging_imports(file_path):
    """Update logging imports in a file to use the centralized logging configuration."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if file already uses our logging config
    if 'from Tabular_to_Neo4j.utils.logging_config import get_logger' in content:
        print(f"File already updated: {file_path}")
        return False
    
    # Replace standard logging import with our centralized config
    if re.search(r'import\s+logging', content) or re.search(r'from\s+logging\s+import', content):
        # Remove existing logger initialization
        content = re.sub(r'logger\s*=\s*logging\.getLogger\(__name__\)', '', content)
        
        # Replace import statements
        content = re.sub(r'import\s+logging', 'from Tabular_to_Neo4j.utils.logging_config import get_logger', content)
        
        # Add logger initialization if not present
        if 'logger = get_logger(__name__)' not in content:
            # Find where to insert the logger initialization
            import_section_end = 0
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    import_section_end = i
            
            # Insert after imports but before code
            insert_pos = import_section_end + 1
            while insert_pos < len(lines) and (not lines[insert_pos].strip() or lines[insert_pos].startswith('#')):
                insert_pos += 1
            
            lines.insert(insert_pos, '\n# Configure logging\nlogger = get_logger(__name__)\n')
            content = '\n'.join(lines)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"Updated logging in: {file_path}")
        return True
    
    return False

def main():
    """Update logging configuration in all Python files in the project."""
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Find all Python files in the project
    python_files = []
    for root, _, files in os.walk(project_root):
        for file in files:
            if file.endswith('.py') and file != 'logging_config.py' and file != 'update_logging.py':
                python_files.append(os.path.join(root, file))
    
    # Update logging in each file
    updated_count = 0
    for file_path in python_files:
        if update_logging_imports(file_path):
            updated_count += 1
    
    print(f"\nUpdated logging in {updated_count} files out of {len(python_files)} total Python files.")

if __name__ == "__main__":
    main()
