"""
Test cube reconstruction with fixed depth scaling.
"""

import subprocess
import sys

# Run volume-test.py with cube images
print("Running volume-test.py with cube dataset...")
print("=" * 60)

# Modify volume-test.py to use cube images temporarily
import shutil
shutil.copy('volume-test.py', 'volume-test-backup.py')

# Read the file
with open('volume-test.py', 'r') as f:
    content = f.read()

# Replace the three_images list with cube images
old_images = """three_images =  [['./world_rotate/trans_01.png','./world_rotate/trans_02.png','./world_rotate/trans_03.png'],
                ['./world_rotate/trans_03.png','./world_rotate/trans_04.png','./world_rotate/trans_05.png'],
                 ['./world_rotate/trans_05.png','./world_rotate/trans_06.png','./world_rotate/trans_07.png'],
                 ['./world_rotate/trans_07.png','./world_rotate/trans_08.png','./world_rotate/trans_09.png'],
                 ['./world_rotate/trans_09.png','./world_rotate/trans_10.png','./world_rotate/trans_11.png'],
                 ['./world_rotate/trans_11.png','./world_rotate/trans_12.png','./world_rotate/trans_13.png'],
                 ['./world_rotate/trans_13.png','./world_rotate/trans_14.png','./world_rotate/trans_15.png'],
                 ['./world_rotate/trans_15.png','./world_rotate/trans_16.png','./world_rotate/trans_17.png'],
                 ['./world_rotate/trans_17.png','./world_rotate/trans_18.png','./world_rotate/trans_19.png'],
                 ['./world_rotate/trans_19.png','./world_rotate/trans_20.png','./world_rotate/trans_21.png'],
                 ['./world_rotate/trans_21.png','./world_rotate/trans_22.png','./world_rotate/trans_23.png'],
                 ['./world_rotate/trans_23.png','./world_rotate/trans_24.png','./world_rotate/trans_25.png']]"""

new_images = """three_images =  [['./cube_images/cube_01.png','./cube_images/cube_02.png','./cube_images/cube_03.png'],
                ['./cube_images/cube_03.png','./cube_images/cube_04.png','./cube_images/cube_05.png'],
                 ['./cube_images/cube_05.png','./cube_images/cube_06.png','./cube_images/cube_07.png']]"""

content = content.replace(old_images, new_images)

# Write back
with open('volume-test.py', 'w') as f:
    f.write(content)

try:
    # Run the modified script
    result = subprocess.run([sys.executable, 'volume-test.py'], 
                          capture_output=True, text=True, timeout=60)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    print(f"\nExit code: {result.returncode}")
finally:
    # Restore original
    shutil.move('volume-test-backup.py', 'volume-test.py')

print("\n" + "=" * 60)
print("Test complete! Check the generated visualization.")
