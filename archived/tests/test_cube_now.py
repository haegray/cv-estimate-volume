"""
Quick test of cube reconstruction with current volume-test.py
"""
import sys
import os

# Temporarily modify the three_images list in volume-test.py
print("Testing cube reconstruction...")
print("=" * 60)

# Read volume-test.py
with open('volume-test.py', 'r') as f:
    lines = f.readlines()

# Find and replace the three_images list
in_three_images = False
new_lines = []
skip_until_bracket = False

for i, line in enumerate(lines):
    if 'three_images =  [[' in line and 'world_rotate' in line:
        # Start of three_images list - replace it
        new_lines.append("three_images =  [['./cube_images/cube_01.png','./cube_images/cube_02.png','./cube_images/cube_03.png'],\n")
        new_lines.append("                ['./cube_images/cube_03.png','./cube_images/cube_04.png','./cube_images/cube_05.png'],\n")
        new_lines.append("                 ['./cube_images/cube_05.png','./cube_images/cube_06.png','./cube_images/cube_07.png']]\n")
        skip_until_bracket = True
    elif skip_until_bracket:
        if line.strip().endswith(']]'):
            skip_until_bracket = False
        # Skip lines until we find the end of the list
        continue
    else:
        new_lines.append(line)

# Write temporary file
with open('volume-test-cube-temp.py', 'w') as f:
    f.writelines(new_lines)

# Run it
print("Running modified volume-test.py with cube images...")
os.system(f'{sys.executable} volume-test-cube-temp.py')

# Clean up
os.remove('volume-test-cube-temp.py')

print("\n" + "=" * 60)
print("Check the generated plot for cube reconstruction!")
