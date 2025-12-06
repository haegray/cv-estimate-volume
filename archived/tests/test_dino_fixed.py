"""
Test dinosaur reconstruction with actual camera parameters.
"""
import subprocess
import sys
import shutil

print("Testing dinosaur reconstruction with actual camera parameters...")
print("=" * 60)

# Backup original
shutil.copy('volume-test.py', 'volume-test-backup-dino.py')

# Read the file
with open('volume-test.py', 'r') as f:
    content = f.read()

# Replace the three_images list with dinosaur images
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

new_images = """three_images =  [['./dinoSparseRing/dinoSparseRing/dinoSR0001.png','./dinoSparseRing/dinoSparseRing/dinoSR0002.png','./dinoSparseRing/dinoSparseRing/dinoSR0003.png'],
                ['./dinoSparseRing/dinoSparseRing/dinoSR0004.png','./dinoSparseRing/dinoSparseRing/dinoSR0005.png','./dinoSparseRing/dinoSparseRing/dinoSR0006.png'],
                 ['./dinoSparseRing/dinoSparseRing/dinoSR0007.png','./dinoSparseRing/dinoSparseRing/dinoSR0008.png','./dinoSparseRing/dinoSparseRing/dinoSR0009.png']]"""

content = content.replace(old_images, new_images)

# Write back
with open('volume-test.py', 'w') as f:
    f.write(content)

try:
    # Run the modified script
    result = subprocess.run([sys.executable, 'volume-test.py'], 
                          capture_output=True, text=True, timeout=120)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    print(f"\nExit code: {result.returncode}")
finally:
    # Restore original
    shutil.move('volume-test-backup-dino.py', 'volume-test.py')

print("\n" + "=" * 60)
print("Test complete! Check the generated visualization.")
