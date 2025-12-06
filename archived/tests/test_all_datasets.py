"""
Test all three datasets with command line arguments.
"""
import subprocess
import sys

datasets = ['world', 'cube', 'dino']

print("Testing all datasets with command line arguments")
print("=" * 60)

for dataset in datasets:
    print(f"\nTesting {dataset.upper()} dataset...")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, 'volume-test.py', '--dataset', dataset, '--num-triplets', '2'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"✅ {dataset.upper()} dataset: SUCCESS")
            # Print key output lines
            for line in result.stdout.split('\n'):
                if 'Using' in line or 'Processing' in line or 'Triangulated' in line or 'volume' in line.lower():
                    print(f"  {line}")
        else:
            print(f"❌ {dataset.upper()} dataset: FAILED")
            print(f"  Error: {result.stderr[:200]}")
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ {dataset.upper()} dataset: TIMEOUT")
    except Exception as e:
        print(f"❌ {dataset.upper()} dataset: ERROR - {e}")

print("\n" + "=" * 60)
print("All dataset tests complete!")
