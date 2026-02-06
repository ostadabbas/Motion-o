#!/usr/bin/env python3
import sys
print("Step 1: Starting", flush=True)

print("Step 2: Adding path", flush=True)
sys.path.insert(0, '/home/bi.ga/Workspace/vlmm-mcot')

print("Step 3: Importing preprocess script", flush=True)
from scripts import preprocess_plm_stc

print("Step 4: Import complete!", flush=True)

print("Step 5: Calling main", flush=True)
sys.argv = ['preprocess_plm_stc.py', '/mnt/data/plm_stc/formatted_test', '/mnt/data/plm_stc/preprocessed_test', '--split', 'train', '--max-frames', '8']
preprocess_plm_stc.main()

print("Step 6: Done!", flush=True)
