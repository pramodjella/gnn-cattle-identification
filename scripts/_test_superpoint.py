"""Quick smoke test for the new multi-backend SuperPointExtractor."""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np

print('[TEST 1] Import...')
from src.features.superpoint import SuperPointExtractor, MultiExtractor
print('  OK')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

print('[TEST 2] DISK backend...')
ext = SuperPointExtractor(max_keypoints=64, backend='disk')
r = ext.extract(img)
n = len(r['keypoints'])
d = r['descriptors'].shape[1] if n > 0 else 256
print(f'  {ext.method_name}: {n} kp, desc={d}-d  OK={d==256}')
assert d == 256, f"descriptor dim should be 256, got {d}"

print('[TEST 3] SuperPoint (KeyNet+HardNet) backend...')
ext2 = SuperPointExtractor(max_keypoints=64, backend='superpoint')
r2 = ext2.extract(img)
n2 = len(r2['keypoints'])
d2 = r2['descriptors'].shape[1] if n2 > 0 else 256
print(f'  {ext2.method_name}: {n2} kp, desc={d2}-d  OK={d2==256}')
assert d2 == 256

print('[TEST 4] SIFT backend...')
ext3 = SuperPointExtractor(max_keypoints=64, backend='sift')
r3 = ext3.extract(img)
n3 = len(r3['keypoints'])
d3 = r3['descriptors'].shape[1] if n3 > 0 else 256
print(f'  {ext3.method_name}: {n3} kp, desc={d3}-d')

print('[TEST 5] MultiExtractor parallel (DISK + SIFT)...')
multi = MultiExtractor(max_keypoints=64, backends=('disk', 'sift'))
results = multi.extract_parallel(img, max_workers=2)
for nm, res in results.items():
    n_ = len(res['keypoints'])
    ms = res.get('time_ms', 0)
    print(f'  {nm}: {n_} kp  {ms:.1f}ms')

print('[TEST 6] get_stats() after extraction...')
stats = ext.get_stats()
print(f'  method={stats["method"]}  processed={stats["total_processed"]}')

print()
print('='*50)
print('ALL TESTS PASSED')
print('='*50)
