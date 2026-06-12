"""Test DeDoDe backend specifically."""
import sys
sys.path.insert(0, '.')
import numpy as np

print('[TEST] DeDoDe backend...')
from src.features.superpoint import SuperPointExtractor
ext = SuperPointExtractor(max_keypoints=64, backend='dedode')
img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
r = ext.extract(img)
n = len(r['keypoints'])
d = r['descriptors'].shape[1] if n > 0 else 'N/A'
print(f'  {ext.method_name}: {n} kp, desc_dim={d}')
if n > 0:
    assert r['descriptors'].shape[1] == 256, "desc dim must be 256"
print('DeDoDe test PASSED')
