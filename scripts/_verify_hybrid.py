import sys; sys.path.insert(0, '.')
from src.utils import load_config
from src.training.image_dataset import create_hybrid_loaders
config = load_config()
loaders = create_hybrid_loaders('data/preprocessed', 'data/graphs', config)
tr = len(loaders['train'].dataset)
vl = len(loaders['val'].dataset)
te = len(loaders['test'].dataset)
print(f'Train: {tr} | Val: {vl} | Test: {te}')
print('Hybrid dataset OK!')
