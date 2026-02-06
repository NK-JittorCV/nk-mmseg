#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from importlib import import_module
from models import build_model

config = getattr(import_module('local_configs.NYUDepthv2.DFormer_Large'), 'C')
model = build_model(config)
state_dict = model.state_dict()

print('All Jittor parameters:')
for key in sorted(state_dict.keys()):
    print(f'  {key}: {state_dict[key].shape}')

print(f'\nTotal parameters: {len(state_dict)}')

# Check for classification related
print('\nClassification related parameters:')
for key in sorted(state_dict.keys()):
    if any(word in key.lower() for word in ['head', 'pred', 'cls', 'classifier', 'fuse', 'linear', 'conv_seg']):
        print(f'  {key}: {state_dict[key].shape}')
