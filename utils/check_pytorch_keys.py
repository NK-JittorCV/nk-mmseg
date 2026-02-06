#!/usr/bin/env python3
import torch

checkpoint = torch.load('checkpoints/trained/NYUv2_DFormer_Large.pth', map_location='cpu')
pytorch_state_dict = checkpoint['model']

print('All PyTorch parameters:')
for key in sorted(pytorch_state_dict.keys()):
    print(f'  {key}: {pytorch_state_dict[key].shape}')

print(f'\nTotal parameters: {len(pytorch_state_dict)}')

# Check for classification related
print('\nClassification related parameters:')
for key in sorted(pytorch_state_dict.keys()):
    if any(word in key.lower() for word in ['head', 'pred', 'cls', 'classifier', 'fuse', 'linear']):
        print(f'  {key}: {pytorch_state_dict[key].shape}')
