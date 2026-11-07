import torch
import numpy as np

data = torch.load('dataset/test_split_FINAL.pt', weights_only=False)

for item in data:
    name = item.name if hasattr(item, 'name') else item.get('name')
    if name == '10.16.txt_2':
        print(f"=== Found {name} ===")
        # check attributes
        for attr in ['p', 'deadlines', 'profits', 'pos', 'coords', 'dist_matrix', 'edge_index', 'edge_attr']:
            if hasattr(item, attr):
                val = getattr(item, attr)
                print(f"{attr}: shape={getattr(val, 'shape', None)}, type={type(val)}")
        
        # let's print all attributes of item
        print("All attributes:", dir(item))
        break
