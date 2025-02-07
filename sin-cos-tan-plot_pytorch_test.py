import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# CHANGE DIRECTORY TO YOURS
writer = SummaryWriter("C:/Users/walab/Documents/Houdini/Entagma/Entagma_ML101/python/runs/test")
r = 5

# Setup Torch
if torch.cuda.is_available():
    my_device = torch.device('cuda')
elif torch.backends.mps.is_available():
    my_device = torch.device('mps')
else:
    my_device = torch.device('cpu')
print(f'Selected device: {my_device}')

# Main Function
print("Adding data to Tensorboard Folder")
for i in range(100):
    writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                    'xcosx':i*np.cos(i/r),
                                    'tanx': np.tan(i/r)}, i)
writer.close()
# This call adds three values to the same scalar plot with the tag
# 'run_14h' in TensorBoard's scalar section.

print("Done")