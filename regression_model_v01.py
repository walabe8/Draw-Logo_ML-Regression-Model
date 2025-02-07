import numpy as np
import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

# -------------------------------------------------------------
# Global Variables
# -------------------------------------------------------------

inputs_path = "C:/Users/walab/Documents/Houdini/Entagma/Entagma_ML101/python/dataset/inputs_dim-2_size-50k_res-float32little.raw"
target_path = "C:/Users/walab/Documents/Houdini/Entagma/Entagma_ML101/python/dataset/targets_dim-1_size-50k_res-float32little.raw"
numsamples = 50000
numdims_input = 2 
numdims_target = 1

num_epochs = 1000
onnx_path = "C:/Users/walab/Documents/Houdini/Entagma/Entagma_ML101/python/onnx/myModel_"

training_split = 0.8
writer = SummaryWriter("C:/Users/walab/Documents/Houdini/Entagma/Entagma_ML101/python/runs/RegressionModel_N_v01")

wedges_hidden_layer_number = [2,3,4]
wedges_hidden_layer_dimension = [5,10,15]
cache_path = "C:/Users/walab/Documents/Houdini/Entagma/Entagma_ML101/python/cache/"

#------------------------------------------------------------------------
# Setup Torch
#------------------------------------------------------------------------

torch.manual_seed(0)

if torch.cuda.is_available():
    my_device = torch.device('cuda')
elif torch.backends.mps.is_available():
    my_device = torch.device('mps')
else:
    my_device = torch.device('cpu')

print("Selected device: ", my_device)

#------------------------------------------------------------------------
# Setup Data
#------------------------------------------------------------------------

with open(inputs_path, 'rb') as inputs_file:
    inputs_raw = inputs_file.read()

inputs_np = np.frombuffer(inputs_raw, np.float32)
print("First 4 entries", inputs_np[:4])

with open(target_path, 'rb') as targets_file:
    targets_raw = targets_file.read()

targets_np = np.frombuffer(targets_raw, np.float32)

inputs = torch.tensor(inputs_np, device=my_device)
inputs = torch.reshape(inputs, (numsamples, numdims_input))
print("Input dimensions: ", inputs.shape)

targets = torch.tensor(targets_np, device=my_device)
targets = torch.reshape(targets, (numsamples, numdims_target))

class MyDataSet(Dataset):
    def __init__(self, inputs, targets, begin, end):
        super().__init__()
        self._inputs = inputs
        self._targets = targets
        self._begin = begin
        self._end = end

    def __len__(self):
        return self._end - self._begin
    
    def __getitem__(self, idx):
        input = self._inputs[self._begin + idx]
        target = self._targets[self._begin + idx]
        return input, target

training_end = int(round(numsamples * training_split))
training_data = MyDataSet(inputs, targets, 0, training_end)
testing_data = MyDataSet(inputs, targets, training_end, numsamples)

training_loader = DataLoader(training_data, batch_size=256, shuffle=True)
testing_loader = DataLoader(testing_data, batch_size=256, shuffle=True)

#------------------------------------------------------------------------
# Create Model
#------------------------------------------------------------------------

class MyRegressionModel(nn.Module):
    def __init__(self, input_dimension, target_dimension, hidden_layer_count, hidden_layer_width):
        super().__init__()

        self.sequence = []

        #Input Layer
        self.sequence.append(nn.Linear(input_dimension, hidden_layer_width))
        self.sequence.append(nn.Sigmoid())

        #Hidden Layers
        for i in range(hidden_layer_count):
            self.sequence.append(nn.Linear(hidden_layer_width, hidden_layer_width))
            self.sequence.append(nn.Sigmoid())

        #Output Layer
        self.sequence.append(nn.Linear(hidden_layer_width, target_dimension))
        self.sequence.append(nn.Sigmoid())

        self.net = nn.Sequential(
            *self.sequence
        )

    def forward(self, x):
        return self.net(x)
    
# model = MyRegressionModel(numdims_input, numdims_target)
# model = model.to(my_device)

# loss_fn = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters())

#------------------------------------------------------------------------
# Training Loop
#------------------------------------------------------------------------

def train_epoch(model, dataloader, loss_fn, optimizer):
    model.train()
    losses = []

    for ins, tgts, in dataloader:
        outs = model(ins)
        loss = loss_fn(outs, tgts)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().numpy())

    return np.mean(losses)

def test_epoch(model, dataloader, loss_fn):
    model.eval()
    losses = []

    with torch.no_grad():
        for ins, tgts, in dataloader:
            outs = model(ins)
            loss = loss_fn(outs, tgts)
            losses.append(loss.detach().cpu().numpy())

    return np.mean(losses)

def export_to_onnx(model, dummy_input, export_path, my_device):
    model.to(torch.device('cpu'))
    model.eval()
    dummy_input = dummy_input.to(torch.device('cpu'))

    torch.onnx.export(model, dummy_input, export_path)
    model.to(my_device)

for num_hidden_layers in wedges_hidden_layer_number:
    for num_hidden_dims in wedges_hidden_layer_dimension:

        nn_name = "HL{}_HD{}_".format(num_hidden_layers, num_hidden_dims)

        model = MyRegressionModel(numdims_input, numdims_target, num_hidden_layers, num_hidden_dims)
        model = model.to(my_device)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, training_loader, loss_fn, optimizer)
        test_loss = test_epoch(model, testing_loader, loss_fn)

        writer.add_scalars('Loss_Train',
                        {
                        nn_name: train_loss,
                        },
                        epoch)
        
        writer.add_scalars('Loss_Test',
                        {
                        nn_name: test_loss,
                        },
                        epoch)
        
        if (epoch+1) % 50 == 0:
            writer.flush()

            save_path = onnx_path + nn_name + str(epoch) + '.onnx'
            dummy = torch.zeros((numdims_input))
            export_to_onnx(model, dummy, save_path, my_device)
            print("Saved ONNX:")

        print(epoch, my_device, train_loss)

    save_path = cache_path + nn_name + str(epoch) + '.pt'
    torch.save(model.state_dict(), save_path)
    print("Saved Cache")

writer.flush()
writer.close()
print('Training Done')