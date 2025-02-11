# Draw Logo - ML Regression Model
**TOOLS**
- Tensorboard: https://pytorch.org/docs/stable/tensorboard.html
- Pytorch
- ONNX: https://en.wikipedia.org/wiki/Open_Neural_Network_Exchange
- Numpy
- Houdini
  - ONNX Interference Node: https://www.sidefx.com/docs/houdini/nodes/sop/onnx.html 

**PROCESS**
1. Create training data in Houdini,
2. Build and train our own neural nets with Python and PyTorch
3. Integrate them into our Houdini Workflows using ONNX

**PROJECT**
- In this project a neural net learns to draw the PyTorch logo.
- To do so we create our own training data in Houdini, we build and train our own neural nets with Python and PyTorch and integrate them into a Houdini workflow using ONNX. 
- The logo is fed into Houdini in where we conver the logo to 50,000 points and convert the logo to a float value image.
- This raw data is fed to our neural network we create with Pytorch and Tensorboard provides a much better judgement over how well our model.
<img width="935" alt="Pytorch_Logo_Tensoboard" src="https://github.com/user-attachments/assets/167b0e80-90d5-44df-bde4-488445db548e" />

- The training data is fed back into Houdini and ONNX data is interpreted and rendered.
<img width="180" alt="Pytorch_Logo_50" src="https://github.com/user-attachments/assets/58888149-34b6-439e-bca9-2a4c98021cf6" />
<img width="180" alt="Pytorch_Logo_100" src="https://github.com/user-attachments/assets/f21d826f-1453-4665-b914-7eeaae04fb88" />
<img width="180" alt="Pytorch_Logo_250" src="https://github.com/user-attachments/assets/b761a41b-e076-44ae-a3ed-f1be6d23d8ba" />
<img width="180" alt="Pytorch_Logo_500" src="https://github.com/user-attachments/assets/3e87b939-fedd-469e-80f0-4b26b1cd96c6" />
<img width="180" alt="Pytorch_Logo_1000" src="https://github.com/user-attachments/assets/9184f7d7-5353-4645-a1ed-d3d173638a58" />

<br><br>
**Houdini Snippets**
<img width="1916" alt="input_a input_b target" src="https://github.com/user-attachments/assets/d1a1c3f0-c8a9-4aa0-b48b-47500dfe017d" />
<img width="1919" alt="ONNX_data" src="https://github.com/user-attachments/assets/57bb4817-74e6-4ce4-802f-5d9a24d2ec1c" />
