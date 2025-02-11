# Machine_Learning_Regression_Model
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
In this project a neural net learns to draw the PyTorch logo.
To do so we create our own training data in Houdini, we build and train our own neural nets with Python and PyTorch and integrate them into a Houdini workflow using ONNX. 
The logo is fed into Houdini in where we conver the logo to 50,000 points and convert the logo to a float value image.
This raw data is fed to our neural network we create with Pytorch and Tensorboard provides a much better judgement over how well our model.
The training data is fed back into Houdini and ONNX data is interpreted and rendered.

<img width="1919" alt="ONNX_data" src="https://github.com/user-attachments/assets/57bb4817-74e6-4ce4-802f-5d9a24d2ec1c" />
