# import libaries
import numpy as np
import torch as tch

# Initialisation
X = np.array([1, 2, 3, 4], dtype = np.float32)
Y = np.array([2, 4, 6, 8], dtype = np.float32)
omega = 0.0

lr = 0.01   # Learning  rate  
n = 10      # Number of iterations

# Model prediction (H)
def forward(omega, X):
    return omega * X

# Loss - Mean square error (MSE)
def loss(y, H):
    return ((y-H)**2).mean()

# Gradient of the loss
def gradient(y, x, H):
    return np.dot(-2*x, (y-H)).mean()

# Training
for epoch in range(n):
    # Prediction
    H = forward(omega, X)
    # Loss
    L = loss(Y, H)
    # Gradient
    dLdomega = gradient(Y,X,H)
    # Update omega
    omega -= lr*dLdomega

    if epoch % 1 == 0:
        print(f'epoch {epoch + 1}: omega = {omega:.3f}, loss = {L:.5f}')

# Check model
print(f'Prediction autotraining: h(5) = {forward(omega,5):.3f}')