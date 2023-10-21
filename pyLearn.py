# import libaries
import torch as tch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Initialisation (Model:'y = 2*x')
X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20,random_state=1)
X = tch.from_numpy(X_numpy.astype(np.float32))
Y = tch.from_numpy(Y_numpy.astype(np.float32))
Y = Y.view(Y.shape[0],1)
n_samples, n_features = X.shape
learnrate = 0.01       # Learning  rate  
n = 1000        # Number of iterations

# Model prediction (H)
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

"""class linear_regression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linear_regression, self).__init__()
        # Define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = linear_regression(input_size,output_size)"""
# Loss
loss = nn.MSELoss()

# Training
optimizer = tch.optim.SGD(model.parameters(), lr=learnrate)
for epoch in range(n):
    # Prediction
    H = model(X)
    # Loss
    L = loss(Y,H)
    # Gradient
    L.backward()
    # Update w
    optimizer.step()
    # Zero geadients
    optimizer.zero_grad()
    # Print epoch
    if epoch % 100 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch + 1}: w = {w[0][0]:.3f}, loss = {L:.5f}')

predicted = model(X).detach().numpy()

# Visualisation
plt.plot(X_numpy,Y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()

# Check model
# print(f'Prediction autotraining: h(5) = {model(x_test).item():.3f}')