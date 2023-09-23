# import libaries
import torch as tch
import torch.nn as nn

# Initialisation (Model:'y = 2*x')
X = tch.tensor([1, 2, 3, 4], dtype = tch.float32)
Y = tch.tensor([2, 4, 6, 8], dtype = tch.float32)
omega = tch.tensor(0.0, dtype=tch.float32, requires_grad=True)

lr = 0.01       # Learning  rate  
n = 1000        # Number of iterations

# Model prediction (H)
def forward(omega, X):
    return omega * X

# Training
optimizer = tch.optim.SGD([omega], lr=lr)
loss = nn.MSELoss()
for epoch in range(n):
    # Prediction
    H = forward(omega, X)
    # Loss
    L = loss(Y,H)
    # Gradient
    L.backward()
    # Update omega
    optimizer.step()
    # Zero geadients
    optimizer.zero_grad()
    # Print epoch
    if epoch % 100 == 0:
        print(f'epoch {epoch + 1}: omega = {omega:.3f}, loss = {L:.5f}')

# Check model
print(f'Prediction autotraining: h(5) = {forward(omega,5):.3f}')