# import libaries
import torch as tch
import torch.nn as nn

# Initialisation (Model:'y = 2*x')
X = tch.tensor([[1], [2], [3], [4]], dtype = tch.float32)
Y = tch.tensor([[2], [4], [6], [8]], dtype = tch.float32)
n_samples, n_features = X.shape
x_test = tch.tensor([5],dtype=tch.float32)
lr = 0.01       # Learning  rate  
n = 1000        # Number of iterations

# Model prediction (H)
input_size = n_features
output_size = 1
# model = nn.Linear(input_size, output_size)

class linear_regression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linear_regression, self).__init__()
        # Define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = linear_regression(input_size,output_size)

# Training
optimizer = tch.optim.SGD(model.parameters(), lr=lr)
loss = nn.MSELoss()
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

# Check model
print(f'Prediction autotraining: h(5) = {model(x_test).item():.3f}')