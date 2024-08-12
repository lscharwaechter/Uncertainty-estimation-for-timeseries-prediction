import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define an LSTM-baased model for time series forecasting
class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMmodel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

# This function creates a dataset given a signal observation
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# This fuction uses the Generalized Gauss-Newton method to approximate the 
# Hessian of the loss with respect to the weights/parameters
def compute_hessian_approximation(model, criterion, x, y, alpha=1e-3):
    model.eval()
    model.zero_grad()
    
    # Forward pass to compute predictions
    outputs = model(x)
    
    # Compute the loss
    loss = criterion(outputs, y)
    
    # Compute gradients w.r.t. model parameters
    loss.backward(retain_graph=True)
    
    # Get the Jacobian of the outputs w.r.t. the parameters
    jacobian = []
    for param in model.parameters():
        if param.requires_grad:
            jacobian.append(param.grad.view(-1))
    
    # Concatenate gradients into a single matrix, Shape of J: [1, D]
    J = torch.cat(jacobian).unsqueeze(0)  
    
    # Compute GG^T
    GG_T = torch.matmul(J.T, J)
    
    # Regularize the Hessian approximation with a small const
    D = GG_T.size(0)
    hessian_approx = GG_T + alpha * torch.eye(D, device=GG_T.device)
    
    return hessian_approx

np.random.seed(42)
torch.manual_seed(42)

# Hyperparameters
INPUT_SIZE = 1
HIDDEN_SIZE = 64
NUM_LAYERS = 1
OUTPUT_SIZE = 1
NUM_EPOCHS = 1000
LEARNING_RATE = 0.001

# Create a periodic time series signal
T1 = 5
T2 = 2
A1 = 0.6
A2 = 0.4
n_observations = 100
t = np.arange(0, n_observations, 1)
signal = A1*np.sin(t/T1)+A2*np.sin(t/T2) 
data = signal + np.random.normal(0, 0.05, size=signal.shape) # Add noise
seq_length = 30 # sequences for training

# Create a dataset from the signal observation
x, y = create_sequences(data, seq_length)
x = torch.from_numpy(x).float().unsqueeze(-1) # Add feature dimension
y = torch.from_numpy(y).float().unsqueeze(-1) # Add feature dimension

# Initialize the model, loss function, and optimizer
model = LSTMmodel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the model
for epoch in range(NUM_EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

# Compute the GGN approximation
print("Compute Hessian approximation...")
hessian_approx = compute_hessian_approximation(model, criterion, x, y, alpha=1e-3)

# Eigenvalues for uncertainty analysis
print("Compute eigenvalues...")
eigenvalues, eigenvectors = torch.linalg.eig(hessian_approx)

# Save the matrices
np.save('uncertainty_hessian.npy',hessian_approx)
np.save('uncertainty_eigenvalues.npy',eigenvalues)
#hessian_approx = np.load('uncertainty_hessian.npy')
#eigenvalues = np.load('uncertainty_eigenvalues.npy')

# The inverse of the eigenvalues can be translated into a sense of uncertainty
small_const = 1e-6
uncertainty = 1.0 / (eigenvalues + small_const)
#uncertainty = uncertainty.detach().numpy()

# Create a figure
y_hat = model(x).squeeze().detach().numpy()
plt.figure(figsize=(14, 7))

# Plot the original signal and training predictions
plt.plot(t, data, label='Original Signal', linewidth=2)
plt.plot(t[seq_length:], y_hat, label='Training Predictions', linewidth=2)

# Calculate uncertainty bounds
mean_uncertainty = np.mean(uncertainty)
std_uncertainty = np.std(uncertainty)

# Plot uncertainty bounds
plt.fill_between(t[seq_length:], y_hat - std_uncertainty, y_hat + std_uncertainty, color='gray', alpha=0.5, label='Uncertainty bounds')

# Add a vertical line
plt.axvline(x=t[seq_length], color='k', linestyle='--', linewidth=2)

# Axes descriptions
plt.legend(loc='upper left', fontsize=14)
plt.xlabel('Timesteps', fontsize=20)
plt.ylabel('Signal value', fontsize=20)
plt.title('Time Series Forecasting with Uncertainty', fontsize=20)
plt.xlim([0, n_observations]) 
plt.tick_params(axis='both', which='major', labelsize=18)

spines = plt.gca().spines
for spine in spines.values():
    spine.set_linewidth(2)
plt.show()

print("Done.")
