import numpy as np
import pandas as pd
import pyswarms as ps
import os

# Load dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "../../data/heart_statlog_cleveland_hungary_final.csv")
data = pd.read_csv(file_path)

# Store the features as X and the labels as y
X = data.drop(columns=['target']).values
y = data['target'].values

# Define neural network structure
n_inputs = X.shape[1]  # Number of input features
n_hidden = 1  # Number of hidden neurons
n_classes = 2  # Number of output classes
num_samples = len(data)  # Number of samples


# Define the logits function
def logits_function(p):
    """Calculate and perform forward propagation."""
    # Reshape weights and biases
    W1 = p[0: n_inputs * n_hidden].reshape((n_inputs, n_hidden))
    b1 = p[n_inputs * n_hidden: n_inputs * n_hidden + n_hidden].reshape((n_hidden,))
    W2 = p[n_inputs * n_hidden + n_hidden: n_inputs * n_hidden + n_hidden + n_hidden * n_classes].reshape(
        (n_hidden, n_classes))
    b2 = p[
         n_inputs * n_hidden + n_hidden + n_hidden * n_classes: n_inputs * n_hidden + n_hidden + n_hidden * n_classes + n_classes].reshape(
        (n_classes,))

    # Perform forward propagation
    z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
    a1 = np.tanh(z1)  # Activation in Layer 1
    logits = a1.dot(W2) + b2  # Pre-activation in Layer 2
    return logits


# Define the forward propagation function
def forward_prop(params):
    """Compute forward propagation and calculate the loss."""
    logits = logits_function(params)
    exp_scores = np.exp(logits)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[range(num_samples), y])
    loss = np.sum(correct_logprobs) / num_samples
    return loss


# Define the fitness function
def f(x):
    """Compute forward propagation loss for the whole swarm."""
    n_particles = x.shape[0]
    return np.array([forward_prop(x[i]) for i in range(n_particles)])


# Initialize swarm
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
dimensions = (n_inputs * n_hidden) + (
            n_hidden * n_classes) + n_hidden + n_classes  # Total dimensions of the parameter space
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(f, iters=1000)


# Prediction function
def predict(pos):
    """Use trained weights to make predictions."""
    logits = logits_function(pos)
    return np.argmax(logits, axis=1)


# Evaluate accuracy
accuracy = (predict(pos) == y).mean()
print(f"Accuracy: {accuracy:.2f}")