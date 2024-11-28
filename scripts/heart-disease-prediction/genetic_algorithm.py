import warnings
import pandas as pd
import mlrose_hiive
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import os

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Load dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "../../data/heart_statlog_cleveland_hungary_final.csv")
data = pd.read_csv(file_path).dropna()

# Prepare features and target
X = data.drop('target', axis=1)
y = data['target']

# Normalize features
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

# Genetic Algorithm Neural Network
ga_nn = mlrose_hiive.NeuralNetwork(
    hidden_nodes=[5],
    activation='sigmoid',
    algorithm='genetic_alg',
    max_iters=500,
    learning_rate=0.02,
    pop_size=100,
    mutation_prob=0.01,
    max_attempts=5,
    random_state=42,
    curve=True
)

# Train the model
ga_nn.fit(X_train, y_train)

# Evaluate
print(f"Genetic Algorithm - Classification Report: \n{classification_report(y_test, ga_nn.predict(X_test))}")

# Plot loss curve
plt.figure(figsize=(10, 6))
plt.title('Loss Curve - Neural Network Using Genetic Algorithm')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.plot(ga_nn.fitness_curve[:, 1], ga_nn.fitness_curve[:, 0], label="Loss(x) = 1/(1 + Fitness(X))")
plt.legend()
plt.grid(True)
plt.show()