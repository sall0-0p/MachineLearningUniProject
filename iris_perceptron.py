# Orest Antoniuk, 232939
# Sorry, I hate Jupyter Notebook

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# configuration and creation of initial files
OUTPUT_DIR = "out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, "iris_training_log.txt")

def log(message):
    print(message)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")

if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000, decay_rate=0.0001, random_seed=None):
        self.lr = learning_rate
        self.epochs = epochs
        self.decay = decay_rate
        self.weights = None
        self.bias = None
        self.loss_history = []
        if random_seed:
            np.random.seed(random_seed)
    
    def _net_input(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initialising weights
        self.weights = np.random.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias = 0

        # transofrming labels
        y_encoded = np.where(y <= 0, -1, 1)

        for epoch in range(self.epochs):
            errors = 0
            current_lr = self.lr / (1 + self.decay * epoch)

            for idx, x_i in enumerate(X):
                linear_output = self._net_input(x_i)
                prediction = np.where(linear_output >= 0, 1, -1)
 
                if prediction != y_encoded[idx]:
                    update = current_lr * (y_encoded[idx] - prediction)
                    self.weights += update * x_i
                    self.bias += update
                    errors += 1
            
            self.loss_history.append(errors)
            if errors == 0:
                log(f"Converged early at epoch {epoch + 1}")
                break
        return self
    
    def predict(self, X):
        linear_output = self._net_input(X)
        return np.where(linear_output >= 0, 1, 0)
    
    def evaluate(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

def visualize_results(X, y, model, filename):
    plt.figure(figsize=(10, 6))

    # meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[y==0, 0], X[y==0, 1], color='blue', marker='x', label='Setosa')
    plt.scatter(X[y==1, 0], X[y==1, 1], color='red', marker='o', label='Non-Setosa')
    
    plt.title(f'Perceptron Decision Boundary (Fricking magic ✨)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    log(f"Saved plot to {save_path}")

def main():
    log("--- Starting Iris Experiment ---")

    # loading data
    iris = load_iris()
    X = iris.data[:, [2, 3]]
    y = (iris.target != 0).astype(int)

    # splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # training model
    perceptron = Perceptron(learning_rate=0.05, epochs=50, decay_rate=0.001, random_seed=1)
    perceptron.fit(X_train, y_train)

    # evaluating
    acc_train = perceptron.evaluate(X_train, y_train)
    acc_test = perceptron.evaluate(X_test, y_test)

    log(f"Final Weights: {perceptron.weights}")
    log(f"Final Bias: {perceptron.bias:.4f}")
    log(f"Train Accuracy: {acc_train*100:.2f}%")
    log(f"Test Accuracy:  {acc_test*100:.2f}%")

    # saving learning curve
    plt.plot(range(1, len(perceptron.loss_history) + 1), perceptron.loss_history, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Misclassifications')
    plt.title('Training Convergence')
    plt.savefig(os.path.join(OUTPUT_DIR, "iris_learning_curve.png"))
    plt.close()

    # saving decision boundary
    visualize_results(X_test, y_test, perceptron, "iris_decision_boundary.png")

if __name__ == "__main__":
    main()


