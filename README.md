# 3-Layer-Neural-Network-from-Scratch-on-Iris-Dataset

This repository contains a modular, class-free Python implementation of a 3-layer feedforward artificial neural network trained from scratch using NumPy. The model is applied to classify species in the famous Iris dataset.


## 🚀 Project Overview
Instead of relying on deep learning libraries like TensorFlow or PyTorch, this project implements all core neural network algorithms from mathematical principles:
- **He Initialization** for weight initialization.
- **Sigmoid** activation function for hidden layers.
- **Softmax** activation function for the output multi-class prediction.
- **Categorical Cross-Entropy Loss** computation.
- **Backpropagation** using analytical gradients.
- **Mini-batch Gradient Descent** optimizer.
- **Early Stopping** based on validation accuracy to prevent overfitting.

---

## 📊 Dataset
The network is trained on the Iris Dataset loaded dynamically from a public Google Sheets CSV link. The features include:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

The target variables are mapped to numerical categories:
- `Iris-setosa` ➔ `0`
- `Iris-versicolor` ➔ `1`
- `Iris-virginica` ➔ `2`

---

## 🛠️ Architecture Details
- **Input Layer:** 4 Features
- **Hidden Layer 1:** 8 Neurons (Sigmoid)
- **Hidden Layer 2:** 6 Neurons (Sigmoid)
- **Output Layer:** 3 Neurons (Softmax for multi-class classification)

---

## ⚙️ How it Works (Core Functions)

### 1. Parameter Initialization
Using He (Kaiming) normal initialization to avoid vanishing/exploding gradients:
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

### 2. Forward Propagation
Linear combination followed by activations layer by layer:
- $Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}$
- $A^{[l]} = \sigma(Z^{[l]})$

### 3. Loss Calculation
Computes Categorical Cross-Entropy Loss:
$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{C} y_{ik} \log(\hat{y}_{ik})$$

### 4. Backpropagation & Parameter Updates
Gradients of the loss with respect to weights ($dW$) and biases ($db$) are computed recursively using the Chain Rule, and parameters are adjusted via:
$$W^{[l]} = W^{[l]} - \alpha \cdot dW^{[l]}$$
$$b^{[l]} = b^{[l]} - \alpha \cdot db^{[l]}$$
where $\alpha$ is the learning rate.

---

## 📈 Performance & Results

- **Training Settings:** Mini-batch size of 16, learning rate of 0.01, and patience of 50 epochs.
- **Test Accuracy achieved:** `~76.67%` (with early stopping saving the best validation parameters).

```text
--- Model Test Performance ---
Test Accuracy: 76.67%

Classification Report:
              precision    recall  f1-score   support

      Setosa       1.00      1.00      1.00        10
  Versicolor       1.00      0.30      0.46        10
   Virginica       0.59      1.00      0.74        10

💻 Setup and Usage
Clone the repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Install required dependencies:

pip install numpy pandas matplotlib seaborn scikit-learn
Run the script: Open the notebook in Google Colab or run your Python file:

python neural_network.py
