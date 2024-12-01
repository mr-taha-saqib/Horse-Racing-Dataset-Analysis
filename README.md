# Perceptron Logic Gates and Horse Racing Dataset Analysis

This project involves two tasks: implementing OR and AND gates using perceptron learning and analyzing the Horse Racing Dataset, including exploratory data analysis and training a classifier.

---

## **Task 1: Implementing OR and AND Gates using Perceptron Learning**

### **1. Perceptron Implementation**

The perceptron algorithm is used to model OR and AND gates. 

#### **OR Gate Implementation**
The truth table for OR gate is:
| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 1      |

#### **AND Gate Implementation**
The truth table for AND gate is:
| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 0       | 1       | 0      |
| 1       | 0       | 0      |
| 1       | 1       | 1      |

### **Code**
```python
import numpy as np

def perceptron_learning(inputs, outputs, epochs=10, learning_rate=0.1):
    weights = np.zeros(inputs.shape[1])
    bias = 0

    for epoch in range(epochs):
        for x, y in zip(inputs, outputs):
            linear_output = np.dot(x, weights) + bias
            prediction = 1 if linear_output >= 0 else 0
            error = y - prediction
            weights += learning_rate * error * x
            bias += learning_rate * error

    return weights, bias

# OR Gate
inputs_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs_or = np.array([0, 1, 1, 1])
weights_or, bias_or = perceptron_learning(inputs_or, outputs_or)
print(f"OR Gate Weights: {weights_or}, Bias: {bias_or}")

# AND Gate
inputs_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs_and = np.array([0, 0, 0, 1])
weights_and, bias_and = perceptron_learning(inputs_and, outputs_and)
print(f"AND Gate Weights: {weights_and}, Bias: {bias_and}")
