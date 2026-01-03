# Micro-Grad Engine and Neural Network Implementation

## Overview
This project implements a **micro-grad automatic differentiation engine** from scratch and demonstrates its correctness by training a **basic neural network** using backpropagation and gradient descent.

The goal is to understand how gradients are computed dynamically using a computation graph, similar to modern deep-learning frameworks.

---

## Deliverables

### (a) Micro-Grad Engine
A custom automatic differentiation engine was implemented using a `Value` class that stores:
- scalar data
- gradient value

Supported operations:
- Addition (`+`)
- Multiplication (`*`)
- Power (`**`)
- Non-linear activation (`tanh')

Reverse-mode automatic differentiation is implemented using:
- computation graph tracking
- topological sorting
- chain rule–based backpropagation

---

### (b) Neural Network Demonstration
A neural network was built using the micro-grad engine, consisting of:
- `Neuron`
- `Layer`
- `MLP (Multi-Layer Perceptron)`

The network was trained on a small dataset using:
- Mean Squared Error (MSE) loss
- Gradient Descent optimization

Training results show:
- decreasing loss
- increasing accuracy (up to 100%)

This confirms the correctness of gradient computation and backpropagation.

---

## Project Structure



