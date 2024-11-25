# Adaptive Sigmoid: A Novel Activation Function with Controlled Saturation

<br>

## Introduction

Artificial Neural Networks (ANNs) rely heavily on activation functions to introduce non-linearity and enable learning of complex data patterns. Traditional activation functions, such as sigmoid and hyperbolic tangent, often face challenges like vanishing gradients and uncontrolled saturation. To address these issues, **Adaptive Sigmoid** introduces a parameterized activation function with controlled saturation, enabling more robust and efficient training of deep learning models. Zhang and others published [A Lightweight System for High-efficiency Edge-Cloud Collaborative Applications](https://www.researchsquare.com/article/rs-1868043/v2) article using that function with **Parametric Activation** name.

<br>

## Motivation

Deep learning architectures often struggle with:
- **Vanishing Gradients**: Gradients diminish as network depth increases, slowing training.
- **Quick Saturation**: Loss of information due to abrupt transitions in activation.

**Adaptive Sigmoid** resolves these challenges by introducing a parameter `α` to control the rate of saturation, balancing non-linearity and stability for better model performance.

<br>

## Methodology

The Adaptive Sigmoid activation function is defined as:

$$ \sigma(x) = \frac{1}{1 + e^{-\alpha x}} $$

Where:
- $x$: Input to the activation function.
- $\alpha$: Parameter controlling the rate of saturation.

The derivative with respect to \( x \) ensures efficient gradient computation during backpropagation:

$$ \frac{d\sigma}{dx} = \frac{\alpha e^{-\alpha x}}{(1 + e^{-\alpha x})^2} $$

<img src="https://github.com/doguilmak/Adaptive-Sigmoid/blob/main/assets/3d.png" alt="Adaptive Sigmoid">

<br>

## Features

- **Controlled Saturation**: Fine-tune the activation using \( \alpha \).
- **Improved Stability**: Smoother convergence curves and reduced vanishing gradient issues.
- **Flexible Training**: Balances non-linearity and stability for various datasets and architectures.
- **Compatibility**: Works seamlessly with PyTorch and TensorFlow.

<br>

## Results

Experiments demonstrate that **Adaptive Sigmoid**:
- Achieves competitive performance compared to ReLU, sigmoid, and tanh.
- Provides smoother convergence during training.
- Reduces vanishing gradients, improving training efficiency and generalization.

<br>

## Usage

PyTorch Implementation


```python
from adaptive_sigmoid import AdaptiveSigmoidLayer
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(input_dim, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    AdaptiveSigmoidLayer(alpha=1.0)
)
```

<br>

## Contributions

Contributions are welcome! Please correct me If I am wrong and feel free to submit a pull request or open an issue for feedback or improvements.

<br>

## Reference

Zhang, Z., Ma, W., Li, H., Tang, H., Yuan, X., Hao, Y., ... & Zhou, Z. (2022). A Lightweight System for High-efficiency Edge-Cloud Collaborative Applications.
