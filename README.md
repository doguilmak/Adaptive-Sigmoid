# An Activation Function with Controlled Saturation

<img src="https://github.com/doguilmak/Adaptive-Sigmoid/blob/main/assets/bg.png" width=900 alt="Banner">

<br>

## Introduction

Artificial Neural Networks (ANNs) rely heavily on activation functions to introduce non-linearity and enable learning of complex data patterns. Traditional activation functions, such as sigmoid and hyperbolic tangent, often face challenges like vanishing gradients and uncontrolled saturation. To address these issues, **Adaptive Sigmoid** introduces a parameterized activation function with controlled saturation, enabling more robust and efficient training of deep learning models. Zhang and others published [A Lightweight System for High-efficiency Edge-Cloud Collaborative Applications](https://www.researchsquare.com/article/rs-1868043/v2) article using that function with **Parametric Activation** name.

<br>

## Motivation

Traditional activation functions like sigmoid, tanh, and ReLU are widely used in deep learning, but they each come with limitations. Sigmoid and tanh suffer from **vanishing gradients** and **quick saturation**, making it difficult for deep networks to learn efficiently. ReLU, while avoiding vanishing gradients, has issues like the **dying ReLU problem**, where neurons can become inactive.


The **Adaptive Sigmoid** was designed to provide more flexibility in handling saturation behavior. By introducing a parameter $\alpha$, the function allows for finer control over the rate of saturation, which can help stabilize training, particularly in deep networks. This additional control is intended to offer a new perspective on managing non-linearity in neural networks, without claiming a groundbreaking scientific advancement. By adjusting $\alpha$, users can explore different saturation behaviors, potentially improving training stability and convergence without altering the fundamental architecture of the network.

<br>

## Methodology

The Adaptive Sigmoid activation function is defined as:

$$ \sigma(x) = \frac{1}{1 + e^{-\alpha x}} $$

Where:
- $x$: Input to the activation function.
- $\alpha$: Parameter controlling the rate of saturation.

The derivative with respect to \( x \) ensures efficient gradient computation during backpropagation:

$$ \frac{d\sigma}{dx} = \frac{\alpha e^{-\alpha x}}{(1 + e^{-\alpha x})^2} $$

<img src="https://github.com/doguilmak/Adaptive-Sigmoid/blob/main/assets/3d.png" width=900 alt="Adaptive Sigmoid">

<br>

## Features

The **Adaptive Sigmoid** offers several key advantages for neural network training. By introducing the parameter $\alpha$, it allows for **controlled saturation**, enabling users to fine-tune the activation function for optimal performance. This flexibility contributes to **improved stability**, helping to achieve smoother convergence and reduce issues like vanishing gradients. The activation function strikes a balance between **non-linearity** and **stability**, making it suitable for a wide range of datasets and architectures. 

Additionally, the **Adaptive Sigmoid** is designed to be **compatible** with both PyTorch and TensorFlow, allowing for seamless integration into existing deep learning workflows.

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
