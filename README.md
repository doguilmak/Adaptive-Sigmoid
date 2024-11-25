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

The **Adaptive Sigmoid** activation function is defined as:

$$ \sigma(x) = \frac{1}{1 + e^{-\alpha x}} $$

Where:
- $x$: Input to the activation function.
- $\alpha$: A parameter that controls the rate of saturation, providing flexibility in the function's behavior.

The derivative of the activation function with respect to $x$ is crucial for efficient gradient computation during backpropagation, enabling smooth and stable training:

$$ \frac{d\sigma}{dx} = \frac{\alpha e^{-\alpha x}}{(1 + e^{-\alpha x})^2} $$

<br>

This derivative ensures that the gradients are properly scaled and effective, allowing the network to learn efficiently, even in deep architectures. To further illustrate the behavior of the Adaptive Sigmoid, we include a 3D plot that visualizes how the activation function varies with different input values $x$ and the control parameter $\alpha$. The plot demonstrates the changes in saturation behavior as $\alpha$ is adjusted.

<img src="https://github.com/doguilmak/Adaptive-Sigmoid/blob/main/assets/3d.png" width=750 height=750 alt="Adaptive Sigmoid">

<i>3D plot of the AdaptiveSigmoid activation function, with input values ranging from $-5$ to $5$ and values ranging from $0.1$ to $10$. The plot shows how the activation function varies with different values of input ($x$) and alpha ($\alpha$).</i>


<br>

## Features

The **Adaptive Sigmoid** offers several key advantages for neural network training. By introducing the parameter $\alpha$, it allows for **controlled saturation**, enabling users to fine-tune the activation function for optimal performance. This flexibility contributes to **improved stability**, helping to achieve smoother convergence and reduce issues like vanishing gradients. The activation function strikes a balance between **non-linearity** and **stability**, making it suitable for a wide range of datasets and architectures. 

Additionally, the **Adaptive Sigmoid** is designed to be **compatible** with both PyTorch and TensorFlow, allowing for seamless integration into existing deep learning workflows.

<br>

## Results

Experiments show that the **Adaptive Sigmoid** activation function provides several advantages over traditional activation functions like ReLU, sigmoid, and tanh:

- **Competitive Performance**: The **Adaptive Sigmoid** achieves performance on par with, or better than, commonly used activation functions such as ReLU, sigmoid, and tanh, across various tasks and datasets. It maintains a high level of accuracy and efficiency in model training.
  
- **Smoother Convergence**: The introduction of the $\alpha$ parameter allows for more controlled saturation, leading to smoother and more stable convergence during training. This helps avoid issues like slow learning and oscillations that are often encountered with traditional activation functions, especially in deep networks.

- **Reduced Vanishing Gradients**: By adjusting the rate of saturation with $\alpha$, the **Adaptive Sigmoid** mitigates the vanishing gradient problem that is typically seen with sigmoid and tanh in deep networks. This results in better gradient propagation, improving training efficiency and accelerating the convergence process.

Overall, the **Adaptive Sigmoid** enhances model training by providing greater flexibility and stability, making it a promising alternative for deep learning models.

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
