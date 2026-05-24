# Deep Learning PJ1

Name: 杜景琦  
Student ID: 23307130163  
Date: 2026-05-24

## 1. Introduction

This project studies handwritten digit classification on the MNIST dataset using only the provided NumPy starter code. According to the project requirements, the work contains three parts: an MLP baseline, a manually implemented CNN, and two additional focused directions. The two additional directions chosen in this report are Direction 1: Optimization and Direction 5: Error Analysis and Visualization.

The goal is not to chase the best possible accuracy, but to correctly implement the basic components of neural networks and make clear comparisons. Therefore, the experiments use a small number of focused settings: first a baseline MLP, then a simple CNN under similar training conditions, and finally a MomentumGD comparison plus visual analysis of model behavior.

## 2. Implementation

### 2.1 Linear Layer

The linear layer is implemented in `codes/mynn/op.py`. For a batch input `X`, the forward pass is:

`Y = XW + b`

where `W` is the weight matrix and `b` is the bias. During backward propagation, if the upstream gradient is `dY`, the gradients are:

`dW = X^T dY`, `db = sum(dY)`, `dX = dY W^T`.

This layer is the basic building block of the MLP. Since the MLP receives flattened MNIST images, the first linear layer maps the 784-dimensional pixel vector into a hidden representation, and the final linear layer maps hidden features into 10 class logits. The input of each forward pass is cached so that the matrix derivatives can be computed during the backward pass.

### 2.2 Softmax Cross Entropy

The loss is implemented in `codes/mynn/op.py` using numerically stable softmax. The forward pass computes the mean cross-entropy loss, and the backward gradient is:

`(softmax(logits) - one_hot(labels)) / batch_size`.

The softmax operation converts raw logits into class probabilities. To avoid numerical overflow, the maximum logit of each sample is subtracted before exponentiation. Combining softmax and cross-entropy gives a simple and stable gradient form, which is useful for implementing backpropagation manually.

### 2.3 Conv2D and CNN Architecture

The 2D convolution layer is implemented manually in `codes/mynn/op.py`, including padding, stride, forward pass, and backward pass. The CNN model is implemented in `codes/mynn/models.py`.

Compared with the MLP, the CNN keeps the 2D spatial arrangement of the image and uses shared local kernels. This gives the model a natural inductive bias for image data, because nearby pixels are processed together and the same feature detector can be reused across positions. That is why the CNN is usually more suitable for MNIST than a fully connected network of similar size.

CNN architecture:

| Layer | Setting | Output Shape |
|---|---|---|
| Input | MNIST grayscale image | 1×28×28 |
| Conv1 + ReLU | 1→8, kernel 3, stride 2, padding 1 | 8×14×14 |
| Conv2 + ReLU | 8→16, kernel 3, stride 2, padding 1 | 16×7×7 |
| Flatten | - | 784 |
| Linear | 784→10 | 10 |

The CNN architecture is intentionally small so that it can be trained and debugged quickly on CPU. Even with only two convolution layers, it can already capture local stroke patterns and edge combinations better than a plain MLP, which is enough to demonstrate the benefit of convolution on image classification.

## 3. MLP Baseline

Training setting:

| Item | Value |
|---|---|
| Model | MLP `[784, 600, 10]` |
| Activation | ReLU |
| Optimizer | SGD |
| Learning rate | 0.06 |
| Scheduler | MultiStepLR |
| Batch size | 32 |
| Epochs | 1 for the recorded quick experiment |

The MLP is used as the required baseline. It treats each 28×28 image as a 784-dimensional vector, so it does not explicitly use the spatial relationship between neighboring pixels. Nevertheless, it is simple, fast, and useful as a reference point for judging whether the CNN and optimization modifications are actually helpful.

Results:

| Model | Optimizer | Scheduler | Val Acc | Test Acc |
|---|---|---|---|---|
| MLP | SGD | MultiStepLR | 0.9085 | 0.9122 |

Learning curve:

![MLP SGD learning curve](PJ1_work/codes/figs/mlp_sgd_multistep_curve.png)

The MLP reaches 0.9122 test accuracy in the recorded quick run. This shows that the implemented linear layer, softmax cross-entropy loss, optimizer, and backpropagation pipeline are functioning correctly. The result is lower than a fully tuned run, but it is sufficient as a baseline for the subsequent CNN and optimization comparisons.

## 4. CNN Model and MLP-vs-CNN Comparison

Results:

| Model | Optimizer | Scheduler | Val Acc | Test Acc |
|---|---|---|---|---|
| MLP | SGD | MultiStepLR | 0.9085 | 0.9122 |
| CNN | SGD | MultiStepLR | 0.9369 | 0.9438 |

The CNN improves validation accuracy from 0.9085 to 0.9369 and test accuracy from 0.9122 to 0.9438 under the same short-run baseline setting. This indicates that the convolutional structure is useful even when the model and training budget are small.

The reason is that convolution helps the network detect local strokes and shapes without learning every spatial pattern independently. The MLP flattens the image and therefore loses explicit 2D topology, while the CNN preserves local neighborhoods and shares kernels across locations. This makes CNN a more natural model for MNIST image classification.

![CNN SGD learning curve](PJ1_work/codes/figs/cnn_sgd_multistep_curve.png)

## 5. Direction 1: Optimization

This section compares standard SGD with Momentum Gradient Descent under similar training settings. `MomentGD`, `MultiStepLR`, and `ExponentialLR` are implemented in `codes/mynn/optimizer.py` and `codes/mynn/lr_scheduler.py`.

Momentum update rule:

`v_t = mu * v_{t-1} + lr * grad`, `theta_t = theta_t - v_t`

Results:

| Model | Optimizer | Scheduler | Val Acc | Test Acc |
|---|---|---|---|---|
| MLP | SGD | MultiStepLR | 0.9085 | 0.9122 |
| MLP | MomentumGD | MultiStepLR | 0.9701 | 0.9711 |
| CNN | SGD | MultiStepLR | 0.9369 | 0.9438 |
| CNN | MomentumGD | MultiStepLR | 0.9614 | 0.9646 |

MomentumGD improves convergence and final accuracy for both MLP and CNN in these runs. For the MLP, the test accuracy increases from 0.9122 to 0.9711. For the CNN, the test accuracy increases from 0.9438 to 0.9646. This result is consistent with the intuition that momentum reduces noisy updates and helps the optimizer keep moving in a useful direction when the loss surface is steep or oscillatory.

![MLP Momentum learning curve](PJ1_work/codes/figs/mlp_momentum_multistep_curve.png)

![CNN Momentum learning curve](PJ1_work/codes/figs/cnn_momentum_multistep_curve.png)

## 6. Direction 5: Error Analysis and Visualization

This direction analyzes the trained CNN through three visualizations: confusion matrix, misclassified examples, and first-layer convolution kernels. The corresponding script is `codes/weight_visualization.py`, and all generated images are saved under `codes/figs/`.

### 6.1 Confusion Matrix

The confusion matrix helps identify which labels are still easy to confuse. In MNIST, common confusions usually happen between digits with similar stroke structures, such as 3/5, 4/9, or 5/8. This type of analysis is useful because a single accuracy number cannot explain where the model fails.

![CNN confusion matrix](PJ1_work/codes/figs/confusion_matrix_cnn.png)

### 6.2 Misclassified Examples

The misclassified examples show concrete failure cases. These samples are often ambiguous, tilted, incomplete, or written in an unusual style. Looking at them helps check whether the errors are caused by implementation problems or by genuinely difficult handwritten samples.

![Misclassified examples](PJ1_work/codes/figs/misclassified_examples_cnn.png)

### 6.3 Convolution Kernels

The first-layer convolution kernels provide a direct view of what low-level filters the CNN has learned. Although the kernels are small, they can still act as local detectors for edges, strokes, or intensity patterns. This supports the explanation that CNNs learn local visual structures instead of treating every pixel independently.

![CNN kernels](PJ1_work/codes/figs/cnn_kernels.png)

## 7. Main Results Table

| Experiment | Model | Optimizer | Scheduler | Val Acc | Test Acc | Notes |
|---|---|---|---|---|---|---|
| Baseline | MLP | SGD | MultiStepLR | 0.9085 | 0.9122 | Part A |
| Baseline | CNN | SGD | MultiStepLR | 0.9369 | 0.9438 | Part B |
| Optimization | MLP | MomentumGD | MultiStepLR | 0.9701 | 0.9711 | Direction 1 |
| Optimization | CNN | MomentumGD | MultiStepLR | 0.9614 | 0.9646 | Direction 1 |

## 8. Discussion and Conclusion

Overall, the experiments show three main points. First, the baseline MLP already reaches reasonable performance, which confirms that the manually implemented linear layer and loss are correct. Second, the CNN achieves better results than the MLP in the baseline setting, which supports the idea that convolution is more suitable for image data. Third, MomentumGD gives a clear boost over plain SGD in these runs, which suggests that optimization dynamics matter a lot even for a relatively small model on MNIST.

Among the two additional directions, optimization gives the most direct quantitative improvement. Error analysis and visualization are also useful because they explain model behavior beyond accuracy. The confusion matrix and misclassified examples show what kinds of samples remain difficult, while the kernel visualization shows that the CNN is learning local image filters.

One limitation is that the recorded experiments use a short one-epoch setting to finish quickly. With more training time and more careful hyperparameter tuning, the absolute accuracy could likely be improved. However, the main conclusions are already clear: the CNN is more suitable than the MLP for MNIST images, and MomentumGD improves the training process under the tested settings.

## Appendix: Code and Model Links

Github code link:  
Model/checkpoint link: the same with the github link

The final checkpoint package is `pj1_weights.zip`, which contains the four checkpoints used in the report:

- `mlp_sgd_multistep.pickle`
- `cnn_sgd_multistep.pickle`
- `mlp_momentum_multistep.pickle`
- `cnn_momentum_multistep.pickle`

The submitted GitHub repository should exclude dataset files, generated checkpoints, `__pycache__`, and other large temporary files. The model/checkpoint link should point to the uploaded `pj1_weights.zip` file on ModelScope or another accessible platform.
