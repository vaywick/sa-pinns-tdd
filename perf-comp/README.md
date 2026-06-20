<font color="blue"> The proposed model, implemented in PyTorch, offers greater memory efficiency and faster training than TensorFlow when solving 2D PDEs, as demonstrated below.
Figure file "comp-tf-pt.png" presents a comparison between the TF and PT frameworks in terms of training speed, memory usage, and power consumption for solving the high-order BS equation.
In this figure, “s/epoch” represents per epoch in seconds and “W” denotes watts. “4-64” represents a neural network with 4 hidden layers and 64 neurons in each layer, and the same notation applies to the other cases.
The number of collocation points is consistently set as 20000 across all cases, except for two entries where 10000 is used.
As can be seen, compared with the TF architecture, the PT architecture offers several advantages: the training time with the Adam optimizer is reduced by approximately 50% for the first three cases.
For the “7-48” neural network architecture with 10000, the PT framework reduces the training time by 71% and 25% compared with the TF framework when using the two optimizers, respectively.
Compared with the TF framework, the PT framework achieves significantly lower GPU memory consumption across all test cases.
In particular, for the architecture with 20000, training under the TF framework fails due to GPU out-of-memory limitations. New in Version 2.0</font>

Software setup:

1. The PyTorch version is deployed across diverse computational environments within the SA-PINNs framework:

PyTorch version 2.0.0 or higher

CUDA versions 11 or 12

2. TensorFlow version is configured to use specific software versions due to the L-BFGS algorithm implementation in the original SA-PINNs:

tensorflow version = 2.3.0

keras version = 2.4.3
