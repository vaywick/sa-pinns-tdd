# SA-PINNs-TDD

Self-adaptive physics-informed neural networks (SA-PINNs) [Journal of Computational Physics, 474:111722, 2023] have demonstrated superior accuracy over vanilla PINNs in solving partial differential equations with steep gradients by assigning higher weights to critical regions.
However, as the spatio-temporal domain extends, SA-PINNs often struggle to maintain high solution accuracy.
To overcome this limitation, we propose SA-PINNs with time-domain decomposition (SA-PINNs-TDD), which enable accurate and stable training across extended temporal domains, as demonstrated on the nonlinear Schr\"{o}dinger equation.
For the Allen-Cahn equation, SA-PINNs-TDD achieves 56 times greater accuracy than standard SA-PINNs, owing to its refined distribution of adaptive weights.
The model also proves effective for long-time simulations of the 2D heat conduction (HC) equation.
SA-PINNs-TDD adopt a sequential domain-wise training strategy, where thorough pre-training of the initial subdomain is essential to establish accurate pseudo initial conditions and reduce error accumulation in subsequent domains.
Pre-training the initial domain requires more epochs to achieve convergence, while subsequent domains utilize transferred knowledge for accelerated optimization.
Solving the NLS equation by SA-PINNs-TDD is crucial for advancing the understanding of rogue waves in ocean and optical engineering, enabling accurate prediction and control of these extreme events.
High-accuracy solutions of the HC equation serve as a foundational tool for thermal transport modeling, enabling precise heat transfer control when integrated with our proposed model.
Furthermore, SA-PINNs-TDD using 1 partition successfully resolves the dynamics of both Euler-Bernoulli and Timoshenko beams, overcoming the limitations reported in previous SA-PINNs [Engineering Applications of Artificial Intelligence 133 (2024) 108085].
Compared to their proposed causal PINNs with transfer learning, our model achieves an order-of-magnitude improvement in accuracy.
It addresses critical challenges in structural engineering and supports advanced methodologies for structural design, optimization, and control.

# Comparison

| Model             | RL2                  |
|-------------------|----------------------|
| bc-PINN       | 1.68e-2              |
| original SA-PINNs | (2.10±1.21)e-2       |
| PT-PINN      | (9.7±0.4)e-3         |
| SA-PINNs-TDD      | **(3.76±1.01)e-4**   |



# Framework

SA-PINNs-TDD are implemented using either the TensorFlow or PyTorch frameworks.
The Euler–Bernoulli-new and Allen–Cahn (AC) simulations are conducted in TensorFlow, while the remaining models are implemented in PyTorch.

PyTorch version 2.0.0 or higher

tensorflow version = 2.3.0 and keras version = 2.4.3
