# SA-PINNs-TDD

Self-adaptive PINNs (SA-PINNs) [Journal of Computational Physics, 474:111722, 2023] have demonstrated superior accuracy over vanilla PINNs in solving PDEs with steep gradients by assigning higher weights to critical regions.
However, as the spatio-temporal domain extends, SA-PINNs often struggle to maintain high solution accuracy.
To overcome this limitation, we propose SA-PINNs with time-domain decomposition (SA-PINNs-TDD), which enable accurate and stable training across extended temporal domains, as demonstrated on the nonlinear Schr\"{o}dinger equation.
For the Allen-Cahn equation, SA-PINNs-TDD achieves a 1-2 order-of-magnitude improvement in accuracy compared to standard SA-PINNs, attributed to its refined distribution of adaptive weights.
The model also proves effective for long-time simulations of the 2D heat conduction equation.
SA-PINNs-TDD adopt a sequential domain-wise training strategy, where thorough pre-training of the initial subdomain is essential to establish accurate pseudo initial conditions and reduce error accumulation in subsequent domains.
Pre-training the initial domain requires more epochs to achieve convergence, while subsequent domains utilize transferred knowledge for accelerated optimization.
Additionally, we address previously reported limitations, wherein SA-PINNs failed to solve the Euler-Bernoulli and Timoshenko beam dynamics on elastic foundations.
In contrast, our developed SA-PINNs successfully solve both problems and achieve over an order-of-magnitude improvement in accuracy compared to causal PINNs with transfer learning [Engineering Applications of Artificial Intelligence 133 (2024) 108085].

# Framework

SA-PINNs-TDD are implemented using either the TensorFlow or PyTorch frameworks.
The Euler–Bernoulli-new and Allen–Cahn (AC) simulations are conducted in TensorFlow, while the remaining models are implemented in PyTorch.

PyTorch version 2.0.0 or higher

tensorflow version = 2.3.0 and keras version = 2.4.3
