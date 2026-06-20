> [!Note]
> termination criteria by monitoring the relative changes of loss and PDE residual.

🔵 To accelerate training with the L-BFGS optimizer, termination criteria are applied every 500 epochs by monitoring the relative differences in either the loss values ($\epsilon_J =| Loss_{k+500} - Loss_{k}|/ Loss_{k}$) or the PDE residuals ($\epsilon_f =| Residual_{k+500} - Residual_{k}|/ Residual_{k}$). The current folder uses $\epsilon_J= \epsilon_f = 0.08$, whereas the folder "test-0.05" employs $\epsilon_J= \epsilon_f = 0.05$. Both cases achieve similar prediction accuracy; however, the former converges and terminates more rapidly, as demonstrated by the three PDF figures provided in each folder.

🔵 To demonstrate the robustness of the proposed model, a series of tests with different numbers of residual points are provided in the folder "nr-test". The results show that the RL2E decreases as the number of residual points increases, while the model already achieves relatively high accuracy when $N_r= 10000$, as shown in the figure file "Nr-test.png" in the folder "nr-test". 
