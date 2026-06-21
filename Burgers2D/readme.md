2D Burgers’ equation is accurately solved by SA-PINNs-TDD using both uniform and nonuniform partitioning strategies, with the nonuniform partitioned training further enhancing the prediction accuracy.
However, the single-subdomain training approach—equivalent to the original SA-PINNs—achieves low accuracy.

Animations of the multi-scale phenomena are provided in the two mp4 files.

> [!Note]
> termination criteria tested for 2D Burgers' equation.

🔵 For the 2D Burgers’ equation, additional tests of the proposed termination criterion were also conducted for the L-BFGS optimizer. Unlike the Schrödinger case, both the required training epochs and the prediction accuracy decrease as the threshold value increases, as demonstrated in each log.out files and the results presented in the "png" and "pdf" figure files. Specifically, the folders "indicators-2", "indicators-2/test-0.05", and "indicators-2/test-0.08" correspond to threshold values of 0.02, 0.05, and 0.08, respectively, with the RL2E evolutions shown for both the $u(t,x,y)$ and $v(t,x,y)$ solutions.

> [!Note]
> The Necessity of Partition Training.

🔵 Comparison of the PDE residual evolution is shown in updated files "test-resi-0.03.pdf" and "test-resi-v-0.03.pdf" for the $u(t,x,y)$ and $v(t,x,y)$ equation between the four-subdomain case and the single-subdomain case: Two plots in panel (a) and (c) compare the PDE residuals obtained in the subdomains [0,1] and [1,2] with those obtained from the single-domain case [0,10] for $u(t,x,y)$ equation. Two plots in panel (b) and (d) compare the PDE residuals obtained in the subdomains [2,5] and [5,10] with those obtained from the single-domain case [0,10] for $v(t,x,y)$ equation. Significantly reduced PDE residuals are achieved using the partitioned approach.
