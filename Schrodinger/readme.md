First-order rogue wave solution of the nonlinear Schrödinger equation is accurately solved using the SA-PINNs-TDD by uniform partitioning.
High-precision predictions are achieved with training over 5 or 7 subdomains, whereas the single-subdomain training approach-equivalent to the original SA-PINNs model-fails to attain comparable accuracy.

> [!Note]
> The Necessity of Partition Training.

🔵 In the present work, the convergence of the PDE residual is employed as an indicator to determine the appropriate number of subdomains through simulation tests, which has proven to be highly efficient, with the required number of subdomains typically not exceeding five. For example, as shown in File "sch12-22-j-3dom-1dom.pdf", SA-PINNs-TDD with only three uniform subdomains achieves substantially lower loss values (with the PDE residual constituting one of the loss terms) than the original SA-PINNs without domain decomposition when solving the Schrödinger equation. Similarly, employing five uniform subdomains also leads to significantly lower loss values, comparable to those observed in File "sch12-22-j-3dom-1dom.pdf", as illustrated in File "sch12-22-j-all-1st.pdf".
