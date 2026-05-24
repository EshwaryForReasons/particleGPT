## Info

I have decided to summarize the main findings of the Kaplan et. all "Scaling Laws for Neural Language Models" paper. Hopefully, this makes it easier to compare our scaling laws studies to Kaplan et. all's findings.

## Summary

### Avoid overfitting
To avoid overfitting the dataset size $D$ should be 
$$
D \gtrsim 5 \times 10^{3} N^{0.74}
$$
Maintaining this avoids overfitting below roughly their random-seed loss variation threshold of about 0.02.

### Model-size scaling
For models trained close to convergence on enough data, Kaplan et. all sees
$$
L(N) = \left( \frac{N_{c}}{N} \right)^{\alpha_N}
$$
with roughly, $\alpha_{N} \approx 0.076$ and $N_{c} \approx 8.8 \times 10^{13}$. This implies increasing model size improves loss, but slowly. Particularly, doubling N only multiplies loss by
$$
2^{-0.076} \approx 0.95
$$
$\implies 2\times$  larger model only provides a $5\%$ loss reduction, assuming no data or compute bottleneck.

### Dataset-size scaling
For large models trained on limited data with early stopping, Kaplan et. all sees
$$
L(D) = \left( \frac{D_c}{D} \right)^{\alpha_D}
$$
with roughly, $\alpha_{D} \approx 0.095$ and $D_{c} \approx 5.4 \times 10^{13}$ tokens. This implies more data improves loss, but slowly. That said, it is a faster improvement than increasing model size.

### Compute scaling
For optimally allocated compute,
$$
L(C_{min}) = \left( \frac{C_{c}^{min}}{C_{min}} \right)^{\alpha_{C}^{min}}
$$
with roughly, $\alpha_{C}^{min} \approx 0.050$ and $C_{c}^{min} \approx 3.1 \times 10^{8}$ PF-days. $C_{min}$ is Kaplan's batch-size adjusted estimate of the minimum compute needed to reach a given loss. They prefer this over the raw fixed-batch compute curve.

### Transformer parameter and FLOP estimate
For a standard transformer shape,
$$
N \approx 12 n_{layer} d_{model}^2
$$
assuming
$$
d_{attn} = d_{model}, \quad d_{ff} = 4d_{model}
$$
The approximate training compute per token is
$$
C_{token} \approx 6N
$$
since the forward pass is about $2N$ FLOPs/token and backward is about twice the forward pass ($4N$ per token).
So we can use,
$$
C = 6 \times N_{non-embd} \times \text{global batch tokens} \times \text{optimizer steps}
$$
where
$$
\text{global batch tokens} = \text{batch size per GPU} \times \text{sequence length} \times \text{grad. accl.} \times \text{world size}
$$

### Critical batch size
Kaplan models the critical batch size as
$$
B_{crit}(L) = \frac{B_*}{L^{1/\alpha_B}}
$$
with $B_{*} \sim 2 \times 10^8$ tokens and $\alpha_{B}\sim 0.21$. This is the batch size where we get a good compromise between wall-clock speed and compute efficiency.

IMPORTANT
- if the batch size is much larger than the critical size, we may be wasting compute; if it is much smaller, we may be compute efficient but slower in wall time.
- Since particleGPT is not natural language, we cannot assume Kaplan's $B_{crit}$ constants transfer.
	- IMPORTANT: It is best to test by running short batch-size sweeps and seeing when increasing batch size stops improving tokens/sec-to-loss efficiency.

### Join model/data scaling
Kaplan's key combiend equation is
$$
L(N, D) = \left[ \left( \frac{N_{c}}{N} \right)^{\alpha_{N}/\alpha_{D}} + \left( \frac{D_c}{D} \right)\right]^{\alpha_{D}}
$$
The Kaplan fit provides
$$
\alpha_{N} \approx 0.076, \quad \alpha_{D} \approx 0.103, \quad D_{c} \approx 1.8 \times 10^{13}, \quad N_{c} \approx 6.4 \times 10^{13}
$$
This equation explains the dataset bottleneck. At fixed $D$, increasing $N$ eventually stops helping because the $D_c/D$ term dominates.

### Overfitting/data bottleneck equation
Kaplan defines the overfitting penalty as
$$
\delta L(N, D) = \frac{L(N, D)}{L(N, \infty)} - 1
$$
Their join law approximately implies
$$
\delta L = \left[1 + \left( \frac{N_{c}}{N} \right)^{\alpha_{N}/\alpha_{D}} \left( \frac{D_c}{D} \right)\right]^{\alpha_{D}}-1
$$
To keep overfitting below roughly run-to-run noise, they estimate we need
$$
D \gtrsim 5 \times 10^{3} N^{0.74}
$$
where the exponent is extracted via
$$
\frac{\alpha_N}{\alpha_{D}} \approx \frac{0.076}{0.103} \approx 0.74
$$
This gives us a clean diagnostic. For each dataset size, plot validation loss versus model size. If curves flatten at large $N$, we are data-limited. Then, we can tryu collapsing our results against
$$
\frac{N^{\gamma}}{D}
$$
and fit $\gamma$. Kaplan finds $\gamma \approx 0.74$, but our value from particleGPT will likely be different (not being a natural language model).


### Training-time / step scaling
Kaplan also fits learning curves with
$$
L(N,S_{\min})=\left(\frac{N_c}{N}\right)^{\alpha_N}+\left(\frac{S_c}{S_{\min}}\right)^{\alpha_S}
$$
The fitted values are roughly
$$
\alpha_N \approx 0.077, \qquad\alpha_S \approx 0.76, \qquad N_c \approx 6.5\times 10^{13}, \qquad S_c \approx 2.1\times 10^3
$$
Here, $S_{\min}$ is an adjusted minimum number of optimizer steps, corrected for batch-size inefficiency. The meaning is that the loss has two pieces:
$$
\text{loss} = \text{model-size-limited loss} + \text{optimization/training-time-limited loss}
$$
The first term,
$$
\left(\frac{N_c}{N}\right)^{\alpha_N}
$$
is the best loss the model could approach if trained long enough. The second term
$$
\left(\frac{S_c}{S_{\min}}\right)^{\alpha_S}
$$
is the extra loss from not training long enough. For particleGPT, a useful practical fitting form is
$$
L(S)=L_\infty + A S^{-\alpha_S}
$$
where $L_\infty$ is the asymptotic loss for a given model and dataset, $A$ is a fitted constant, and $\alpha_S$ tells us how quickly training improves with more optimizer steps. This means individual training curves should look approximately like power-law decay after the early warmup/transient region. If our fitted $\alpha_S$ is stable across model sizes, then we can conclude particleGPT has Kaplan-like training dynamics.

### Early-stopping lower bound  
For finite data, Kaplan gives a rough lower bound on the useful early-stopping step:  
  
$$
S_{\rm stop}(N,D)  \gtrsim  \frac{S_c}{\left[L(N,D)-L(N,\infty)\right]^{1/\alpha_S}}  
$$
Here, $L(N,D)$ is the loss for a model with $N$ parameters trained on dataset size $D$, while $L(N,\infty)$ is the loss the same model would get with effectively infinite data. The difference 
$$
L(N,D)-L(N,\infty)  
$$
measures how much the finite dataset is hurting performance. If this difference is large, then the model is strongly data-limited and useful training stops earlier. If this difference is small, then more training steps can still help. For particleGPT, this supports using validation-loss patience rather than a fixed number of epochs. If validation loss has stopped improving, but training loss keeps decreasing, the model is probably overfitting or becoming data-limited.

### Compute-optimal allocation  
Kaplan derives that, under their fitted scaling laws, the compute-optimal model size, batch size, and number of steps scale as powers of the total compute budget. The general forms are  
$$
N \propto C_{\min}^{\alpha_C^{\min}/\alpha_N},  \qquad
B \propto C_{\min}^{\alpha_C^{\min}/\alpha_B},  \qquad
S \propto C_{\min}^{\alpha_C^{\min}/\alpha_S},  \qquad
D = BS
$$
Empirically, Kaplan summarizes this approximately as  
  
$$  
N \propto C_{\min}^{0.73},  \qquad  
B \propto C_{\min}^{0.24},  \qquad
S \propto C_{\min}^{0.03}
$$
The important conclusion is: as compute increases, most of the extra compute should go into increasing model size, not into training the same model for dramatically more steps.

In words:    
$$  
\text{more compute} \Rightarrow \text{much larger model} + \text{somewhat larger batch} + \text{only slightly more steps}
$$
This is one of Kaplan's most important conclusions: compute-efficient training means using a large model and stopping well before full convergence, rather than training a small model to convergence.  

For particleGPT, this means we should compare models at fixed compute budgets. A larger model trained for fewer steps may beat a smaller model trained for many more steps, even if the smaller model is closer to convergence.