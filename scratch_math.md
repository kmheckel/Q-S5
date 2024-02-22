SSM Continuous - note that $A$ is diagonalized in SSMs to facilitate efficent computation via recurrent scans, making it equivalent to individually learned time constants in SNNs! This means that LI neurons can be computed in logarithmic time on sufficient hardware. Then just need to combine with a stochastic firing mechanism? The Sigmoid-Bernoulli activation from SPSN was overactive and controlled by spike count regularization, but perhaps a more intelligent interpretation of the membrane potential could be used?

Sigmoid-Bernoulli:
$$
p[n] = \sigma(\textbf{u}[n])
$$
$$
\textbf{s}[n] = Bernoulli(p[n])
$$
$$
\frac{\delta \textbf{s}[n]}{p[n]} \approx p[n]
$$

Perhaps a better alternative would be to include some approximation of a refractory period for high preactivations/membrane potentials:


$$
p[n] = \sigma(k_1\textbf{u}[n]) \odot sin(k_2\textbf{u}[n])
$$

Where $k_1$ governs the probabilities around the threshold and $k_2$  is a tunable refractory parameter. 
$$
\textbf{s}[n] = Bernoulli(p[n])
$$
$$
\frac{\delta \textbf{s}[n]}{p[n]} \approx p[n]
$$




---------------------------------------------------------------------------------------------------------------------------------------------------------------

$$
\frac{dx(t)}{dt} = Ax(t) + Bu(t)
$$

$$
y(t) = Cx(t) + Du(t)
$$

SSM Discrete

$$
x_k = Ax_{k-1} + Bu_k
$$

$$
y_k = Cx_k + Du_k
$$

LI Discrete:

$$
    V_k = \beta V_{k-1} + WX_k
$$

Rewritten:

$$
x_k = \beta x_{k-1} + Bu(t)
$$

CuBaLI Discrete

$$
I_k = \alpha I_{k-1} + WX_k
$$

$$
V_k = \beta V_{k-1} + I_k
$$

Rewritten:

$$
x_k = \alpha x_{k-1} + Bu_k
$$

$$
y_k = x_k + \beta y_{k-1}
$$

It is interesting to note the differences between SSM discrete and CuBaLI discrete - SSM has a linear decoding operator C and learnable skip connection via D, while CuBaLI's output takes the hidden state directly and is itself recurrent. It would be interesting to compare the performance differences of S5 and CuBaLI when both are initialized the same way.
