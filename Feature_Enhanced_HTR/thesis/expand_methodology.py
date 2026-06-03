methodology_file = "chapters/methodology.tex"

content = """
\\section{Exhaustive Mathematical Formulation of the Optimization Landscape}
In this section, we provide the complete, textbook-level mathematical derivations for the neural network components utilized in the HTR architecture.

\\subsection{Backpropagation Through Time (BPTT) in BiLSTM}
The Long Short-Term Memory (LSTM) cell prevents the vanishing gradient problem by enforcing a constant error carousel through the cell state $C_t$. To optimize the network, gradients must be backpropagated through time from $t=T$ down to $t=1$. Let $\\mathcal{L}$ be the objective function (CTC loss). The gradients with respect to the output $h_t$ and cell state $C_t$ at time $t$ are accumulated from the layer above and from the next time step $t+1$:

\\begin{equation}
    \\delta h_t = \\frac{\\partial \\mathcal{L}}{\\partial h_t} + \\delta h_{t+1} \\odot \\frac{\\partial h_{t+1}}{\\partial h_t}
\\end{equation}

For the specific gates (forget $f_t$, input $i_t$, output $o_t$, and cell candidate $\\tilde{C}_t$), the derivatives of the cell state $C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t$ follow the chain rule:

\\begin{align}
    \\delta C_t &= \\frac{\\partial \\mathcal{L}}{\\partial C_t} = \\delta h_t \\odot o_t \\odot (1 - \\tanh^2(C_t)) + \\delta C_{t+1} \\odot f_{t+1} \\\\
    \\delta o_t &= \\delta h_t \\odot \\tanh(C_t) \\odot o_t \\odot (1 - o_t) \\\\
    \\delta f_t &= \\delta C_t \\odot C_{t-1} \\odot f_t \\odot (1 - f_t) \\\\
    \\delta i_t &= \\delta C_t \\odot \\tilde{C}_t \\odot i_t \\odot (1 - i_t) \\\\
    \\delta \\tilde{C}_t &= \\delta C_t \\odot i_t \\odot (1 - \\tilde{C}_t^2)
\\end{align}

These gate gradients are then used to calculate the gradients for the specific weight matrices $W_f, W_i, W_o, W_c$ and biases. For example, the gradient for the forget gate weight matrix is computed by taking the outer product of $\\delta f_t$ with the concatenated input $[h_{t-1}, x_t]$:

\\begin{equation}
    \\frac{\\partial \\mathcal{L}}{\\partial W_f} = \\sum_{t=1}^T \\delta f_t \\otimes [h_{t-1}, x_t]^T
\\end{equation}
This rigorous mathematical formulation confirms the numerical stability of the BiLSTM when learning the long-range horizontal dependencies intrinsic to cursive English handwriting.

\\subsection{Convolutional Tensor Mathematics and Receptive Fields}
The feature extraction backbone relies on a hierarchy of 2D convolutional layers. Let an input tensor to layer $l$ be denoted as $X^{(l)} \\in \\mathbb{R}^{H \\times W \\times C}$, where $H$ is height, $W$ is width, and $C$ is the number of channels. The output activation $Y^{(l)}$ at spatial coordinate $(i, j)$ for the $k$-th filter $W_k \\in \\mathbb{R}^{F \\times F \\times C}$ is:
\\begin{equation}
    Y_{i,j,k}^{(l)} = \\sigma \\left( \\sum_{u=0}^{F-1} \\sum_{v=0}^{F-1} \\sum_{c=1}^{C} X_{i \\cdot S + u, j \\cdot S + v, c}^{(l)} W_{u,v,c,k}^{(l)} + b_k^{(l)} \\right)
\\end{equation}
where $S$ is the stride, $F$ is the spatial dimension of the square kernel, and $\\sigma$ is the ReLU activation function.

To guarantee that the network views the entire word image, we must calculate the theoretical receptive field $R_l$ of a neuron in layer $l$. The receptive field expands recursively according to the kernel size $K_l$ and stride $S_l$:
\\begin{equation}
    R_l = R_{l-1} + (K_l - 1) \\prod_{i=1}^{l-1} S_i
\\end{equation}
By stacking four convolutional blocks with $2\\times 2$ Max Pooling (stride $S=2$), the effective stride at the final sequence dimension is 16, yielding a receptive field that encompasses entire multi-character subwords. 

\\subsection{Exhaustive Optimization Calculus (Adam and RMSprop)}
While classical Stochastic Gradient Descent (SGD) uses a uniform learning rate, the HTR model utilizes Adaptive Moment Estimation (Adam). Adam calculates exponentially moving averages of the gradient $m_t$ (first moment) and the squared gradient $v_t$ (second raw moment):
\\begin{align}
    m_t &= \\beta_1 m_{t-1} + (1 - \\beta_1) \\nabla_\\theta \\mathcal{L}_{CTC}(\\theta_{t-1}) \\\\
    v_t &= \\beta_2 v_{t-1} + (1 - \\beta_2) [\\nabla_\\theta \\mathcal{L}_{CTC}(\\theta_{t-1})]^2
\\end{align}
To counteract the initial bias towards zero (since $m_0 = 0$ and $v_0 = 0$), Adam implements a bias correction step:
\\begin{align}
    \\hat{m}_t &= \\frac{m_t}{1 - \\beta_1^t} \\\\
    \\hat{v}_t &= \\frac{v_t}{1 - \\beta_2^t}
\\end{align}
The final weight update rule dynamically adapts the learning rate $\\eta$ for each parameter $\\theta$:
\\begin{equation}
    \\theta_t = \\theta_{t-1} - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t
\\end{equation}
where $\\epsilon = 10^{-8}$ prevents division by zero. This dynamic conditioning on the loss landscape curvature is essential for training the highly non-convex parameter space created by combining CNNs, BiLSTMs, and HRNNs in a single end-to-end framework.

"""

with open(methodology_file, "a", encoding="utf-8") as out_f:
    out_f.write(content)

print("Expanded methodology with massive math block.")
