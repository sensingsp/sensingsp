"""
tex.py: Functions for generating LaTeX strings for sensingSP visualization module.
Each function generates reusable LaTeX strings for radar equations, positions,
and other mathematical notations.
"""

def target_position():
    return r"\mathbf{p}^t_k"

def tx_position():
    return r"\mathbf{p}^{tx}"

def rx_position():
    return r"\mathbf{p}^{rx}"

def tx_rx_phase():
    return r"\phi_{k,m,n}"

def wavenumber():
    return r"\frac{2\pi}{\lambda}"

def wavenumber2():
    return r"\frac{4\pi}{\lambda}"

def tx_target_distance():
    return r"\|" + target_position() + r" - " + tx_position() + r"_m\|"

def rx_target_distance():
    return r"\|" + target_position() + r" - " + rx_position() + r"_m\|"

def target_range():
    return r"\text{R}_k"

def target_unit_vector():
    return r"\hat{\mathbf{r}}_k"

def target_unit_vector_angles():
    return (
        r"\begin{bmatrix}"
        r"\cos\theta_k \cos\phi_k \\"
        r"\sin\theta_k \cos\phi_k \\"
        r"\sin\phi_k"
        r"\end{bmatrix}"
    )

def tx_ura():
    return (
        r"\mathbf{p}^{\text{tx}}_{m_y,m_z} = \begin{bmatrix}"
        r"0\\"
        r"(m_y - \frac{M_y - 1}{2})d_{y}^{tx}\\"
        r"(m_z - \frac{M_z - 1}{2})d_{z}^{tx}\\"
        r"\end{bmatrix}"
    )

def rx_ura():
    return (
        r"\mathbf{p}^{\text{rx}}_{n_y,n_z} = \begin{bmatrix}"
        r"0\\"
        r"(n_y - \frac{N_y - 1}{2})d_{y}^{rx}\\"
        r"(n_z - \frac{N_z - 1}{2})d_{z}^{rx}\\"
        r"\end{bmatrix}"
    )

def my_limit():
    return r"m_y = 0,\hdots,M_y-1"

def ny_limit():
    return r"n_y = 0,\hdots,N_y-1"

def mz_limit():
    return r"m_z = 0,\hdots,M_z-1"

def nz_limit():
    return r"n_z = 0,\hdots,N_z-1"

def raw_data_cube():
    return r"\bm{\mathsf{T}}_q \in \mathbb{C}^{K_q \times L_q \times N_q}"
def signal_steering_matrix():
    return "\(\mathbf{S}_q(\mathbf{p}, \mathbf{v}) = \mathbf{c}_q(\mathbf{p}, \mathbf{v}) \mathbf{b}^T_q(\mathbf{p})\)"
def cq():
    return (
    r"\begin{aligned}"
    r"\mathbf{c}_q(\mathbf{p}, \mathbf{v}) = (\mathbf{W}_q \mathbf{a}_q(\mathbf{p})) \odot \mathbf{d}_q(f_d(\mathbf{p}, \mathbf{v}))"
    r"\end{aligned}"
    )
def transmit_steering_vector():
    return r"\mathbf{a}_{q}(\mathbf{p}) = \left[e^{-j\frac{2\pi}{\lambda} \|\mathbf{p} - \mathbf{p}_{q,1}^{tx}\|}, \ldots, e^{-j\frac{2\pi}{\lambda} \|\mathbf{p} - \mathbf{p}_{q,M_q}^{tx}\|}\right]^T"
def receive_steering_vector():
    return r"\mathbf{b}_{q}(\mathbf{p}) = \left[e^{-j\frac{2\pi}{\lambda} \|\mathbf{p} - \mathbf{p}_{q,1}^{rx}\|}, \ldots, e^{-j\frac{2\pi}{\lambda} \|\mathbf{p} - \mathbf{p}_{q,N_q}^{rx}\|}\right]^T"
def Doppler_steering_vector():
    return r"\mathbf{d}_{q}(f) = \left[1, \ldots, e^{j2\pi f(L_q-1)\tau}\right]^T"
def doppler():
    return r"f_d(\mathbf{p}, \mathbf{v}) = -\frac{2(\mathbf{v}-\mathbf{v}_c)^T\mathbf{n}}{\lambda}"

def doppler_processing_x():
    return (
    r"\begin{aligned}"
    r"\mathbf{x}^{(n)}_{q,f_d} = \mathbf{W}_q^\dagger \left(\mathbf{y}^{(n)}_q \odot\mathbf{d}^*_q(f_d)\right) \in \mathbb{C}^{M_q}"
    r"\end{aligned}"
    )


def detector_MF():
    return r"t_{\text{MF}}(\mathbf{x}) = \mathbf{x}^H\mathbf{M}^{-1} \mathbf{S}(\mathbf{S}^H\mathbf{M}^{-1}\mathbf{S})^{-1}\mathbf{S}^H\mathbf{M}^{-1}\mathbf{x}"
def detector_AMF():
    return r"t_{\text{AMF}}(\mathbf{x}) = \mathbf{x}^H\mathbf{\Sigma}^{-1} \mathbf{S}(\mathbf{S}^H\mathbf{\Sigma}^{-1}\mathbf{S})^{-1}\mathbf{S}^H\mathbf{\Sigma}^{-1}\mathbf{x}"
def detector_NAMF():
    return r"t_{\text{NAMF}}(\mathbf{x}) = \frac{\mathbf{x}^H\mathbf{\Sigma}^{-1} \mathbf{S}(\mathbf{S}^H\mathbf{\Sigma}^{-1}\mathbf{S})^{-1}\mathbf{S}^H\mathbf{\Sigma}^{-1}\mathbf{x}}{\mathbf{x}^H\mathbf{\Sigma}^{-1}\mathbf{x}}"
def detector_Kelly():
    return (
        r"\begin{aligned}"
        r"t(\mathbf{x}) = \frac{\mathbf{x}^H \mathbf{\Sigma}^{-1} \mathbf{S} \left(\mathbf{S}^H \mathbf{\Sigma}^{-1} \mathbf{S}\right)^{-1} \mathbf{S}^H \mathbf{\Sigma}^{-1} \mathbf{x}}"
        r"{1 + \mathbf{x}^H \mathbf{\Sigma}^{-1} \mathbf{x}}"
        r"\mathop{\gtrless}\limits^{\mathcal{H}_1}_{\mathcal{H}_0} \eta''"
        r"\end{aligned}"
    )
def detector_Kelly_vec():
    return (
        r"\begin{aligned}"
        r"t(\mathbf{x}) = \frac{\| \mathbf{s}^H \mathbf{\Sigma}^{-1} \mathbf{x}\|^2}"
        r"{\mathbf{s}^H \mathbf{\Sigma}^{-1} \mathbf{s}(1 + \mathbf{x}^H \mathbf{\Sigma}^{-1} \mathbf{x})}"
        r"\mathop{\gtrless}\limits^{\mathcal{H}_1}_{\mathcal{H}_0} \eta"
        r"\end{aligned}"
    )
def detector_Kelly_siso():
    return (
        r"\begin{aligned}"
        r"t(\mathbf{x}) = \frac{\|D^{-1}m_{\sigma^2}^{-1} x\|^2}"
        r"{D^{-1}m_{\sigma^2}^{-1}(1 + x^* D^{-1}m_{\sigma^2}^{-1} x)}="
        r"\frac{\|x\|^2}"
        r"{m_{\sigma^2} (D + \| x\|^2/m_{\sigma^2})}"
        r"\mathop{\gtrless}\limits^{\mathcal{H}_1}_{\mathcal{H}_0} \eta"
        r"\end{aligned}"
    )
def detector_Kelly_siso_simple():
    return (
        r"\begin{aligned}"
        r"t(\mathbf{x}) ="
        r"\frac{\|x\|^2}"
        r"{m_{\sigma^2}}"
        r"\mathop{\gtrless}\limits^{\mathcal{H}_1}_{\mathcal{H}_0} \eta"
        r"\end{aligned}"
    )

def hypothesis_text():
    return (
        r"\begin{aligned}"
        r"\begin{cases}"
        r"\mathcal{H}_0 :& \mathbf{x} = \mathbf{n}, \, \mathbf{x}_d = \mathbf{n}_d; \, d = 1, \hdots, D\\"
        r"\mathcal{H}_1 :& \mathbf{x} = \mathbf{S}\boldsymbol{\alpha} + \mathbf{n}, \, \mathbf{x}_d = \mathbf{n}_d; \, d = 1, \hdots, D."
        r"\end{cases}"
        r"\end{aligned}"
    )
def hypothesisH1_text():
    return (
        r"\mathcal{H}_1 :& \mathbf{x} = \mathbf{S}\boldsymbol{\alpha} + \mathbf{n}, \, \mathbf{x}_d = \mathbf{n}_d; \, d = 1, \hdots, D."
     )
def hypothesisH0_text():
    return (
        r"\mathcal{H}_0 :& \mathbf{x} = \mathbf{n}, \, \mathbf{x}_d = \mathbf{n}_d; \, d = 1, \hdots, D\\"
            )

def glrt_ratio_text():
    return (
        r"\begin{aligned}"
        r"\frac{\max\limits_{\mathbf{M}, \boldsymbol{\alpha}} p(\mathbf{x}, \mathbf{x}_1, \mathbf{x}_2, \hdots, \mathbf{x}_D | \mathcal{H}_1)}"
        r"{\max\limits_{\mathbf{M}} p(\mathbf{x}, \mathbf{x}_1, \mathbf{x}_2, \hdots, \mathbf{x}_D | \mathcal{H}_0)}"
        r"\mathop{\gtrless}\limits^{\mathcal{H}_1}_{\mathcal{H}_0} \eta"
        r"\end{aligned}"
    )

def pdfH1():
    return (
        r"p(\mathbf{x}, \mathbf{x}_1, \mathbf{x}_2, \hdots, \mathbf{x}_D | \mathcal{H}_1)=\frac{1}{\sigma |\mathbf{M}|^{D+1}}e^{-\mathrm{Tr}(\mathbf{M}^{-1}((\mathbf{x} - \mathbf{S}\boldsymbol{\alpha})(\mathbf{x} - \mathbf{S}\boldsymbol{\alpha})^H + \mathbf{\Sigma}))}}"
    )

def pdfH0():
    return (
        r"p(\mathbf{x}, \mathbf{x}_1, \mathbf{x}_2, \hdots, \mathbf{x}_D | \mathcal{H}_0)=\frac{1}{\sigma |\mathbf{M}|^{D+1}}e^{-\mathrm{Tr}(\mathbf{M}^{-1}(\mathbf{x}\mathbf{x}^H + \mathbf{\Sigma}))}}"
    )

def Sigma():
    return r"\mathbf{\Sigma} = \sum_{d=1}^D \mathbf{x}_d \mathbf{x}_d^H"
def yofx():
    return r"\mathbf{y} = \mathbf{x} - \mathbf{S}\boldsymbol{\alpha}"
def sigma():
    return r"\sigma = \pi^{\Omega (D+1)}"
def Omega():
    return r"\Omega = \sum_{i=1}^{Q'} M_{l_i}N_{l_i}"
def ml_estimates_text():
    return (
        r"\begin{aligned}"
        r"\mathbf{\hat{M}}_1 = \frac{1}{D+1}(\mathbf{y}\mathbf{y}^H + \mathbf{\Sigma}), \quad"
        r"\mathbf{\hat{M}}_0 = \frac{1}{D+1}(\mathbf{x}\mathbf{x}^H + \mathbf{\Sigma})"
        r"\end{aligned}"
    )

def determinant_relation_text():
    return (
        r"\begin{aligned}"
        r"|\mathbf{\Sigma} + \mathbf{y}\mathbf{y}^H| = (1 + \mathbf{y}^H\mathbf{\Sigma}^{-1}\mathbf{y})|\mathbf{\Sigma}|"
        r"\end{aligned}"
    )

def simplified_detector_text():
    return (
        r"\begin{aligned}"
        r"|\mathbf{\hat{M}}_0|\max\limits_{\boldsymbol{\alpha}}\frac{1}{|\mathbf{\hat{M}}_1|} = "
        r"\frac{1 + \mathbf{x}^H\mathbf{\Sigma}^{-1}\mathbf{x}}"
        r"{1 + \min\limits_{\boldsymbol{\alpha}}(\mathbf{x} - \mathbf{S}\boldsymbol{\alpha})^H \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{S}\boldsymbol{\alpha})}"
        # r"\mathop{{\gtrless }}\limits^{{\rm \mathcal{H}}_1}_{{\rm \mathcal{H}}_{0}}\eta"
        r"\end{aligned}"
    )

def final_detector_text():
    return (
        r"\begin{aligned}"
        r"t(\mathbf{x}) = \frac{\mathbf{x}^H\mathbf{\Sigma}^{-1} \mathbf{S}(\mathbf{S}^H\mathbf{\Sigma}^{-1}\mathbf{S})^{-1}\mathbf{S}^H\mathbf{\Sigma}^{-1}\mathbf{x}}"
        r"{1 + \mathbf{x}^H\mathbf{\Sigma}^{-1}\mathbf{x}}"
        # r"\mathop{{\gtrless }}\limits^{{\rm \mathcal{H}}_1}_{{\rm \mathcal{H}}_{0}}\eta"
        r"\end{aligned}"
    )
def xd_distribution_text():
    return (
        r"\mathbf{n}_d \sim \mathcal{CN}(\mathbf{0}, \mathbf{M}) \text{ for } d = 1, \hdots, D"
    )
def glrtks_text():
    return (
        r"    \begin{aligned}"
        r"    \frac{\max\limits_{\mathbf{M},\boldsymbol{\alpha}} \frac{1}{\sigma |\mathbf{M}|^{D+1}}"
        r"    e^{-\mathrm{Tr}(\mathbf{M}^{-1}(\mathbf{y}\mathbf{y}^H + \mathbf{\Sigma}))}}"
        r"    {\max\limits_{\mathbf{M}} \frac{1}{\sigma |\mathbf{M}|^{D+1}}"
        r"    e^{-\mathrm{Tr}(\mathbf{M}^{-1}(\mathbf{x}\mathbf{x}^H + \mathbf{\Sigma}))}}"
        r"\mathop{\gtrless}\limits^{\mathcal{H}_1}_{\mathcal{H}_0} \eta"
        r"    \end{aligned}"
    )

# \begin{equation}\label{eq:detect1}
#     \begin{aligned}
#         \begin{cases}
#           \mathcal{H}_0 :&\mathbf{x} = \mathbf{n}, \, \mathbf{x}_d = \mathbf{n}_d; \, d = 1, \hdots, D\\
#           \mathcal{H}_1 :&\mathbf{x} = \mathbf{S}\boldsymbol{\alpha} + \mathbf{n}, \, \mathbf{x}_d = \mathbf{n}_d; \, d = 1, \hdots, D.
#         \end{cases}
#     \end{aligned}
# \end{equation}

# \begin{equation}\label{eq:glrtks}
#     \begin{aligned}
#     \frac{\max\limits_{\mathbf{M},\boldsymbol{\alpha}}p(\mathbf{x},\mathbf{x}_1,\mathbf{x}_2,\hdots,\mathbf{x}_D|\mathcal{H}_1)}{\max\limits_{\mathbf{M}}p(\mathbf{x},\mathbf{x}_1,\mathbf{x}_2,\hdots,\mathbf{x}_D|\mathcal{H}_0)}\mathop{{\gtrless }}\limits^{{\rm
#  \mathcal{H}}_1}_{{\rm \mathcal{H}}_{0}}\eta
#     \end{aligned}
# \end{equation}

# By defining \(\mathbf{\Sigma} = \sum_{d=1}^D \mathbf{x}_d \mathbf{x}_d^H\), \(\mathbf{y} = \mathbf{x} - \mathbf{S}\boldsymbol{\alpha}\), and \(\sigma = \pi^{\Omega (D+1)}\)

# \begin{equation}\label{eq:glrtks}
#     \begin{aligned}
#     \frac{\max\limits_{\mathbf{M},\boldsymbol{\alpha}}\frac{1}{\sigma|\mathbf{M}|^{D+1}}
#     e^{-\mathrm{Tr}(\mathbf{M}^{-1}(\mathbf{y}\mathbf{y}^H+\mathbf{\Sigma}))}}{\max\limits_{\mathbf{M}}\frac{1}{\sigma|\mathbf{M}|^{D+1}}
#     e^{-\mathrm{Tr}(\mathbf{M}^{-1}(\mathbf{x}\mathbf{x}^H+\mathbf{\Sigma}))}}\mathop{{\gtrless }}\limits^{{\rm
#  \mathcal{H}}_1}_{{\rm \mathcal{H}}_{0}}\eta
#     \end{aligned}
# \end{equation}

# \(
# \mathbf{\hat{M}}_1 = \frac{1}{D+1}(\mathbf{y}\mathbf{y}^H + \mathbf{\Sigma})
# \)
# and
# \(
# \mathbf{\hat{M}}_0 = \frac{1}{D+1}(\mathbf{x}\mathbf{x}^H + \mathbf{\Sigma})
# \)
# Using these ML estimates and the lemma \( |\mathbf{\Sigma} + \mathbf{y}\mathbf{y}^H| = (1 + \mathbf{y}^H\mathbf{\Sigma}^{-1}\mathbf{y})|\mathbf{\Sigma}| \),
# \begin{equation}\label{eq:glrtks}
#     \begin{aligned}
#     |\mathbf{\hat{M}}_0|\max\limits_{\boldsymbol{\alpha}}\frac{1}{|\mathbf{\hat{M}}_1|}=\frac{1+\mathbf{x}^H\mathbf{\Sigma}^{-1}\mathbf{x}}{1+\min\limits_{\boldsymbol{\alpha}}(\mathbf{x}-\mathbf{S}\boldsymbol{\alpha})^H\mathbf{\Sigma}^{-1}(\mathbf{x}-\mathbf{S}\boldsymbol{\alpha})}
#     \mathop{{\gtrless }}\limits^{{\rm
#  \mathcal{H}}_1}_{{\rm \mathcal{H}}_{0}}\eta
#     \end{aligned}
# \end{equation}

# The ML estimation of \(\boldsymbol{\alpha}\) can then be easily calculated as:
# \(
# \hat{\boldsymbol{\alpha}} = \mathbf{\Gamma}\mathbf{x}
# \)
# where \(\mathbf{\Gamma} = (\mathbf{S}^H\mathbf{\Sigma}^{-1}\mathbf{S})^{-1}\mathbf{S}^H\mathbf{\Sigma}^{-1}\). Therefore,
# % \small
# \begin{equation}\label{eq:glrtks}
#     \begin{aligned}
#     \frac{1+\mathbf{x}^H\mathbf{\Sigma}^{-1}\mathbf{x}}{1+(\mathbf{x}-\mathbf{S}\mathbf{\Gamma}\mathbf{x})^H\mathbf{\Sigma}^{-1}(\mathbf{x}-\mathbf{S}\mathbf{\Gamma}\mathbf{x})}
#     \mathop{{\gtrless }}\limits^{{\rm
#  \mathcal{H}}_1}_{{\rm \mathcal{H}}_{0}}\eta
#     \end{aligned}
# \end{equation}
# % Due to $f(x)=1-1/x$ is a monotonically increasing function, we can define
# Since \( f(x) = 1 - \frac{1}{x} \) is a monotonically increasing function, we can define:
# % \small
# \begin{equation}\label{eq:glrtks}
#     \begin{aligned}    \frac{\mathbf{x}^H\mathbf{\Gamma}^H\mathbf{S}^H\mathbf{\Sigma}^{-1}\mathbf{S}\mathbf{\Gamma}\mathbf{x}}{1+\mathbf{x}^H\mathbf{\Sigma}^{-1}\mathbf{x}}
#     \mathop{{\gtrless }}\limits^{{\rm
#  \mathcal{H}}_1}_{{\rm \mathcal{H}}_{0}}\eta
#     \end{aligned}
# \end{equation}
# % and finally
# Finally, the detector under the assumption of a known covariance matrix is derived as:
# \begin{equation}\label{eq:glrtfinal}
#     \begin{aligned}
#     t(\mathbf{x})=\frac{\mathbf{x}^H\mathbf{\Sigma}^{-1} \mathbf{S}(\mathbf{S}^H\mathbf{\Sigma}^{-1}\mathbf{S})^{-1}\mathbf{S}^H\mathbf{\Sigma}^{-1}\mathbf{x}}{1+\mathbf{x}^H\mathbf{\Sigma}^{-1}\mathbf{x}}
#     \mathop{{\gtrless }}\limits^{{\rm
#  \mathcal{H}}_1}_{{\rm \mathcal{H}}_{0}}\eta
#     \end{aligned}
# \end{equation}

# # Example usage in code
# if __name__ == "__main__":
#     print("Target Position LaTeX String:", target_position())
#     print("Tx Position LaTeX String:", tx_position())
#     print("Wavenumber LaTeX String:", wavenumber())
#     print("Tx Target Distance:", tx_target_distance())
