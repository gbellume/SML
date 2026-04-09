"""
Shared PDE residual functions for the 2D vorticity-streamfunction formulation.

These are the single source of truth: generate.py calls them with numpy FD
derivatives, train.py calls them with torch tensor derivatives (differentiable).

No framework imports — pure arithmetic so they work with both numpy and torch.
"""


def vorticity_transport_residual(dw_dt, u, dw_dx, v, dw_dy, d2w_dx2, d2w_dy2, nu):
    """
    Vorticity transport equation residual.

        R = ∂ω/∂t + u·∂ω/∂x + v·∂ω/∂y − ν(∂²ω/∂x² + ∂²ω/∂y²)

    R = 0 for an exact solution.  All arguments are arrays/tensors of the
    same shape (or broadcastable).
    """
    return dw_dt + u * dw_dx + v * dw_dy - nu * (d2w_dx2 + d2w_dy2)


def poisson_residual(d2psi_dx2, d2psi_dy2, omega):
    """
    Poisson equation residual for the streamfunction.

        R = ∂²ψ/∂x² + ∂²ψ/∂y² + ω

    R = 0 for an exact solution.
    """
    return d2psi_dx2 + d2psi_dy2 + omega


def velocity_from_streamfunction(dpsi_dy, dpsi_dx):
    """
    Recover velocity components from streamfunction derivatives.

        u =  ∂ψ/∂y
        v = −∂ψ/∂x

    Returns (u, v).
    """
    return dpsi_dy, -dpsi_dx
