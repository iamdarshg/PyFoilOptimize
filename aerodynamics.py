import os
import numpy as np
from numpy.random import uniform
from numpy import meshgrid, linspace, sqrt, pi, arccos, cos, empty_like, copy, sin, append, radians, empty, fill_diagonal, dot, vectorize, ones_like, ones, loadtxt
from scipy.integrate import quad
from numpy.linalg import solve
import warnings

class Panel:
    """A class representing a single panel on the airfoil surface."""
    def __init__(self, xa, ya, xb, yb):
        self.xa, self.ya = xa, ya
        self.xb, self.yb = xb, yb

        self.xc, self.yc = (xa + xb) / 2, (ya + yb) / 2
        self.length = sqrt((xb - xa)**2 + (yb - ya)**2)

        if xb - xa <= 0.0:
            self.beta = arccos((yb - ya) / self.length)
        else:
            self.beta = pi + arccos(-(yb - ya) / self.length)

        self.loc = 'upper' if self.beta <= pi else 'lower'
        self.sigma = 0.0
        self.vt = 0.0
        self.cp = 0.0

class Freestream:
    """A class representing the freestream conditions."""
    def __init__(self, u_inf=1.0, alpha=4.0):
        self.u_inf = u_inf
        self.alpha = radians(alpha)

def define_panels(x, y, N=40):
    """Discretizes the geometry into panels."""
    R = (x.max() - x.min()) / 2.0
    x_center = (x.max() + x.min()) / 2.0

    theta = linspace(0.0, 2.0 * pi, N + 1)
    x_circle = x_center + R * cos(theta)

    x_ends = copy(x_circle)
    y_ends = empty_like(x_ends)

    x, y = append(x, x[0]), append(y, y[0])

    I = 0
    for i in range(N):
        while I < len(x) - 1:
            if (x[I] <= x_ends[i] <= x[I + 1]) or (x[I + 1] <= x_ends[i] <= x[I]):
                break
            else:
                I += 1
        a = (y[I + 1] - y[I]) / (x[I + 1] - x[I])
        b = y[I + 1] - a * x[I + 1]
        y_ends[i] = a * x_ends[i] + b
    y_ends[N] = y_ends[0]

    panels = empty(N, dtype=object)
    for i in range(N):
        panels[i] = Panel(x_ends[i], y_ends[i], x_ends[i + 1], y_ends[i + 1])

    return panels

def integral(x, y, panel, dxdk, dydk):
    """Computes the integral for the source and vortex contributions."""
    xa = panel.xa
    ya = panel.ya
    beta = panel.beta
    warnings.filterwarnings("ignore")
    def integrand(s):
        return (((x - (xa - sin(beta) * s)) * dxdk +
                (y - (ya + cos(beta) * s)) * dydk) /
                ((x - (xa - sin(beta) * s))**2 +
                (y - (ya + cos(beta) * s))**2) )
    return quad(integrand, 0.0, (panel.length+uniform(2e-20, 1e-20)), limit=int(5e4))[0]

def source_contribution_normal(panels):
    """Computes the source contribution to the normal velocity."""
    A = empty((panels.size, panels.size), dtype=float)
    fill_diagonal(A, 0.5)
    for i, panel_i in enumerate(panels):
        for j, panel_j in enumerate(panels):
            if i != j:
                A[i, j] = 0.5 / pi * integral(panel_i.xc, panel_i.yc,
                                                    panel_j,
                                                    cos(panel_i.beta),
                                                    sin(panel_i.beta))
    return A

def vortex_contribution_normal(panels):
    """Computes the vortex contribution to the normal velocity."""
    A = empty((panels.size, panels.size), dtype=float)
    fill_diagonal(A, 0.0)
    for i, panel_i in enumerate(panels):
        for j, panel_j in enumerate(panels):
            if i != j:
                A[i, j] = -0.5 / pi * integral(panel_i.xc, panel_i.yc,
                                                    panel_j,
                                                    sin(panel_i.beta),
                                                    -cos(panel_i.beta))
    return A

def kutta_condition(A_source, B_vortex):
    """Builds the Kutta condition array."""
    b = empty(A_source.shape[0] + 1, dtype=float)
    b[:-1] = B_vortex[0, :] + B_vortex[-1, :]
    b[-1] = - np.sum(A_source[0, :] + A_source[-1, :])
    return b

def build_singularity_matrix(A_source, B_vortex):
    """Builds the singularity matrix."""
    A = empty((A_source.shape[0] + 1, A_source.shape[1] + 1), dtype=float)
    A[:-1, :-1] = A_source
    A[:-1, -1] = np.sum(B_vortex, axis=1)
    A[-1, :] = kutta_condition(A_source, B_vortex)
    return A

def build_freestream_rhs(panels, freestream):
    """Builds the freestream right-hand side."""
    b = empty(panels.size + 1, dtype=float)
    for i, panel in enumerate(panels):
        b[i] = -freestream.u_inf * cos(freestream.alpha - panel.beta)
    b[-1] = -freestream.u_inf * (sin(freestream.alpha - panels[0].beta) +
                                sin(freestream.alpha - panels[-1].beta) )
    return b

def compute_tangential_velocity(panels, freestream, gamma, A_source, B_vortex):
    """Computes the tangential velocity on each panel."""
    A = empty((panels.size, panels.size + 1), dtype=float)
    A[:, :-1] = B_vortex
    A[:, -1] = -np.sum(A_source, axis=1)
    b = freestream.u_inf * sin([freestream.alpha - panel.beta
                                    for panel in panels])
    strengths = append([panel.sigma for panel in panels], gamma)
    tangential_velocities = dot(A, strengths) + b
    for i, panel in enumerate(panels):
        panel.vt = tangential_velocities[i]
    return panels

def compute_pressure_coefficient(panels, freestream):
    """Computes the pressure coefficient on each panel."""
    for panel in panels:
        panel.cp = 1.0 - (panel.vt / freestream.u_inf)**2
    return panels

def get_velocity_field(panels, freestream, X, Y):
    """Computes the velocity field on a mesh grid."""
    u = freestream.u_inf * cos(freestream.alpha) * ones_like(X, dtype=float)
    v = freestream.u_inf * sin(freestream.alpha) * ones_like(X, dtype=float)
    vec_intregral = vectorize(integral)
    for panel in panels:
        u += panel.sigma / (2.0 * pi) * vec_intregral(X, Y, panel, 1.0, 0.0)
        v += panel.sigma / (2.0 * pi) * vec_intregral(X, Y, panel, 0.0, 1.0)
    return u, v

def calculate_lift_coefficient(panels, freestream, gamma):
    """Calculates the lift coefficient."""
    c = abs(max(panel.xa for panel in panels) - min(panel.xa for panel in panels))
    return (gamma * sum(panel.length for panel in panels) / (0.5 * freestream.u_inf * c))

def calculate_drag_coefficient(panels):
    """Calculates the drag coefficient."""
    front_panels = [panel for panel in panels if panel.xc > 0]
    back_panels = [panel for panel in panels if panel.xc <= 0]
    front_cp = sum(panel.cp for panel in front_panels)
    back_cp = sum(panel.cp for panel in back_panels)
    return (front_cp - back_cp) / len(panels)

def run_simulation(x, y):
    """Runs the aerodynamic simulation."""
    panels = define_panels(x, y, N=80)
    freestream = Freestream(u_inf=1.0, alpha=4.0)

    A_source = source_contribution_normal(panels)
    B_vortex = vortex_contribution_normal(panels)

    A = build_singularity_matrix(A_source, B_vortex)
    b = build_freestream_rhs(panels, freestream)

    strengths = solve(A, b)

    for i, panel in enumerate(panels):
        panel.sigma = strengths[i]

    gamma = strengths[-1]

    panels = compute_tangential_velocity(panels, freestream, gamma, A_source, B_vortex)
    panels = compute_pressure_coefficient(panels, freestream)

    cl = calculate_lift_coefficient(panels, freestream, gamma)
    cd = calculate_drag_coefficient(panels)

    return cl, cd, panels, freestream
