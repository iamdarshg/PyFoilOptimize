import os
import numpy as np
from numpy.random import uniform
from numpy import meshgrid, linspace, sqrt, pi, arccos, cos, empty_like, copy, sin, append, radians, empty, fill_diagonal, dot, vectorize, ones_like, ones, loadtxt
from scipy.integrate import quad
from numpy.linalg import solve
from matplotlib import pyplot
from time import time
naca_filepath = os.path.join('0012.txt')
width = 10
from time import time, sleep
import scipy
alfa = 10
from numba import jit
from multiprocessing import Process, Queue


def plots(x,res):  
    nx, ny = 20, 20
    y = res.x
    try:
        np.savetxt("00121.txt", y)
    except:
        print("txt failed")
    y = (y+uniform(1e-20, 1e-21))
    q1 = Queue()
    q2 = Queue()
    q3 = Queue()
    p1 = Process(target = define_panels, args=(q1, x, y, 70))
    p1.start()
    freestream = Freestream(u_inf=1.0, alpha=4.0)
    panels = q1.get()
    p1.join()
    p2 = Process(target = source_contribution_normal, args=(panels, q2))
    p2.start()
    p3 = Process(target = vortex_contribution_normal, args=(panels, q3))
    p3.start()
    while q2.empty() or q3.empty():
        sleep(1e-6)
    A_source = q2.get()
    B_vortex = q3.get()
    p2.join()
    p3.join()
    while not q1.empty():
        q1.get()
    while not q2.empty():
        q2.get()
    A = build_singularity_matrix(A_source, B_vortex)
    b = build_freestream_rhs(panels, freestream)

    width = 10
    pyplot.figure(figsize=(width, width))
    pyplot.grid()
    pyplot.xlabel('x', fontsize=16)
    pyplot.ylabel('y', fontsize=16)
    pyplot.plot(x, y, color='k', linestyle='-', linewidth=2)
    pyplot.plot(append([panel.xa for panel in panels], panels[0].xa),
                append([panel.ya for panel in panels], panels[0].ya),
                linestyle='-', linewidth=1, marker='o', markersize=6, color='#CD2305')
    pyplot.axis('scaled')
    pyplot.xlim(-0.1, 1.1)
    pyplot.ylim(-0.1, 0.1)

    strengths = solve(A, b)

    for i , panel in enumerate(panels):
        panel.sigma = strengths[i]

    gamma = strengths[-1]

    compute_tangential_velocity(panels, freestream, gamma, A_source, B_vortex)

    compute_pressure_coefficient(panels, freestream)
    c = abs(max(panel.xa for panel in panels) -
        min(panel.xa for panel in panels))
    cl = (gamma * sum(panel.length for panel in panels) /
        (0.5 * freestream.u_inf * c))
    pyplot.figure(figsize=(10, 6))
    pyplot.grid()
    pyplot.xlabel('$x$', fontsize=16)
    pyplot.ylabel('$C_p$', fontsize=16)
    pyplot.plot([panel.xc for panel in panels if panel.loc == 'upper'],
                [panel.cp for panel in panels if panel.loc == 'upper'],
                label='upper surface',
                color='r', linestyle='-', linewidth=2, marker='o', markersize=6)
    pyplot.plot([panel.xc for panel in panels if panel.loc == 'lower'],
                [panel.cp for panel in panels if panel.loc == 'lower'],
                label= 'lower surface',
                color='b', linestyle='-', linewidth=1, marker='o', markersize=6)
    pyplot.legend(loc='best', prop={'size':16})
    pyplot.xlim(-0.1, 1.1)
    pyplot.ylim(1.0, -2.0)
    pyplot.title(f'Number of panels: {panels.size}', fontsize=16);
    pyplot.savefig("test2.pdf", dpi=250)

    accuracy = sum(panel.sigma * panel.length for panel in panels)
    print('sum of singularity strengths: {:0.6f}'.format(accuracy))


    x_start, x_end = -1.5, 2.0
    y_start, y_end = -0.5, 0.5
    X, Y = meshgrid(linspace(x_start, x_end, nx), linspace(y_start, y_end, ny))

    # compute the velocity field on the mesh grid
    u, v = get_velocity_field(panels, freestream, X, Y)
    width = 10
    pyplot.figure(figsize=(width, width))
    pyplot.xlabel('x', fontsize=16)
    pyplot.ylabel('y', fontsize=16)
    pyplot.streamplot(X, Y, u, v,
                    density=1, linewidth=1, arrowsize=1, arrowstyle='->')
    pyplot.fill([panel.xc for panel in panels],
                [panel.yc for panel in panels],
                color='k', linestyle='solid', linewidth=2, zorder=2)
    pyplot.axis('scaled', )
    pyplot.xlim(x_start, x_end)
    pyplot.ylim(y_start, y_end)
    pyplot.title(
        f'Streamlines around a NACA 0012 airfoil (AoA = ${alfa}^o$)',
        fontsize=16,
    )
    pyplot.savefig("test1.pdf", dpi=250)   
    # compute the pressure field
    cp = 1.0 - (u**2 + v**2) / freestream.u_inf**2

    # plot the pressure field
    width = 10
    pyplot.figure(figsize=(width, width))
    pyplot.xlabel('x', fontsize=16)
    pyplot.ylabel('y', fontsize=16)
    contf = pyplot.contourf(X, Y, cp,
                            levels=linspace(-2.0, 1.0, 100), extend='both', linestyles= 'solid', algorithm= 'threaded', )
    cbar = pyplot.colorbar(contf,
                        orientation='horizontal',
                        shrink=0.5, pad = 0.1,
                        ticks=[-2.0, -1.0, 0.0, 1.0])
    cbar.set_label('$C_p$', fontsize=16)
    pyplot.fill([panel.xc for panel in panels],
                [panel.yc for panel in panels],
                color='k', linestyle='solid', linewidth=2, zorder=2)
    pyplot.axis('scaled', )
    pyplot.xlim(x_start, x_end)
    pyplot.ylim(y_start, y_end)
    pyplot.title('Contour of pressure field', fontsize=16)
    cd= drag(panels)
    print('lift coefficient: CL = {:0.3f}'.format(cl))
    print(f"drag coefficient= {cd}")
    pyplot.savefig("test.pdf", dpi=250)
    pyplot.show()

def for_par(x,y, itera):
    
    baounds = scipy.optimize.Bounds(lb=(-ones(y.shape)*2), ub=(ones(y.shape)*2), keep_feasible=False)
    return scipy.optimize.minimize(
        x0=y,
        fun=main,
        method='Nelder-Mead',
        options={
            'maxiter': int(itera),
        },
        bounds=baounds,
    )
    
class Panel():
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
    
def define_panels(q1, x, y, N=40,):  
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
    
    q1.put(panels)

class Freestream:
    def __init__(self, u_inf=int(3.43e3), alpha=alfa):
        self.u_inf = u_inf
        self.alpha = radians(alpha)  # degrees to radians

def integral(x, y, panel, dxdk, dydk):
    xa = panel.xa
    ya = panel.ya
    beta = panel.beta
    # @jit(nopython = True, fastmath = True)
    def integrand(s):
        return (((x - (xa - sin(beta) * s)) * dxdk +
                (y - (ya + cos(beta) * s)) * dydk) /
                ((x - (xa - sin(beta) * s))**2 +
                (y - (ya + cos(beta) * s))**2) )
    return quad(integrand, 0.0, (panel.length+uniform(2e-20, 1e-20)), limit=int(5e2))[0]

def source_contribution_normal(panels, q2):
    A = empty((panels.size, panels.size), dtype=float)

    fill_diagonal(A, 0.5)
    x=0
    for i, panel_i in enumerate(panels):
        A = xaye(panels, i, panel_i, A)
    q2.put(A)

def xaye(panels, i, panel_i, A):
    for j, panel_j in enumerate(panels):
        if i != j:
            A[i, j] = 0.5 / pi * integral(panel_i.xc, panel_i.yc, 
                                                panel_j,
                                                cos(panel_i.beta),
                                                sin(panel_i.beta))
    return A

def vortex_contribution_normal(panels, q3):

    A = empty((panels.size, panels.size), dtype=float)
    # vortex contribution on a panel from itself
    fill_diagonal(A, 0.0)
    # vortex contribution on a panel from others
    for i, panel_i in enumerate(panels):
        A = xxye(A, i, panels, panel_i)
    q3.put(A)

def xxye(A, i, panels, panel_i):
    for j, panel_j in enumerate(panels):
        if i != j:
            A[i, j] = -0.5 / pi * integral(panel_i.xc, panel_i.yc, 
                                                panel_j,
                                                sin(panel_i.beta),
                                                -cos(panel_i.beta)) 
    return A
                                
def kutta_condition(A_source, B_vortex):
    b = empty(A_source.shape[0] + 1, dtype=float)
    b[:-1] = B_vortex[0, :] + B_vortex[-1, :]
    b[-1] = - np.sum(A_source[0, :] + A_source[-1, :])
    return b

def build_singularity_matrix(A_source, B_vortex):
    A = empty((A_source.shape[0] + 1, A_source.shape[1] + 1), dtype=float)
    # source contribution matrix
    A[:-1, :-1] = A_source
    # vortex contribution array
    A[:-1, -1] = np.sum(B_vortex, axis=1)
    # Kutta condition array
    A[-1, :] = kutta_condition(A_source, B_vortex)
    return A

def build_freestream_rhs(panels, freestream):
    b = empty(panels.size + 1, dtype=float)
    for i, panel in enumerate(panels):
        b[i] = -freestream.u_inf * cos(freestream.alpha - panel.beta)
    b[-1] = -freestream.u_inf * (sin(freestream.alpha - panels[0].beta) +
                                sin(freestream.alpha - panels[-1].beta) )
    return b

def compute_tangential_velocity(panels, freestream, gamma, A_source, B_vortex):
    A = empty((panels.size, panels.size + 1), dtype=float)
    A[:, :-1] = B_vortex
    A[:, -1] = -np.sum(A_source, axis=1)

    b = freestream.u_inf * sin([freestream.alpha - panel.beta 
                                    for panel in panels])
    
    strengths = append([panel.sigma for panel in panels], gamma)
    
    tangential_velocities = dot(A, strengths) + b
    
    for i, panel in enumerate(panels):
        panel.vt = tangential_velocities[i]

def compute_pressure_coefficient(panels, freestream):
    for panel in panels:
        panel.cp = 1.0 - (panel.vt / freestream.u_inf)**2

def get_velocity_field(panels, freestream, X, Y):
    u = freestream.u_inf * cos(freestream.alpha) * ones_like(X, dtype=float)
    v = freestream.u_inf * sin(freestream.alpha) * ones_like(X, dtype=float)
    vec_intregral = vectorize(integral)
    for panel in panels:
        u += panel.sigma / (2.0 * pi) * vec_intregral(X, Y, panel, 1.0, 0.0)
        v += panel.sigma / (2.0 * pi) * vec_intregral(X, Y, panel, 0.0, 1.0)
    
    return u, v

def test(panels):
    return max(panel.xa for panel in panels) - min(panel.xa for panel in panels)

def ctest(gamma, panel, freestream, c, panels):
    return (gamma * check(panel=panel, panels=panels) /(0.5 * freestream * c))

def check(panel, panels):
    out=0
    for panel in panels:
        out=out+ sqrt((panel.xb - panel.xa)**2 + (panel.yb - panel.ya)**2) 
    return out

def main(y):
    q1 = Queue()
    q2 = Queue()
    q3 = Queue()
    p1 = Process(target = define_panels, args=(q1, x, y, 70))
    p1.start()
    freestream = Freestream(u_inf=1.0, alpha=4.0)
    panels = q1.get()
    p1.join()
    p2 = Process(target = source_contribution_normal, args=(panels, q2))
    p2.start()
    p3 = Process(target = vortex_contribution_normal, args=(panels, q3))
    p3.start()
    while q2.empty() or q3.empty():
        sleep(1e-6)
    A_source = q2.get()
    B_vortex = q3.get()
    p2.join()
    p3.join()
    while not q1.empty():
        q1.get()
    while not q2.empty():
        q2.get()
    A = build_singularity_matrix(A_source, B_vortex)
    b = build_freestream_rhs(panels, freestream)

    strengths = solve(A, b)

    for i , panel in enumerate(panels):
        panel.sigma = strengths[i]

    gamma = strengths[-1]

    compute_tangential_velocity(panels, freestream, gamma, A_source, B_vortex)
    compute_pressure_coefficient(panels, freestream)
    c = test(panels=panels)
    cl= ctest(gamma=gamma, panels=panels, freestream=freestream.u_inf, c=c, panel = panel)

    return 1e6-(cl**5)

def drag(panels):
    frontcp = sum([panel.cp for panel in panels if panel.xb > 0.5])
    backcp = sum([panel.cp for panel in panels if panel.xb <= 0.5])
    return (frontcp-backcp)


if __name__ == '__main__':
    global x
    starttime = time()
    with open(naca_filepath, 'r') as infile:
        x, y = loadtxt(infile, dtype=float, unpack=True)
    y = loadtxt(open('00121.txt', 'r'), dtype = float, unpack = True)
    class oof():
        x=y
    plots(x=x, res = oof)
    plots(x, res = for_par(x,y, itera=1e2))
    endtime = time()
    print(f"Time take = {round((endtime-starttime)/60, 6)}")
