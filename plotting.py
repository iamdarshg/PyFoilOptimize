import os
import numpy as np
from matplotlib import pyplot
from aerodynamics import get_velocity_field

def generate_plots(x, y, panels, freestream, cl, cd):
    """Generates and saves the simulation plots."""
    os.makedirs("assets", exist_ok=True)

    # Plot 1: Airfoil Shape
    pyplot.figure(figsize=(10, 6))
    pyplot.grid()
    pyplot.xlabel('x', fontsize=16)
    pyplot.ylabel('y', fontsize=16)
    pyplot.plot(x, y, color='k', linestyle='-', linewidth=2)
    pyplot.plot(np.append([panel.xa for panel in panels], panels[0].xa),
                np.append([panel.ya for panel in panels], panels[0].ya),
                linestyle='-', linewidth=1, marker='o', markersize=6, color='#CD2305')
    pyplot.axis('scaled')
    pyplot.xlim(-0.1, 1.1)
    pyplot.ylim(-0.1, 0.1)
    pyplot.savefig("assets/airfoil_shape.svg", dpi=300)

    # Plot 2: Pressure Coefficient
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
    pyplot.savefig("assets/pressure_coefficient.svg", dpi=300)

    # Plot 3: Streamlines
    nx, ny = 50, 50
    x_start, x_end = -1.5, 2.0
    y_start, y_end = -0.5, 0.5
    X, Y = np.meshgrid(np.linspace(x_start, x_end, nx), np.linspace(y_start, y_end, ny))
    u, v = get_velocity_field(panels, freestream, X, Y)

    pyplot.figure(figsize=(10, 6))
    pyplot.xlabel('x', fontsize=16)
    pyplot.ylabel('y', fontsize=16)
    pyplot.streamplot(X, Y, u, v,
                    density=1, linewidth=1, arrowsize=1, arrowstyle='->')
    pyplot.fill([panel.xc for panel in panels],
                [panel.yc for panel in panels],
                color='k', linestyle='solid', linewidth=2, zorder=2)
    pyplot.axis('scaled')
    pyplot.xlim(x_start, x_end)
    pyplot.ylim(y_start, y_end)
    pyplot.title('Streamlines around the airfoil', fontsize=16)
    pyplot.savefig("assets/streamlines.svg", dpi=300)
