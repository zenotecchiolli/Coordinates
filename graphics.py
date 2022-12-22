import matplotlib.pyplot as plt
import numpy as np
from auxiliary import unpacking
from functional import coordinates
from metric import jacobian
from matplotlib import cm


def plot_grid(N_rad, N_pol, N_tor, xx, yy, N_fp):

    """plots the 3d grid:
            Parameters
            ----------
            N_rad: int
                number of radial points
            N_pol: int
                number of poloidal points
            N_tor: int
                number of toroidal points
            xx: ndarray, shape(N_rad + 2, N_pol, N_tor)
                set of x points, x_ijk
            yy: ndarray, shape(N_rad + 2, N_pol, N_tor)
                set of y points, x_ijk
            N_fp: int
                number of field periods
            Returns
            -------
        """

    zeta = np.linspace(0, 2*np.pi/N_fp, N_tor + 1)

    for z in range(N_tor):
        for i in range(1, N_rad + 2):
            plt.plot(xx[i, :, z], yy[i, :, z], 'slategray')
            plt.plot([xx[i, N_pol - 1, z], xx[i, 0, z]], [yy[i, N_pol - 1, z], yy[i, 0, z]], 'slategray')
            if i == N_rad + 1:
                plt.plot(xx[i, :, z], yy[i, :, z], 'r-')
                plt.plot([xx[i, N_pol - 1, z], xx[i, 0, z]], [yy[i, N_pol - 1, z], yy[i, 0, z]], 'r-')

        for j in range(N_pol):
            plt.plot(xx[:, j, z], yy[:, j, z], 'slategray')

        plt.plot(xx[0, :, z], yy[0, :, z], 'bo')
        title = '$\zeta = %f$' %  zeta[z]
        plt.rcParams['font.size'] = '16'
        plt.title(title)
        plt.xlabel("$R$", fontsize=16)
        plt.ylabel("$Z$", fontsize=16)
        plt.show(bbox_inches='tight')

        plt.show()


def plot_out(xy_m, const):

    """plots the 3d grid associated to a set of dof:
        Parameters
        ----------
        xy_m: ndarray, shape(N_dof)
            vector with dof
        const: list of constants
            Returns
        -------
    """

    N_rad, N_pol, N_tor, n_tor, n_z, N_dofx, N_fp, omega, N_factor, s, f, dsdtdz, inv_dsdtdz, r, z, n_vec, z_vec, m_vec, R_mat, cos_mat, sen_mat, DX, DY, DX_bc, DY_bc = const
    out = unpacking(xy_m, n_tor, n_z, n_vec, z_vec, m_vec, N_dofx, r, z)
    x_mc = out[0]
    y_ms = out[1]

    topolino = coordinates(x_mc, y_ms, R_mat, cos_mat, sen_mat)
    xx = topolino[0]
    yy = topolino[1]

    plot_grid(N_rad, N_pol, N_tor, xx, yy, N_fp)


def plot_jacobian(xy_m, const, N_c):

    """plots the jacobian associated to a set of dof:
        Parameters
        ----------
        xy_m: ndarray, shape(N_dof)
            vector with dof
        const: list of constants
            Returns
        -------
    """

    N_rad, N_pol, N_tor, n_tor, n_z, N_dofx, N_fp, omega, N_factor, s, f, dsdtdz, inv_dsdtdz, r, z, n_vec, z_vec, m_vec, R_mat, cos_mat, sen_mat, DX, DY, DX_bc, DY_bc = const
    jac = jacobian(xy_m, const)
    s = np.linspace(0, 1, N_rad + 2)
    t = np.linspace(0, 2 * np.pi, N_pol + 1)
    M = np.zeros((N_rad + 2, N_pol))
    [S, T] = np.meshgrid(s, t[0:N_pol])

    for i in range(N_rad + 2):
        for j in range(N_pol):
            M[i][j] = jac[i][j][N_c]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.xlabel('$s$')
    plt.ylabel('$\\theta$')
    surf = ax.plot_surface(S, T, np.transpose(M), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('$\mathcal{J}$$(s, \\theta)$')
    plt.show()

    for j in range(N_pol):
        plt.plot(s, M[:, j])
    plt.title('$\mathcal{J}(s)$')
    plt.xlabel("$s$")
    plt.ylabel("$\mathcal{J}$")
    plt.show()
    for i in range(N_rad + 2):
        plt.plot(t[0:N_pol], M[i, :])
    plt.title('$\mathcal{J}$$(\\theta)$')
    plt.xlabel("$\\theta$")
    plt.ylabel("$\mathcal{J}$")
    plt.show()