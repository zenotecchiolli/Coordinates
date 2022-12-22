import numpy as np
from auxiliary import unpacking
from auxiliary import Rz
from auxiliary import der_Rz


def jacobian(xy_m, const):

    """computes the jacobian associated to a set of dof:
        Parameters
        ----------
        xy_m: ndarray, shape(N_dof)
            vector with dof
        const: list of constants
            Returns
            -------
        jac: ndarray, shape(N_rad + 2, N_pol, N_tor)
            jacobian over the grid points
    """

    N_rad, N_pol, N_tor, n_tor, n_z, N_dofx, N_fp, omega, N_factor, s, f, dsdtdz, inv_dsdtdz, r, z, n_vec, z_vec, m_vec, R_mat, cos_mat, sen_mat, DX, DY, DX_bc, DY_bc = const

    e_s = np.zeros((N_rad + 2, N_pol, N_tor, 3))
    e_t = np.zeros((N_rad + 2, N_pol, N_tor, 3))
    e_z = np.zeros((N_rad + 2, N_pol, N_tor, 3))
    jac = np.zeros((N_rad + 2, N_pol, N_tor))
    t = np.linspace(0, 2 * np.pi, N_pol + 1)
    zeta = np.linspace(0, 2 * np.pi, N_tor + 1)
    s = np.linspace(0, 1, N_rad + 2)
    [x_mc, y_ms] = unpacking(xy_m, n_tor, n_z, n_vec, z_vec, m_vec, N_dofx, r, z)
    aus_s = np.zeros(3)
    aus_t = np.zeros(3)
    aus_z = np.zeros(3)

    # computing the basis vectors from the mapping
    for k in range(N_tor):
        for i in range(N_rad + 2):
            for j in range(N_pol):
                for p in range(3):
                    aus_s[p] = 0.0
                    aus_t[p] = 0.0
                    aus_z[p] = 0.0

                for i_n in range(- n_tor, n_tor + 1):
                    for i_z in range(n_z + 1):
                        for i_m in range(n_z):

                            # adding only contributions for the non-zero x_nlm and y_nlm
                            if i_m <= i_z and (i_z - i_m) % 2 == 0:

                                aus_s[0] = aus_s[0] + x_mc[i_n + n_tor, i_z, i_m] * der_Rz(i_z, i_m, s[i]) * np.cos(i_m * t[j] - N_fp*i_n * zeta[k]) * np.cos(zeta[k])
                                aus_s[1] = aus_s[1] + x_mc[i_n + n_tor, i_z, i_m] * der_Rz(i_z, i_m, s[i]) * np.cos(i_m * t[j] - N_fp*i_n * zeta[k]) * np.sin(zeta[k])
                                aus_s[2] = aus_s[2] + y_ms[i_n + n_tor, i_z, i_m] * der_Rz(i_z, i_m, s[i]) * np.sin(i_m * t[j] - N_fp*i_n * zeta[k])

                                aus_t[0] = aus_t[0] + x_mc[i_n + n_tor, i_z, i_m] * Rz(i_z, i_m, s[i]) * (- i_m) * np.sin(i_m * t[j] - N_fp * i_n * zeta[k]) * np.cos(zeta[k])
                                aus_t[1] = aus_t[1] + x_mc[i_n + n_tor, i_z, i_m] * Rz(i_z, i_m, s[i]) * (- i_m) * np.sin(i_m * t[j] - N_fp * i_n * zeta[k]) * np.sin(zeta[k])
                                aus_t[2] = aus_t[2] + y_ms[i_n + n_tor, i_z, i_m] * Rz(i_z, i_m, s[i]) * i_m * np.cos(i_m * t[j] - N_fp * i_n * zeta[k])

                                aus_z[0] = aus_z[0] + x_mc[i_n + n_tor, i_z, i_m] * Rz(i_z, i_m, s[i]) * (i_n * N_fp * np.sin(i_m * t[j] - N_fp * i_n * zeta[k]) * np.cos(zeta[k]) - np.cos(i_m * t[j] - N_fp * i_n * zeta[k]) * np.sin(zeta[k]))
                                aus_z[1] = aus_z[1] + x_mc[i_n + n_tor, i_z, i_m] * Rz(i_z, i_m, s[i]) * (i_n * N_fp * np.sin(i_m * t[j] - N_fp * i_n * zeta[k]) * np.cos(zeta[k]) + np.cos(i_m * t[j] - N_fp * i_n * zeta[k]) * np.cos(zeta[k]))
                                aus_z[2] = aus_z[2] + y_ms[i_n + n_tor, i_z, i_m] * Rz(i_z, i_m, s[i]) * (- N_fp * i_n) * np.cos(i_m * t[j] - N_fp * i_n * zeta[k])

                for p in range(3):
                    e_s[i][j][k][p] = aus_s[p]
                    e_t[i][j][k][p] = aus_t[p]
                    e_z[i][j][k][p] = aus_z[p]

    # computing the jacobian as e_s.(e_t x e_z)
    for k in range(N_tor):
        for i in range(N_rad + 2):
            for j in range(N_pol):
                jac[i][j][k] = np.abs(np.dot(e_s[i][j][k], np.cross(e_t[i][j][k], e_z[i][j][k])))

    return jac