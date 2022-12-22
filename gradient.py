from auxiliary import unpacking
from functional import coordinates
import numpy as np
from functional import action


def finite_difference(xy_m, const):

    """Computes the gradient vector using final differences.
                Parameters
                ----------
                xy_m: ndarray, shape(N_dof)
                    vector with dof
                const: ndarray
                    list of constants
                Returns
                -------
                der: ndarray shape(N_dof)
                    gradient vector
    """


    N_rad, N_pol, N_tor, n_tor, n_z, N_dofx, N_fp, omega, N_factor, s, f, dsdtdz, inv_dsdtdz, r, z, n_vec, z_vec, m_vec, R_mat, cos_mat, sen_mat, DX, DY, DX_bc, DY_bc = const
    dm = 1E-5
    N_dof = len(n_vec)
    der = np.zeros(N_dof)
    vec_p = np.zeros(N_dof)
    vec_m = np.zeros(N_dof)
    vec_out_plus = np.zeros(N_dof)
    vec_out_min = np.zeros(N_dof)

    # initialise plus vector
    for i in range(N_dof):
        vec_p[i] = xy_m[i]

    # initialise minus vector
    for i in range(N_dof):
        vec_m[i] = xy_m[i]

    for i in range(N_dof):

        # create forward and backward vectors
        vec_p[i] = vec_p[i] + dm
        vec_m[i] = vec_m[i] - dm
        f_p = action(vec_p, const)
        vec_out_plus[i] = f_p
        f_m = action(vec_m, const)
        vec_out_min[i] = f_m

        # compute final difference
        der[i] = (f_p - f_m)/(2*dm)
        for j in range(N_dof):
            vec_p[j] = xy_m[j]
            vec_m[j] = xy_m[j]

    return der


def gradient_action(xy_m, const):

    """Computes the gradient vector using the explicit formula.
                Parameters
                ----------
                xy_m: ndarray, shape(N_dof)
                    vector with dof
                const: ndarray
                    list of constants
                Returns
                -------
                der: ndarray shape(N_dof)
                    gradient vector
        """

    N_rad, N_pol, N_tor, n_tor, n_z, N_dofx, N_fp, omega, N_factor, s, f, dsdtdz, inv_dsdtdz, r, z, n_vec, z_vec, m_vec, R_mat, cos_mat, sen_mat, DX, DY, DX_bc, DY_bc = const

    # get x_nlm and y_nlm coefficients
    x_mc, y_ms = unpacking(xy_m, n_tor, n_z, n_vec, z_vec, m_vec, N_dofx, r, z)

    # compute the 3d grid
    xx, yy = coordinates(x_mc, y_ms, R_mat, cos_mat, sen_mat)

    # initialise volume elements
    dsdtdz = dsdtdz[:, :, :, None, None, None]
    inv_dsdtdz = inv_dsdtdz[:, :, :, None, None, None]

    # apply boundary conditions on the grid

    xx_bc = np.zeros((N_rad + 2, N_pol + 1, N_tor))
    yy_bc = np.zeros((N_rad + 2, N_pol + 1, N_tor))
    xx_bc[:, :-1, :] = xx[:, :, :]
    yy_bc[:, :-1, :] = yy[:, :, :]
    xx_bc[:, N_pol, :] = xx[:, 0, :]
    yy_bc[:, N_pol, :] = yy[:, 0, :]

    # computing areas and F

    Area = np.zeros((N_rad + 1, N_pol, N_tor))
    sign_Area = np.zeros((N_rad + 1, N_pol, N_tor))
    Area[:, :, :] = 0.5*np.abs((xx_bc[:-1, :-1, :] - xx_bc[1:, 1:, :]) * (yy_bc[:-1, 1:, :] - yy_bc[1:, :-1, :])
                            - (xx_bc[:-1, 1:, :] - xx_bc[1:, :-1, :]) * (yy_bc[:-1, :-1, :] - yy_bc[1:, 1:, :]))
    sign_Area[:, :, :] = np.sign((xx_bc[:-1, :-1, :] - xx_bc[1:, 1:, :]) * (yy_bc[:-1, 1:, :] - yy_bc[1:, :-1, :])
                            - (xx_bc[:-1, 1:, :] - xx_bc[1:, :-1, :]) * (yy_bc[:-1, :-1, :] - yy_bc[1:, 1:, :]))
    A = Area[:, :, :, None, None, None]
    S_A = sign_Area[:, :, :, None, None, None]
    F = f[:-1, None, None, None, None, None]

    # computing expression for areas for the derivative respect to x_nlm

    diff_y_1 = np.zeros((N_rad + 1, N_pol, N_tor))
    diff_y_2 = np.zeros((N_rad + 1, N_pol, N_tor))
    diff_y_1[:, :, :] = yy_bc[:-1, 1:, :] - yy_bc[1:, :-1, :]
    diff_y_2[:, :, :] = yy_bc[:-1, :-1, :] - yy_bc[1:, 1:, :]

    DiffY1 = diff_y_1[:, :, :, None, None, None]
    DiffY2 = diff_y_2[:, :, :, None, None, None]

    DX_der1 = np.zeros((N_rad + 1, N_pol, N_tor, 2*n_tor + 1, n_z + 1, n_z + 1))
    DX_der2 = np.zeros((N_rad + 1, N_pol, N_tor, 2*n_tor + 1, n_z + 1, n_z + 1))

    DX_der1[:, :, :, :, :, :] = DX_bc[:-1, :-1, :, :, :, :] - DX_bc[1:, 1:, :, :, :, :]
    DX_der2[:, :, :, :, :, :] = DX_bc[:-1, 1:, :, :, :, :] - DX_bc[1:, :-1, :, :, :, :]

    # computing expression for areas for the derivative respect to y_nlm

    diff_x_1 = np.zeros((N_rad + 1, N_pol, N_tor))
    diff_x_2 = np.zeros((N_rad + 1, N_pol, N_tor))

    diff_x_1[:, :, :] = xx_bc[:-1, :-1, :] - xx_bc[1:, 1:, :]
    diff_x_2[:, :, :] = xx_bc[:-1, 1:, :] - xx_bc[1:, :-1, :]

    DiffX1 = diff_x_1[:, :, :, None, None, None]
    DiffX2 = diff_x_2[:, :, :, None, None, None]

    DY_der1 = np.zeros((N_rad + 1, N_pol, N_tor, 2 * n_tor + 1, n_z + 1, n_z + 1))
    DY_der2 = np.zeros((N_rad + 1, N_pol, N_tor, 2 * n_tor + 1, n_z + 1, n_z + 1))

    DY_der1[:, :, :, :, :, :] = DY_bc[:-1, 1:, :, :, :, :] - DY_bc[1:, :-1, :, :, :, :]
    DY_der2[:, :, :, :, :, :] = DY_bc[1:, 1:, :, :, :, :] - DY_bc[:-1, :-1, :, :, :, :]

    # computing the expression for length

    aus_l_X = np.zeros((N_rad + 1, N_pol, N_tor))
    aus_l_Y = np.zeros((N_rad + 1, N_pol, N_tor))
    der_LX = np.zeros((N_rad + 1, N_pol, N_tor, 2 * n_tor + 1, n_z + 1, n_z + 1))
    der_LY = np.zeros((N_rad + 1, N_pol, N_tor, 2 * n_tor + 1, n_z + 1, n_z + 1))
    aus_l_X[:, :, :] = ((xx[1:, :, :] - xx[:-1, :, :]) / np.sqrt((xx[1:, :, :] - xx[:-1, :, ]) ** 2 + (yy[1:, :, :] - yy[:-1, :, :]) ** 2))
    aus_l_Y[:, :, :] = ((yy[1:, :, :] - yy[:-1, :, :]) / np.sqrt((xx[1:, :, :] - xx[:-1, :, ]) ** 2 + (yy[1:, :, :] - yy[:-1, :, :]) ** 2))

    Aus_LX = aus_l_X[:, :, :, None, None, None]
    Aus_LY = aus_l_Y[:, :, :, None, None, None]

    der_LX[:, :, :, :, :, :] = DX[1:, :, :, :, :, :] - DX[:-1, :, :, :, :, :]
    der_LY[:, :, :, :, :, :] = DY[1:, :, :, :, :, :] - DY[:-1, :, :, :, :, :]

    # packing everything together and summing over ijk

    SX_mode = np.sum(0.5 * A * F * S_A * (DiffY1 * DX_der1 - DiffY2 * DX_der2) * inv_dsdtdz + omega * Aus_LX * der_LX * dsdtdz, axis=(0, 1, 2))
    SY_mode = np.sum(0.5 * A * F * S_A * (DiffX1 * DY_der1 + DiffX2 * DY_der2) * inv_dsdtdz + omega * Aus_LY * der_LY * dsdtdz, axis=(0, 1, 2))

    # taking only the term relative to the degrees of freedom

    der = np.zeros(len(n_vec))
    n_vec = n_vec.astype(int)
    z_vec = z_vec.astype(int)
    m_vec = m_vec.astype(int)
    der[:N_dofx] = SX_mode[n_vec[:N_dofx], z_vec[:N_dofx], m_vec[:N_dofx]]/N_factor
    der[N_dofx:] = SY_mode[n_vec[N_dofx:], z_vec[N_dofx:], m_vec[N_dofx:]]/N_factor

    return der

