import numpy as np
from auxiliary import unpacking
from auxiliary import counting
from auxiliary import R_mn_s
from auxiliary import Z_mn_s
from auxiliary import der_R_mn_s
from auxiliary import der_Z_mn_s
from auxiliary import j_spec


def set_initial_guess(n_z, n_tor):

    """Initialise the starting point x0.
        Parameters
        ----------
        n_z : int
            Zernike resolution
        n_tor:
            fourier toroidal resolution

        Returns
        -------
        xy_m: ndarray, shape(N_dof)
            vector with dof
        n_vec: ndarray, shape(N_dof)
            n_vec[i] is the toroidal number of xy_m[i]
        m_vec: ndarray, shape(N_dof)
            n_vec[i] is the poloidal number of xy_m[i]
        z_vec: ndarray, shape(N_dof)
            z_vec[i] is the zernike number of xy_m[i]
        N_dofx: int
           number of ndof for x_nlm
    """

    N_dofx, N_dofy = counting(n_z, n_tor)
    xy_m = np.zeros(N_dofx + N_dofy)
    n_vec = np.zeros(N_dofx + N_dofy)
    m_vec = np.zeros(N_dofx + N_dofy)
    z_vec = np.zeros(N_dofx + N_dofy)
    i_dofx = 0
    i_dofy = 0
    c = 0.0

    # looping on all the mode numbers
    for i_n in range(-n_tor, n_tor + 1):
        for i_z in range(n_z + 1):
            for i_m in range(n_z + 1):

                # condition for being an independent variable
                if i_m <= i_z and (i_z - i_m) % 2 == 0 and i_z != i_m:

                    # condition for x_0l0
                    if i_m == 0 and i_n == 0:
                        xy_m[i_dofx] = c
                        n_vec[i_dofx] = i_n + n_tor
                        m_vec[i_dofx] = i_m
                        z_vec[i_dofx] = i_z
                        i_dofx = i_dofx + 1

                    # condition for x_nlm and y_nlm
                    else:
                        xy_m[i_dofx] = c
                        n_vec[i_dofx] = i_n + n_tor
                        m_vec[i_dofx] = i_m
                        z_vec[i_dofx] = i_z
                        i_dofx = i_dofx + 1

                        xy_m[N_dofx + i_dofy] = c
                        n_vec[N_dofx + i_dofy] = i_n + n_tor
                        m_vec[N_dofx + i_dofy] = i_m
                        z_vec[N_dofx + i_dofy] = i_z
                        i_dofy = i_dofy + 1

    return xy_m, n_vec, m_vec, z_vec, N_dofx


def coordinates(x_mc, y_ms, R_mat, cos_mat, sen_mat):

    """Computes the Zernike-Fourier transform in real space.
            Parameters
            ----------
            x_mc: ndarray, shape(2N + 1, M + 1, M + 1)
                set of x_nlm
            y_ms: ndarray, shape(2N + 1, M + 1, M + 1)
                set of y_nlm
            R_mat: ndarray, shape(M + 1, M + 1)
                R_ilm, Zernike tensor of Zernike radial polinomials at s[i] for the mode l,m
            cos_mat: ndarray, shape(M + 1, M + 1)
                Cos_jknm, cosine tensor at theta[j] and zeta[k] for the modes n,m
            sen_mat: ndarray, shape(M + 1, M + 1)
                sen_jknm, sine tensor at theta[j] and zeta[k] for the modes n,m
            Returns
            -------
            xx: ndarray, shape(N_rad + 2, N_pol, N_tor)
                3d grid points x_ijk
            yy: ndarray, shape(N_rad + 2, N_pol, N_tor)

    """

    X = x_mc[None, None, None, :, :, :]
    Y = y_ms[None, None, None, :, :, :]
    R = R_mat[:, None, None, None, :, :]

    # transform from Zernike/Fourier space to real space
    xx = np.sum(X * R * cos_mat, axis=(3, 4, 5))
    yy = np.sum(Y * R * sen_mat, axis=(3, 4, 5))

    return xx, yy


def action(xy_m, const):

    """Computes the action for a set of dof.
            Parameters
            ----------
            xy_m: ndarray, shape(N_dof)
                vector with dof
            const: list of constants

            Returns
            -------
            out: float
                value of the action

    """

    N_rad, N_pol, N_tor, n_tor, n_z, N_dofx, N_fp, omega, N_factor, s, f, dsdtdz, inv_dsdtdz, r, z, n_vec, z_vec, m_vec, R_mat, cos_mat, sen_mat, DX, DY, DX_bc, DY_bc = const

    # compute the coefficients x_nlm y_nlm
    x_mc, y_ms = unpacking(xy_m, n_tor, n_z, n_vec, z_vec, m_vec, N_dofx, r, z)

    # compute the 3D grid x_ijk, y_ijk
    xx, yy = coordinates(x_mc, y_ms, R_mat, cos_mat, sen_mat)
    L = np.zeros((N_rad + 1, N_pol, N_tor))
    A = np.zeros((N_rad + 1, N_pol, N_tor))

    # take the radial weight excluding the boundary
    f = f[:-1, None, None]

    # apply boundary condition on the grid
    xx_bc = np.zeros((N_rad + 2, N_pol + 1, N_tor))
    yy_bc = np.zeros((N_rad + 2, N_pol + 1, N_tor))
    xx_bc[:, :-1, :] = xx[:, :, :]
    yy_bc[:, :-1, :] = yy[:, :, :]
    xx_bc[:, N_pol, :] = xx[:, 0, :]
    yy_bc[:, N_pol, :] = yy[:, 0, :]

    # compute the infinitesimal jacobian squared (A_ijk/dsdtdz)^2
    A[:, :, :] = (0.5*(np.abs((xx_bc[:-1, :-1, :] - xx_bc[1:, 1:, :]) * (yy_bc[:-1, 1:, :] - yy_bc[1:, :-1, :])
                            - (xx_bc[:-1, 1:, :] - xx_bc[1:, :-1, :]) * (yy_bc[:-1, :-1, :] - yy_bc[1:, 1:, :]))/ dsdtdz[:, :, :])) ** 2

    # compute the radial length L_ijk
    L[:, :, :] = omega * np.sqrt((xx[1:, :, :] - xx[:-1, :, :]) ** 2 + (yy[1:, :, :] - yy[:-1, :, :]) ** 2)

    # sum over the 3d grid for getting the action
    S = (0.5 * f * A * dsdtdz + omega * L * dsdtdz)*(1/N_factor)
    out = np.sum(S)
    print(out)
    return out


def out_quantities(xy_m, const, out_name_SPEC):

    """Computes the set of SPEC modes and write the into out_name_SPEC.
                Parameters
                ----------
                xy_m: ndarray, shape(N_dof)
                    vector with dof
                const: list of constants

                out_name_SPEC: basestring
                    string with file name for SPEC modes
                Returns
                -------

        """


    N_rad, N_pol, N_tor, n_tor, n_z, N_dofx, N_fp, omega, N_factor, s, f, dsdtdz, inv_dsdtdz, r, z, n_vec, z_vec, m_vec, R_mat, cos_mat, sen_mat, DX, DY, DX_bc, DY_bc = const

    # compute coefficients
    [x_mc, y_ms] = unpacking(xy_m, n_tor, n_z, n_vec, z_vec, m_vec, N_dofx, r, z)

    # SPEC number of modes
    mz = (2 * n_tor + 1) * (n_z + 1) - n_tor

    m_spec = np.zeros(mz)
    n_spec = np.zeros(mz)
    R_mat = np.zeros((N_rad + 2, mz))
    der_R_mat = np.zeros((N_rad + 2, mz))
    open(out_name_SPEC, 'w').close()

    # looping over every spatial point
    for i_s in range(1, N_rad + 2):
        file = open(out_name_SPEC, 'a+')
        R_vec = np.zeros(mz)
        Z_vec = np.zeros(mz)
        der_R_vec = np.zeros(mz)
        der_Z_vec = np.zeros(mz)

        # for every mode n,m
        for i_n in range(- n_tor, n_tor + 1):
            for i_m in range(n_z + 1):

                # using spec convention n>= 0
                if i_m == 0 and i_n > 0:

                    # Rn0_spec = R_n0 + R_-n0
                    R_vec[j_spec(i_n, i_m, n_tor)] = R_mn_s(i_s, i_n, i_m, x_mc, n_tor, n_z, s) + R_mn_s(i_s, - i_n, i_m, x_mc, n_tor, n_z, s)

                    # Zn0_spec = Z_n0 - Z_-n0
                    Z_vec[j_spec(i_n, i_m, n_tor)] = Z_mn_s(i_s, i_n, i_m, y_ms, n_tor, n_z, s) - Z_mn_s(i_s, - i_n, i_m, y_ms, n_tor, n_z, s)
                    R_mat[i_s, j_spec(i_n, i_m, n_tor)] = R_vec[j_spec(i_n, i_m, n_tor)]

                    #  der_Rn0_spec = der_R_n0 + der_R_-n0
                    der_R_vec[j_spec(i_n, i_m, n_tor)] = der_R_mn_s(i_s, i_n, i_m, x_mc, n_tor, n_z, s) + der_R_mn_s(i_s, - i_n, i_m, x_mc, n_tor, n_z, s)

                    #  der_Zn0_spec = der_Z_n0 - der_Z_-n0
                    der_Z_vec[j_spec(i_n, i_m, n_tor)] = der_Z_mn_s(i_s, i_n, i_m, y_ms, n_tor, n_z, s) - der_Z_mn_s(i_s, - i_n, i_m, y_ms, n_tor, n_z, s)
                    der_R_mat[i_s, j_spec(i_n, i_m, n_tor)] = der_R_vec[j_spec(i_n, i_m, n_tor)]
                    m_spec[j_spec(i_n, i_m, n_tor)] = i_m  # mode vectors with spec convention
                    n_spec[j_spec(i_n, i_m, n_tor)] = i_n  # mode vectors with spec convention

                elif i_m == 0 and i_n < 0:
                    pass
                else:

                    # Rnm_spec = R_nm
                    R_vec[j_spec(i_n, i_m, n_tor)] = R_mn_s(i_s, i_n, i_m, x_mc, n_tor, n_z, s)

                    # Rnm_spec = Z_nm
                    Z_vec[j_spec(i_n, i_m, n_tor)] = Z_mn_s(i_s, i_n, i_m, y_ms, n_tor, n_z, s)
                    R_mat[i_s, j_spec(i_n, i_m, n_tor)] = R_vec[j_spec(i_n, i_m, n_tor)]

                    # der_Rnm_spec = der_R_nm
                    der_R_vec[j_spec(i_n, i_m, n_tor)] = der_R_mn_s(i_s, i_n, i_m, x_mc, n_tor, n_z, s)

                    # der_Znm_spec = der_Z_nm
                    der_Z_vec[j_spec(i_n, i_m, n_tor)] = der_Z_mn_s(i_s, i_n, i_m, y_ms, n_tor, n_z, s)
                    der_R_mat[i_s, j_spec(i_n, i_m, n_tor)] = der_R_vec[j_spec(i_n, i_m, n_tor)]
                    m_spec[j_spec(i_n, i_m, n_tor)] = i_m
                    n_spec[j_spec(i_n, i_m, n_tor)] = i_n
        stringa = ''
        for i_j in range(mz):
            aus = '%E \t' % R_vec[i_j]  # first line Rnm[i]
            stringa = stringa + aus
        file.write(stringa + '\n')
        stringa = ''
        for i_j in range(mz):
            aus = '%E \t' % Z_vec[i_j]  # second line Znm[i]
            stringa = stringa + aus
        file.write(stringa + '\n')
        stringa = ''
        for i_j in range(mz):
            aus = '%E \t' % der_R_vec[i_j]  # third line der_Rnm[i]
            stringa = stringa + aus
        file.write(stringa + '\n')
        stringa = ''
        for i_j in range(mz):
            aus = '%E \t' % der_Z_vec[i_j]   # fourth line line der_Rnm[i]
            stringa = stringa + aus
        file.write(stringa + '\n')
        file.close()


def out_vec(xy_m, out_file):

    """Save in a text file the dof vector.
                Parameters
                ----------
                xy_m: ndarray, shape(N_dof)
                    vector with dof
                const: list of constants

                out_name: basestring
                    string with file name for output of xy_m
                Returns
                -------

        """

    open(out_file, 'w').close()
    for i in range(len(xy_m)):

        # write xy_m as a column
        string = '%E\n' % (xy_m[i])
        file = open(out_file, 'a+')
        file.write(string)
        file.close()
