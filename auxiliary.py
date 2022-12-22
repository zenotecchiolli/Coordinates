import numpy as np
import math


def counting(n_z, n_tor):

    """Counts the ndof for x_nlm and y_nlm.
                Parameters
                ----------
                n_z: int
                    Zernike resolution
                n_tor: int
                    toroidal fourier resolution
                Returns
                -------
                N_dofx: int
                    degrees of freedom for x_nlm
                N_dofy: int
                    degrees of freedom for y_nlm
        """


    N_z = n_z + 1
    mz = (2 * n_tor + 1) * (n_z + 1)
    N_dofx = 0
    N_dofy = 0

    # checks every possible mode
    for i_z in range(N_z):
        i_mz = n_tor * (n_z + 1)

        # condition for x_0l0
        if (i_z - m(i_mz, n_z)) % 2 == 0 and i_z != 0:
            N_dofx = N_dofx + 1
        for i_mz in range(mz):

            # condition for x_nlm and y_nlm
            if m(i_mz, n_z) < i_z and i_mz != n_tor * (n_z + 1) and (i_z - m(i_mz, n_z)) % 2 == 0:
                N_dofx = N_dofx + 1
                N_dofy = N_dofy + 1

    return N_dofx, N_dofy


def n(p, m_pol, n_tor):

    """gives the toroidal number corresponding to the vectorized index p.
                Parameters
                ----------
                p: int
                    vectorized boundary index
                m_pol: int
                    poloidal fourier resolution
                n_tor: int
                    toroidal fourier resolution
                Returns
                -------
                n: int
                    toroidal number
        """

    return int(p/(m_pol + 1)) - n_tor


def m(p, m_pol):

    """gives the poloidal number corresponding to the vectorized index p.
                Parameters
                ----------
                p: int
                    vectorized boundary index
                m_pol: int
                    poloidal fourier resolution
                Returns
                -------
                m: int
                    poloidal number
        """
    return p % (m_pol + 1)


def p(m, n, m_pol, n_tor):

    """gives vectorized index p from the toroidal and poloidal numbers.
                Parameters
                ----------
                m: int
                    poloidal number
                n: int
                    toroidal number
                m_pol: int
                    poloidal fourier resolution
                n_tor: int
                    toroidal fourier resolution
                Returns
                -------
                p: int
                    vectorized index
        """
    return (m_pol + 1)*(n + n_tor) + m


def Rz(i_z, i_m, s):

    """gives the zernike polynomial l,m evaluated at s.
                    Parameters
                    ----------
                    i_z: int
                        zernike number
                    i_m: int
                        poloidal number
                    s: float
                        s value
                    Returns
                    -------
                    np.sum(v): float
                        zernike polynomial (i_z, i_m) at s
            """

    if i_z < i_m:
        return 0.0

    if (i_z - i_m) % 2 == 1:
        return 0.0

    v = np.zeros(i_z - i_m + 2)

    for k in range(int(0.5*(i_z - i_m)) + 1):
        v[k] = ((((-1)**k)*math.factorial(i_z - k))/(math.factorial(k)*math.factorial(int(0.5*(i_z + i_m)) - k)*math.factorial(int(0.5*(i_z - i_m)) - k)))*(s**(i_z - 2*k))

    return np.sum(v)


def der_Rz(i, mp, s):

    """gives the derivative of zernike polynomial l,m evaluated at s.
                    Parameters
                    ----------
                    i: int
                        zernike number
                    mp: int
                        poloidal number
                    s: float
                        s value
                    Returns
                    -------
                    np.sum(v): float
                        derivative of zernike polynomial (i_z, i_m) at s
            """

    v = np.zeros(i - mp + 2)
    if i < mp and (i - mp) % 2 == 1:
        return 0.0
    else:
        for k in range(int(0.5 * (i - mp)) + 1):
            if i - 2 * k == 0:
                v[k] = 0
            else:
                v[k] = (i - 2*k)*(((((-1) ** k) * math.factorial(i - k)) / ( math.factorial(k) * math.factorial(int(0.5 * (i + mp)) - k) * math.factorial(int(0.5 * (i - mp)) - k))) * (s ** (i - 2 * k - 1)))

        return np.sum(v)


def R_matrix(s, n_z):

    """Return the matrix constructed from the set of radial zernike polynomials.
            Parameters
            ----------
            s: ndarray
                vector of s values
            n_z: int
                zernike resolution

            Returns
            -------
            R_mat: ndarray, shape(N_rad + 2, n_z, n_z))
                Rradial zernike matrix R_ilm
            """

    dim_i = len(s)
    R_mat = np.zeros((dim_i, n_z + 1, n_z + 1))

    # loops on every mode and computes R_ilm
    for i_i in range(dim_i):
        for i_z in range(n_z + 1):
            for i_m in range(n_z + 1):
                R_mat[i_i, i_z, i_m] = Rz(i_z, i_m, s[i_i])

    return R_mat


def prep_cos_sen_matrix(teta, zeta, N_pol, N_tor, n_tor, m_pol, N_fp):

    """Returns the cosine and sine matrices:
                    Parameters
                    ----------
                    teta: ndarray
                        vector of teta values
                    zeta: ndarray
                        vector of zeta values
                    N_pol: int
                        number of poloidal points
                    N_tor: int
                        number of toroidal points
                    n_pol: int
                        toroidal fourier discretisation
                    m_pol: int
                        poloidal fourier discretisation
                    N_fp: int
                        field period
                    Returns
                    -------
                    cos_mat: ndarray, shape(N_pol, N_tor, n_tor, n_z))
                       cosine matrix C_jknm
                    sin_mat: ndarray, shape(N_pol, N_tor, n_tor, n_z))
                       sin matrix S_jknm
                    dt: ndarray
                       difference between points in teta vector
                     dz: ndarray
                       difference between points in zeta vector
            """

    n_vec = np.arange(-n_tor, n_tor + 1)
    m_vec = np.arange(m_pol + 1)
    dz = np.zeros(N_tor)
    dt = np.zeros(N_pol)
    dt[:] = teta[1:] - teta[:-1]
    dz[:] = zeta[1:] - zeta[:-1]
    m_mat = m_vec[None, None, None, None, None, :]
    teta_mat = teta[None, :-1, None, None, None, None]
    n_mat = n_vec[None, None, None, :, None, None]
    zeta_mat = zeta[None, None, :-1, None, None, None]

    cos_mat = np.cos(m_mat * teta_mat - N_fp * n_mat * zeta_mat)
    sin_mat = np.sin(m_mat * teta_mat - N_fp * n_mat * zeta_mat)
    return cos_mat, sin_mat, dt, dz


def prep_cos_sen_matrix_bc(teta, zeta, n_tor, m_pol, N_fp):

    """Returns the cosine and sine matrices with boundary conditions applied:
            Parameters
            ----------
            teta: ndarray
                vector of teta values
            zeta: ndarray
                vector of zeta values
            N_pol: int
                number of poloidal points
            N_tor: int
                number of toroidal points
            n_pol: int
                toroidal fourier discretisation
            m_pol: int
                poloidal fourier discretisation
            N_fp: int
                field period
            Returns
                -------
            cos_mat_bc: ndarray, shape(N_pol + 1, N_tor, n_tor, n_z))
                cosine matrix C_jknm with boundary conditions
            sin_mat_bc: ndarray, shape(N_pol + 1, N_tor, n_tor, n_z))
                sin matrix S_jknm with boundary conditions
    """

    n_vec = np.arange(-n_tor, n_tor + 1)
    m_vec = np.arange(m_pol + 1)

    # the boundary condition is allowing going back to 2pi at N_pol

    m_mat = m_vec[None, None, None, None, None, :]
    teta_mat = teta[None, :, None, None, None, None]
    n_mat = n_vec[None, None, None, :, None, None]
    zeta_mat = zeta[None, None, :-1, None, None, None]

    cos_mat_bc = np.cos(m_mat * teta_mat - N_fp * n_mat * zeta_mat)
    sin_mat_bc = np.sin(m_mat * teta_mat - N_fp * n_mat * zeta_mat)
    return cos_mat_bc, sin_mat_bc


def prep_heaviside_matrix(n_z):

    """Returns the haviside matrix:
        Parameters
        ----------
        n_z: int
            Zernike resolution
        Returns
        -------
        H_mat: ndarray, shape(n_z, n_z)
            Heaviside matrix H_l,m
        """

    m_vec = np.arange(n_z + 1)
    l_vec = np.arange(n_z + 1)

    m_mat = m_vec[None, :]
    l_mat = l_vec[:, None]
    Id = np.ones((n_z + 1, n_z + 1))
    H_mat = np.heaviside(l_mat - m_mat - Id, 1)

    return H_mat


def prep_RH_matrix(H_mat, R_mat, N_rad, n_z):

    """Return the product between R_ilm and Hlm:
            Parameters
            ----------
            H_mat: ndarray, shape(n_z, n_z)
                Heaviside matrix H_l,m
            R_mat: ndarray, shape(N_rad + 2, n_z, n_z))
                Rradial zernike matrix R_ilm
            N_rad: int
                number of radial points
            n_z: int
               zernike resolution
            Returns
            -------
            R*H: ndarray, shape(N_rad + 2, n_z, n_z)
                product between the R matrix and the heaviside matrix
            """

    R_new = np.zeros((N_rad + 2, n_z + 1))
    for i_s in range(N_rad + 2):
        for i_m in range(n_z + 1):
            R_new[i_s, i_m] = R_mat[i_s, i_m, i_m]

    R = R_new[:, None, :]
    H = H_mat[None, :, :]
    return R*H


def prep_diff_matrix(RH_mat, R_mat, cos_mat, sen_mat):

    """Return the matrices for the explicit differentiation:
            Parameters
            ----------
            RH_mat: ndarray, shape(N_rad + 2, n_z, n_z)
                product between the R matrix and the heaviside matrix
            R_mat: ndarray, shape(N_rad + 2, n_z, n_z))
                Rradial zernike matrix R_ilm
            cos_mat: ndarray, shape(N_pol, N_tor, n_tor, n_z))
                cosine matrix C_jknm
            sin_mat: ndarray, shape(N_pol, N_tor, n_tor, n_z))
                sin matrix S_jknm
            Returns
            -------
            DX: ndarray, shape(N_rad + 2, N_pol, N_tor, n_tor, n_z, n_z)
                differentiation matrix for x_nlm in a point (i,j,k)
            DY: ndarray, shape(N_rad + 2, N_pol, N_tor, n_tor, n_z, n_z)
                differentiation matrix for y_nlm in a point (i,j,k)
            """

    RH = RH_mat[:, None, None, None, :, :]
    R = R_mat[:, None, None, None, :, :]

    DX = cos_mat*(R - RH)
    DY = sen_mat*(R - RH)

    return DX, DY


def unpacking(xy_m, n_tor, n_z, n_vec, z_vec, m_vec, N_dofx, r, z):

    """unpacks xy_m for giving x_nlm, y_nlm:
        Parameters
        ----------
        xy_m: ndarray, shape(N_dof)
            vector with dof
        n_tor: int
            toroidal fourier resolution
        n_z: int
            zernike resolution
        n_vec: ndarray, shape(N_dof)
            n_vec[i] is the toroidal number of xy_m[i]
        m_vec: ndarray, shape(N_dof)
            n_vec[i] is the poloidal number of xy_m[i]
        z_vec: ndarray, shape(N_dof)
            z_vec[i] is the zernike number of xy_m[i]
        N_dofx: int
           number of ndof for x_nlm
        r: ndarray, shape((2N + 1)(M + 1))
            R_nm boundary modes
        z: ndarray, shape((2N + 1)(M + 1)
            Z_mn boundary modes
        Returns
        -------
        x_mc: ndarray, shape(2n_tor + 1, n_z, n_z)
            x zernike/fourier mode coefficients x_nlm
        y_ms: ndarray, shape(2n_tor + 1, n_z, n_z)
            y zernike/fourier mode coefficients x_nlm
        """

    n_vec = n_vec.astype(int)
    z_vec = z_vec.astype(int)
    m_vec = m_vec.astype(int)
    x_mc = np.zeros((2 * n_tor + 1, n_z + 1, n_z + 1))
    y_ms = np.zeros((2 * n_tor + 1, n_z + 1, n_z + 1))

    # assign the degrees of freedom
    x_mc[n_vec[:N_dofx], z_vec[:N_dofx], m_vec[:N_dofx]] = xy_m[:N_dofx]
    y_ms[n_vec[N_dofx:], z_vec[N_dofx:], m_vec[N_dofx:]] = xy_m[N_dofx:]

    # Create a (n, m, m) dim. mask, with ones in the (m, m) diagonal
    all_ones = np.ones((2*n_tor + 1,  n_z + 1, n_z + 1))
    diag_mask = (1 - (np.tril(all_ones, -1) + np.triu(all_ones, 1))).astype(bool)
    upper_with_diag_mask = (np.triu(all_ones)).astype(bool)

    X_lower = x_mc.copy()
    X_lower[upper_with_diag_mask] = 0
    Y_lower = y_ms.copy()
    Y_lower[upper_with_diag_mask] = 0

    # imposing the boundary conditions x_nmm = R_nm - sum_{l=l+1}^M x_nlm, same for y_nmm
    x_mc[diag_mask] = r.flatten() - np.sum(X_lower, axis=1).flatten()
    y_ms[diag_mask] = z.flatten() - np.sum(Y_lower, axis=1).flatten()
    return x_mc, y_ms


def build_const(N_rad, N_pol, N_tor, n_tor, n_z, omega, N_fp, s, teta, zeta, r, z, n_vec, z_vec, m_vec, flag):

    """build the constant list:
            Parameters
            ----------
            N_rad: int
                number of radial points
            N_tor: int
                number of zeta points
            N_pol: int
                number of poloidal points
            n_tor: int
                fourier toroidal resolution
            n_z: int
                zernike resolution
            omega: float
                omega value
            N_fp: int
               number of field period
            r: ndarray, shape((2N + 1)(M + 1))
                R_nm boundary modes
            z: ndarray, shape((2N + 1)(M + 1)
                Z_mn boundary modes
            n_vec: ndarray, shape(N_dof)
                n_vec[i] is the toroidal number of xy_m[i]
            m_vec: ndarray, shape(N_dof)
                n_vec[i] is the poloidal number of xy_m[i]
            z_vec: ndarray, shape(N_dof)
                z_vec[i] is the zernike number of xy_m[i]
            flag: float
                normalisation flag = 1
            Returns
            -------
            const: list
                list of constants
            """

    f = np.zeros(N_rad + 2)
    ds = np.zeros(N_rad + 1)

    # weight factor
    f[:] = 1/(s[:] + 0.1)

    # radial infinitesimal element
    ds[:] = s[1:]-s[:-1]

    # cos and sin matrices
    cos_mat, sen_mat, dt, dz = prep_cos_sen_matrix(teta, zeta, N_pol, N_tor, n_tor, n_z, N_fp)
    cos_mat_bc, sen_mat_bc = prep_cos_sen_matrix_bc(teta, zeta, n_tor, n_z, N_fp)
    ds = ds[:, None, None]
    dt = dt[None, :, None]
    dz = dz[None, None, :]

    # volume element
    dsdtdz = ds * dt * dz
    inv_dsdtdz = np.zeros(np.shape(dsdtdz))
    inv_dsdtdz[:, :, :] = 1 / dsdtdz[:, :, :]

    # Zernike matrix
    R_mat = R_matrix(s, n_z)

    # Heaviside matrix
    H_mat = prep_heaviside_matrix(n_z)

    # counting degrees of freedom
    N_dofx, N_dofy = counting(n_z, n_tor)

    # Zernike times Heavides for the derivative
    RH = prep_RH_matrix(H_mat, R_mat, N_rad, n_z)

    # differentiation matrices
    DX, DY = prep_diff_matrix(RH, R_mat, cos_mat, sen_mat)

    # differentiation matrices with boundary conditions
    DX_bc, DY_bc = prep_diff_matrix(RH, R_mat, cos_mat_bc, sen_mat_bc)

    # flag for normalisation
    if flag == 0:
        N_factor = 1.0
    else:
        N_factor = flag

    return [N_rad, N_pol, N_tor, n_tor, n_z, N_dofx, N_fp, omega, N_factor, s, f, dsdtdz, inv_dsdtdz, r, z, n_vec, z_vec, m_vec, R_mat, cos_mat, sen_mat, DX, DY, DX_bc, DY_bc]


def R_mn_s(i, n, m, x_mc, n_tor, n_z, s):

    """returns the R_nm mode for SPEC:
        Parameters
        ----------
        i: int
            radial index for s
        n: int
            toroidal number
        m: int
            poloidal number
        x_mc: ndarray, shape(2n_tor + 1, n_z + 1, n_z + 1)
            set of x_nlm
        n_tor:
            fourier toroidal resolution
        n_z: int
            zernike resolution
        s: ndarray, shape(N_rad + 2)
            list of radial points
        Returns
        -------
        R_mn_s: float
            SPEC mode R_nm in s[i]

    """
    aus = 0
    for i_l in range(m, n_z + 1):
        aus = aus + Rz(i_l, m, s[i])*x_mc[n + n_tor, i_l, m]

    return aus


def Z_mn_s(i, n, m, y_ms, n_tor, n_z, s):

    """returns the Z_nm mode for SPEC:
            Parameters
            ----------
            i: int
                radial index for s
            n: int
                toroidal number
            m: int
                poloidal number
            x_mc: ndarray, shape(2n_tor + 1, n_z + 1, n_z + 1)
                set of x_nlm
            n_tor:
                fourier toroidal resolution
            n_z: int
                zernike resolution
            s: ndarray, shape(N_rad + 2)
                list of radial points
            Returns
            -------
            Z_mn_s: float
                SPEC mode Z_nm in s[i]

        """
    aus = 0
    for i_l in range(m, n_z + 1):
        aus = aus + Rz(i_l, m, s[i])*y_ms[n + n_tor, i_l, m]

    return aus


def der_R_mn_s(i, n, m, x_mc, n_tor, n_z, s):

    """returns the derivative of R_nm mode for SPEC:
            Parameters
            ----------
            i: int
                radial index for s
            n: int
                toroidal number
            m: int
                poloidal number
            x_mc: ndarray, shape(2n_tor + 1, n_z + 1, n_z + 1)
                set of x_nlm
            n_tor:
                fourier toroidal resolution
            n_z: int
                zernike resolution
            s: ndarray, shape(N_rad + 2)
                list of radial points
            Returns
            -------
            der_R_mn_s: float
                s derivative of SPEC mode R_nm in s[i]

        """
    aus = 0
    for i_l in range(m, n_z + 1):
        aus = aus + der_Rz(i_l, m, s[i])*x_mc[n + n_tor, i_l, m]

    return aus


def der_Z_mn_s(i, n, m, y_ms, n_tor, n_z, s):

    """returns the derivative of Z_nm mode for SPEC:
            Parameters
            ----------
            i: int
                radial index for s
            n: int
                toroidal number
            m: int
                poloidal number
            x_mc: ndarray, shape(2n_tor + 1, n_z + 1, n_z + 1)
                set of x_nlm
            n_tor:
                fourier toroidal resolution
            n_z: int
                zernike resolution
            s: ndarray, shape(N_rad + 2)
                list of radial points
            Returns
            -------
            der_R_mn_s: float
                s derivative of SPEC mode Z_nm in s[i]

        """
    aus = 0
    for i_l in range(m, n_z + 1):
        aus = aus + der_Rz(i_l, m, s[i])*y_ms[n + n_tor, i_l, m]

    return aus


def j_spec(n, m, n_tor):

    """returns the j index from n and m with the spec convention:
        Parameters
        ----------
        n: int
            toroidal mode
        m: int
            poloidal mode
        n_tor: int
            fourier toroidal resolution
        Returns
        -------
        j: int
            SPEC mode index
    """
    if m == 0:
        return np.abs(n)
    else:
        return n_tor + 1 + (m - 1) * (2 * n_tor + 1) + n_tor + n