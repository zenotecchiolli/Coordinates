import numpy as np
from set_input import reading_input
from functional import set_initial_guess
from functional import coordinates
from auxiliary import unpacking
from functional import action
from auxiliary import build_const
from gradient import gradient_action
from graphics import plot_out
from scipy import optimize
from functional import out_quantities
from functional import out_vec


def run(file_in, file_out, file_spec):

    """runs the code:
            Parameters
            ----------
            file_in: basestring
                input text file
            file_in: basestring
                output text file with vector of optimised dof
            file_spec: basestring
                output text file with SPEC modes
            -------
        """

    # reads the input
    n_tor, m_pol, N_rad, N_tor, N_pol, N_fp, omega, s, r, z = reading_input(file_in)

    # prepares the starting vector
    s_data = np.zeros(len(s))
    s_data[:] = (s[:]+1)/2 #converts s-input into sbar-input
    s = np.linspace(0, 1, N_rad + 2)
    n_z = m_pol
    teta = np.linspace(0, 2 * np.pi, N_pol + 1)
    zeta = np.linspace(0, (2 * np.pi) / (N_fp), N_tor + 1)
    xy_m, n_vec, m_vec, z_vec, N_dofx = set_initial_guess(n_z, n_tor)
    const = build_const(N_rad, N_pol, N_tor, n_tor, n_z, omega, N_fp, s, teta, zeta, r, z, n_vec, z_vec, m_vec, 0.0)
    print('Number of degrees of freedom:', 2*N_dofx)
    N_rad, N_pol, N_tor, n_tor, n_z, N_dofx, N_fp, omega, N_factor, s, f, dsdtdz, inv_dsdtdz, r, z, n_vec, z_vec, m_vec, R_mat, cos_mat, sen_mat, DX, DY, DX_bc, DY_bc = const
    flag = action(xy_m, const)
    const = build_const(N_rad, N_pol, N_tor, n_tor, n_z, omega, N_fp, s, teta, zeta, r, z, n_vec, z_vec, m_vec, flag)

    # runs the minimisation
    print('minimisation starts:')
    x0 = xy_m
  
    plot_out(x0,const)

    res = optimize.minimize(action, x0, method='BFGS', args=const, jac=gradient_action,
               options={'disp': True})
    vec = res.x

    # plots the result
    plot_out(vec, const)

    # saves the output on SPEC and in vector format on file
    const = build_const(len(s_data) - 2, N_pol, N_tor, n_tor, n_z, omega, N_fp, s_data, teta, zeta, r, z, n_vec, z_vec, m_vec, flag)
    out_quantities(vec, const, file_spec)
    out_vec(vec, file_out)

