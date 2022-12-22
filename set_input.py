import numpy as np


def reading_input(input_file):

    """Reads the input file.
    Parameters
    ----------
    input_file : ndarray
        string of path for input file

    Returns
    -------
    N: int
        fourier toroidal resolution
    M: int
        fourier poloidal resolution
    L_rad: int
        number of radial points for SPEC
    N_tor: int
        number of toroidal points in real space
    M_pol: int
        number of poloidal points in real space
    N_fp: int
        number of field period
    omega: float
        value of omega parameter
    s: ndarray
        s points for SPEC evaluation
    r: ndarray, shape((2N + 1)(M + 1))
        R_nm boundary modes
    z: ndarray, shape((2N + 1)(M + 1)
        Z_mn boundary modes
    """


    file1 = open(input_file, 'r')
    Lines = file1.readlines()
    N = 0
    M = 0
    N_pol = 0
    N_tor = 0
    L_rad = 0
    N_fp = 0
    omega = 0
    s = 0

    # loops on the lines
    for i_line in range(8):
        if i_line == 0:
            line = Lines[i_line]
            splitByComma = line.split('=')  # first line toroidal resolution number
            N = int(splitByComma[1])
        if i_line == 1:
            line = Lines[i_line]
            splitByComma = line.split('=')  # second line poloidal resolution number
            M = int(splitByComma[1])
        if i_line == 2:
            line = Lines[i_line]
            splitByComma = line.split('=')  # third line number of radial points to be evaluated on spec
            L_rad = int(splitByComma[1])
        if i_line == 3:
            line = Lines[i_line]
            splitByComma = line.split('=')  # fourth line number of toroidal points
            N_tor = int(splitByComma[1])
        if i_line == 4:
            line = Lines[i_line]
            splitByComma = line.split('=')  # fifth line number of poloidal points
            N_pol = int(splitByComma[1])
        if i_line == 5:
            line = Lines[i_line]
            splitByComma = line.split('=')  # sixth line number of field period
            N_fp = int(splitByComma[1])
        if i_line == 6:
            line = Lines[i_line]
            splitByComma = line.split('=')  # seventh line omega value
            omega = float(splitByComma[1])
        if i_line == 7:
            line = Lines[i_line]
            splitByComma = line.split('=')  # eight line s values
            splitByComma1 = splitByComma[1].replace('[', '').replace(']', '')
            s_split = splitByComma1.split(',')
            s = np.zeros(len(s_split))
            for i_s in range(len(s_split)):
                s[i_s] = float(s_split[i_s])
    r = np.zeros((2 * N + 1, M + 1))
    z = np.zeros((2 * N + 1, M + 1))

    # looping for having the boundary modes R_nm and Z_nm
    for i_line in range(len(Lines)):
        if i_line > 7:
            line = Lines[i_line]
            splitByComma = line.split(',')
            n0 = int(splitByComma[0].replace('R', '').replace('B', '').replace('C', '').replace('(', ''))
            m0 = int(splitByComma[1].split('=')[0].replace(')', ''))
            if abs(n0) <= N and m0 <= M:
                R0 = float(splitByComma[1].split('=')[1])
                Z0 = float(splitByComma[3].split('=')[1])
                r[N + n0][m0] = R0
                z[N + n0][m0] = Z0
    file1.close()

    return N, M, L_rad, N_tor, N_pol, N_fp, omega, s, r, z

