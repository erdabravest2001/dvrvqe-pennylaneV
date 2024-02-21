from pennylane import numpy as np
from .pot_gen_2d import * 

def get_FBR_R(N_R, l):
    offset_R = l + 1
    delta_R = np.zeros((N_R, N_R), dtype=float)
    for i in range(N_R):
        for j in range(N_R):
            if i == j:
                delta_R[i, j] = 2 * (i + offset_R)
            elif i == j - 1:
                delta_R[i, j] = -np.sqrt((j + offset_R + l) * (j + offset_R - l - 1))
            elif i == j + 1:
                delta_R[i, j] = -np.sqrt((j + offset_R - l) * (j + offset_R + l + 1))

    return delta_R# / 2

def get_DVR_R(N_R, l, return_T=True):
    delta_R = get_FBR_R(N_R, l=l)
    R, eigvecs = np.linalg.eigh(delta_R)
    T_R = np.stack(eigvecs)
    if return_T:
        return R, T_R
    return R

def get_FBR_X(N_theta, K):
    offset_theta = K
    delta_thetaK = np.zeros((N_theta, N_theta), dtype=float)
    for i in range(N_theta):
        for j in range(N_theta):
            if i == j + 1:
                delta_thetaK[i, j] = np.sqrt((j + offset_theta + K + 1) * (j + offset_theta - K + 1) / ((2 * (j + offset_theta) + 1) * (2 * (j + offset_theta) + 3)))
            elif i == j - 1:
                delta_thetaK[i, j] = np.sqrt((j + offset_theta + K) * (j + offset_theta - K) / ((2 * (j + offset_theta) + 1) * (2 * (j + offset_theta) - 1)))

    return delta_thetaK

def get_DVR_X(N_theta, K, return_T=True):
    delta_thetaK = get_FBR_X(N_theta, K)
    X, eigvecs = np.linalg.eigh(delta_thetaK)
    T_thetaK = np.stack(eigvecs)
    if return_T:
        return X, T_thetaK
    return X

def get_sturmian_int(i1, i2, l, p):
    out = (-1)**(i1 + i2 + 2*l) * np.math.factorial(2 * l + p + 2)
    out *= np.sqrt(np.math.factorial(i1 - l - 1) * np.math.factorial(i2 - l - 1) / (np.math.factorial(i1 + l) * np.math.factorial(i2 + l)))
    s = 0
    for t in range(min(i1 - l - 1, i2 - l - 1) + 1):
        if p + 1 >= 0:
            a = np.math.comb(p + 1, i1 - l - t - 1)
            b = np.math.comb(p + 1, i2 - l - t - 1)
        else:
            a = (-1)**(i1 - l - t - 1) * np.math.comb(i1 - l - t - p - 3, i1 - l - t - 1)
            b = (-1)**(i2 - l - t - 1) * np.math.comb(i2 - l - t - p - 3, i2 - l - t - 1)
        c = np.math.comb(2 * l + p + 2 + t, t)
        s += a * b * c
    return out * s

def get_orthonormal_sturmian_int(i1, i2, l, p):
    return get_sturmian_int(i1, i2, l, p - 1)

def get_D_R_FBR(N_R, l, mol_params=arhcl_params):
    offset_R = l + 1
    def get_D_R_FBR_ij(i, j):
        out = (j - l*(l+1) - 3/4) * get_orthonormal_sturmian_int(i, j, l, -2)
        out += (j - 1/2) * get_orthonormal_sturmian_int(i, j, l, -1)
        if i == j:
            out -= 1/4
        if j > (l + 1):
            out -= np.sqrt(j*(j-1) - l*(l+1)) * get_orthonormal_sturmian_int(i, j-1, l, -2)
        out *= 1 / (2 * mol_params['mu'])
        return out
    D_R = np.zeros((N_R, N_R), dtype=float)
    for i in range(N_R):
        for j in range(N_R):
            D_R[i, j] = get_D_R_FBR_ij(i + offset_R, j + offset_R)
    return D_R

def get_D_R_DVR(N_R, l, T_R=None, mol_params=arhcl_params):
    if T_R is None:
        _, T_R = get_DVR_R(N_R, l=l, return_T=True)
    D_R_FBR = get_D_R_FBR(N_R, l=l, mol_params=mol_params)
    return np.matmul(T_R.T, np.matmul(D_R_FBR, T_R))

def get_D_thetaK_FBR(N_theta, K, mol_params=arhcl_params):
    offset_theta = K
    js = np.arange(N_theta) + offset_theta
    D_thetaK = np.diag(js * (js + 1)) / (2 * mol_params['mu'])
    return D_thetaK

def get_D_thetaK_DVR(N_theta, K, T_thetaK=None):
    if T_thetaK is None:
        _, T_thetaK = get_DVR_X(N_theta, K=K, return_T=True)
    D_thetaK_FBR = get_D_thetaK_FBR(N_theta, K=K)
    return np.matmul(T_thetaK.T, np.matmul(D_thetaK_FBR, T_thetaK))

def get_lambda_plus(J, K):
    return np.sqrt(J * (J + 1) - K * (K + 1))

def get_lambda_neg(J, K):
    return np.sqrt(J * (J + 1) - K * (K - 1))

def get_B_plus(K, T_thetaK):
    js = np.arange(T_thetaK.shape[0]) + K
    lam = np.diag(get_lambda_plus(js, K))
    return np.matmul(T_thetaK.T, np.matmul(lam, T_thetaK))

def get_B_neg(K, T_thetaK):
    js = np.arange(T_thetaK.shape[0]) + K
    lam = np.diag(get_lambda_neg(js, K))
    return np.matmul(T_thetaK.T, np.matmul(lam, T_thetaK))

def get_DVR_Rtheta(dvr_options, mol_params=arhcl_params, return_T=False):
    Rs_S, T_R = get_DVR_R(dvr_options['N_R'], l=dvr_options['l'])
    Xs_K = []
    T_thetaK = []
    for K in range(dvr_options['K_max'] + 1):
        Xs, T_theta = get_DVR_X(dvr_options['N_theta'], K)
        Xs_K.append(Xs)
        T_thetaK.append(T_theta)
    Rs = Rs_S * mol_params['S']
    Rs_angs = Rs * au_to_angs

    if dvr_options['r_max'] is None:
        Rs_DVR = (Rs_angs + dvr_options['r_min']) / au_to_angs / mol_params['S']
    else:
        Rs_lim = Rs_angs[Rs_angs + dvr_options['r_min'] <= dvr_options['r_max']] + dvr_options['r_min']
        Rs_lim /= au_to_angs
        Rs_DVR = Rs_lim / mol_params['S']
    if return_T:
        return (Rs_DVR, T_R), (Xs_K, T_thetaK)
    return Rs_DVR, Xs_K

def get_ham_DVR(pot2d, dvr_options, mol_params=arhcl_params, count_thresh=None, v_thresh=None):
    (Rs_DVR, T_R), (Xs_K, T_thetaK) = get_DVR_Rtheta(dvr_options, mol_params=mol_params, return_T=True)
    N_R_lim = Rs_DVR.shape[0]
    JK = np.min([dvr_options['J'], dvr_options['K_max']]) + 1
    Vs_K = []
    for K in range(JK):
        Rs_grid, Xs_grid = np.meshgrid(Rs_DVR * mol_params['S'] * au_to_angs, Xs_K[K])
        Rs_grid, Xs_grid = Rs_grid.flatten(), Xs_grid.flatten()
        Vs = pot2d(Rs_grid, Xs_grid)
        Vs_K.append(Vs)
    Vs_K = np.array(Vs_K)
    D_R = get_D_R_DVR(dvr_options['N_R'], dvr_options['l'], T_R=T_R, mol_params=mol_params)[:N_R_lim, :N_R_lim]
    D_thetaK = []
    for K in range(JK):
        D_theta = get_D_thetaK_DVR(dvr_options['N_theta'], K=K, T_thetaK=T_thetaK[K])
        D_thetaK.append(D_theta)
    D_thetaK = np.array(D_thetaK)
    C_s = []
    E_s = []
    if dvr_options['trunc'] == 0:
        C_s = [np.eye(N_R_lim) for i in range(dvr_options['N_theta'])]
    else:
        for j1 in range(dvr_options['N_theta']):
            Vs_beta = Vs[j1 * N_R_lim:(j1 + 1) * N_R_lim]
            D_R_beta = D_R + np.diag(Vs_beta) / hartree
            eigvals, eigvecs = np.linalg.eigh(D_R_beta)
            if dvr_options['trunc'] == 0:
                cutoff = eigvals.shape[0]
            else:
                cutoff = dvr_options['trunc']
            E_s.append(eigvals[:cutoff])
            C_s.append(eigvecs[:, :cutoff])
    C_s = np.array(C_s)
    h_dvr = np.zeros((JK, dvr_options['N_theta'], N_R_lim, JK, dvr_options['N_theta'], N_R_lim), dtype=float)
    for K in range(JK):
        for j1 in range(len(Xs_K[K])):
            h_dvr[K, j1, :, K, j1, :] += D_R
        for i1 in range(N_R_lim):
            h_dvr[K, :, i1, K, :, i1] += D_thetaK[K] * (1 / np.square(Rs_DVR[i1]) + 1 / np.square(mol_params['r_e']))
            coef = 1 / (2 * mol_params['mu'] * np.square(Rs_DVR[i1]))
            if K + 1 < JK:
                lam_plus = get_lambda_plus(dvr_options['J'], K)
                b_plus = get_B_plus(K, T_thetaK[K])
                coef_plus = -np.sqrt(2) if K == 0 else 1
                h_dvr[K + 1, :, i1, K, :, i1] = coef * coef_plus * lam_plus * b_plus
            if K > 0:
                lam_neg = get_lambda_neg(dvr_options['J'], K)
                b_neg = get_B_plus(K, T_thetaK[K])
                coef_neg = -np.sqrt(2) if K == 1 else 1
                h_dvr[K - 1, :, i1, K, :, i1] = coef * coef_neg * lam_neg * b_neg

            for j1 in range(len(Xs_K[K])):
                h_dvr[K, j1, i1, K, j1, i1] += (dvr_options['J'] * (dvr_options['J'] + 1) - 2 * K**2) / (2 * mol_params['mu'] * np.square(Rs_DVR[i1]))
                h_dvr[K, j1, i1, K, j1, i1] += Vs_K[K, i1 + j1 * N_R_lim] / hartree
    h_dvr2 = np.einsum('ijk,oijplm,lmn->oikpln', C_s, h_dvr, C_s)
    h_dvr2 = h_dvr2.reshape((h_dvr2.shape[0] * h_dvr2.shape[1] * h_dvr2.shape[2], h_dvr2.shape[3] * h_dvr2.shape[4] * h_dvr2.shape[5]))
    if (v_thresh is None) and (count_thresh is not None) and (count_thresh < h_dvr2.shape[0]):
        v_thresh = np.sort(Vs)[count_thresh]
    if v_thresh is not None:
        h_dvr2 = h_dvr2[Vs < v_thresh][:, Vs < v_thresh]
    return h_dvr2

