from pennylane import numpy as np
hartree = 219474.64
da_to_au = 1822.888486209
au_to_angs = 0.529177249

m_ar = 39.9623831238 * da_to_au
m_h = 1.007825031898 * da_to_au
m_cl = 34.96885269 * da_to_au

M_arhcl = m_ar + m_h + m_cl
mu_arhcl = np.sqrt(m_ar * m_h * m_cl / M_arhcl)
S_arhcl = np.power(M_arhcl * m_h * m_cl / (m_ar * (m_h + m_cl)**2), 0.25)
r_e_hcl = 1.2908 / au_to_angs

arhcl_params = {
    'name': 'arhcl',
    'M': M_arhcl,
    'mu': mu_arhcl,
    'S': S_arhcl,
    'r_e': r_e_hcl * S_arhcl
}

m_mg = 23.985041689 * da_to_au
m_n = 14.003074004251 * da_to_au

M_mgnh = m_mg + m_n + m_h
mu_mgnh = np.sqrt(m_mg * m_n * m_h / M_mgnh)
S_mgnh = np.power(M_mgnh * m_n * m_h / (m_mg * (m_n + m_h)**2), 0.25)
r_e_nh = 1.0367 / au_to_angs

mgnh_params = {
    'name': 'mgnh',
    'M': M_mgnh,
    'mu': mu_mgnh,
    'S': S_mgnh,
    'r_e': r_e_nh * S_mgnh
}
