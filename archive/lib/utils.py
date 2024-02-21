import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit.algorithms.optimizers import COBYLA, SPSA

base_dir = '/arc/project/st-rkrems-1/kasra/dvr_vqe/2d/'
scratch_dir = '/scratch/st-rkrems-1/kasra/dvr_vqe/'

hartree = 219474.64
da_to_au = 1822.888486209
au_to_angs = 0.529177249

def print_log(s, file, end='\n', overwrite=False):
   s = str(s)
   if not overwrite:
      print(s, end=end)
   else:
      print('\r' + s, end=end)
   if file is not None:
      if not overwrite:
         f = open(file, 'a', encoding="utf-8")
         f.write(s)
         f.write(end)
      else:
         f = open(file, 'r', encoding="utf-8")
         lines = f.readlines()[:-1]
         f.close()
         lines.append(s + end)
         f = open(file, 'w', encoding="utf-8")
         f.writelines(lines)
      f.close()

def cssg(all_converge_cnts_list, all_converge_vals_list, all_h_dvr_list, spin_names):
    # Set seaborn theme and plot size
    sns.set_theme()
    plt.rcParams['figure.figsize'] = (12, 8)

    optimizers = ['COBYLA', 'SPSA']
    techniques = ['No Error Mitigation', 'TREX', 'ZNE', 'PEC']

    for spin_idx, (converge_cnts_list, converge_vals_list, h_dvr) in enumerate(zip(all_converge_cnts_list, all_converge_vals_list, all_h_dvr_list)):
        ref_value = np.min(h_dvr)  # Compute reference value for each spin
        eigvals = [np.min(h_dvr)]  # As an example, you can replace it with your eigenvalues

        for res_idx, (converge_cnts_res, converge_vals_res) in enumerate(zip(converge_cnts_list, converge_vals_list)):
            plt.figure()

            for opt_idx, (converge_cnts_opt, converge_vals_opt) in enumerate(zip(converge_cnts_res, converge_vals_res)):
                percent_error = np.abs((converge_vals_opt - ref_value) / ref_value) * 100

                print(techniques[res_idx], optimizers[opt_idx], converge_vals_opt[-1], percent_error[-1])

                # Plot energy convergence
                plt.subplot(2, 2, opt_idx + 1)
                plt.plot(converge_cnts_opt, converge_vals_opt, label=optimizers[opt_idx])
                plt.axhline(eigvals[0], ls='--', c='blue', label='Numerical g.s.')
                plt.axhline(ref_value, ls='--', c='black', label='Converged g.s.')
                plt.xlabel('Evaluation count')
                plt.ylabel('Energy (cm$^{-1}$)')
                plt.title('Energy Convergence with {}'.format(techniques[res_idx]))
                plt.ylim(ref_value - 100, ref_value + 100)
                plt.legend(loc='upper right')

                # Plot percent error
                plt.subplot(2, 2, opt_idx + 3)
                plt.plot(converge_cnts_opt, percent_error, label=optimizers[opt_idx])
                plt.axhline(0, ls='--', c='black', label='Reference')
                plt.xlabel('Evaluation count')
                plt.ylabel('Percent Error')
                plt.title('Percent Error with {}'.format(techniques[res_idx]))
                plt.ylim(0, 100)
                plt.legend(loc='upper right')

                plt.tight_layout()

                # Save as side-by-side graph
                plt.savefig(f'side_by_side_{spin_names[spin_idx]}_{techniques[res_idx].lower().replace(" ", "_")}.png')
                plt.close()