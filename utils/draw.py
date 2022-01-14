import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
plt.style.use("ggplot")

beta = [0.3, 0.6, 0.9, 1.2, 1.5]

beta_hu = [0.1, 0.4, 0.7, 1, 1.3]

ABG_uh_beta = [0.6929,  0.6771,  0.7040, 0.6743, 0.6773]
ABG_hu_beta = [0.7192, 0.7432, 0.7452, 0.7310, 0.7433]

gamma_uh = [0.4, 0.6, 0.8, 1, 1.2 , 1.4]
gamma_hu = [8, 10, 12, 14]

ABG_uh_gamma = [0.6975, 0.6929, 0.6918,  0.7040, 0.6921, 0.6921]
ABG_hu_gamma = [0.7437, 0.7452, 0.7410, 0.7423]

fig, axs = plt.subplots(ncols=2, nrows=2)
ax1, ax2, ax3, ax4 = axs.ravel()

ax1.plot(beta, ABG_uh_beta, label='Ours')
ax1.set_ylabel("HOS")
ax1.set_xlabel(r"$\beta$")
ax1.set_ylim(0.55, 0.8)
plt.tight_layout()
plt.subplots_adjust(left=None, bottom=0.12, right=None, top=None, wspace=0.3, hspace=None)
ax1.legend()

ax2.plot(gamma_uh, ABG_uh_gamma, label='Ours')
ax2.set_ylabel("HOS")
ax2.set_xlabel(r"$\gamma$")
ax2.set_ylim(0.6, 0.8)
ax2.legend()

ax3.plot(beta_hu, ABG_hu_beta, label='Ours')
ax3.set_ylabel("HOS")
ax3.set_xlabel(r"$\beta$")
ax3.set_ylim(0.6, 0.85)
ax3.legend()

ax4.plot(gamma_hu, ABG_hu_gamma, label='Ours')
ax4.set_ylabel("HOS")
ax4.set_xlabel(r"$\gamma$")
ax4.set_ylim(0.65, 0.85)
ax4.legend()
# ax4.margins(0)


plt.show()
plt.savefig("Parameter Sensitivity")