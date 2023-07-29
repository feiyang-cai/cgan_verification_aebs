import numpy as np

data = np.load("Verified_ret_[Customized_model]_start=0_end=10000_iter=5_b=16_timeout=360.0_branching=kfsb-max-3_lra-init=0.25_lra=0.05_lrb=0.1_PGD=before_cplex_cuts=False_multiclass=allclass_domain.npy", allow_pickle=True)
print(data)