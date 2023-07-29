from nnenum.onnx_network import load_onnx_network_optimized,load_onnx_network
import numpy as np
from nnenum import nnenum
from nnenum.nnenum import make_spec
from nnenum.specification import Specification
from nnenum.settings import Settings
import time 


onnx_file = "../cgan_benchmark2023/onnx/cGAN_imgSz32_nCh_1.onnx"
#vnnlib_file = "../cgan_benchmark2023/vnnlib/cGAN_imgSz32_nCh_1_prop_0_input_eps_0.010_output_eps_0.015.vnnlib"
try:
    network = load_onnx_network_optimized(onnx_file)
except:
    network = load_onnx_network(onnx_file)

#spec_list, input_dtype = make_spec(vnnlib_file, onnx_file)

#for init_box, spec in spec_list:
#    init_box = np.array(init_box, dtype=input_dtype)

bound = 0.05

def check_property(d_lb, d_ub, mid, sign):
    print(mid)
    init_box = [[d_lb, d_ub],
                [-bound, bound],
                [-bound, bound],
                [-bound, bound],
                [-bound, bound]]
    init_box = np.array(init_box, dtype=np.float32)

    if sign == "<=":
        mat = np.array([[-1.]])
        rhs = np.array([mid*-1])
    elif sign == ">=":
        mat = np.array([[1.]])
        rhs = np.array([mid])
    spec = Specification(mat, rhs)
    nnenum.set_image_settings()
    Settings.TRY_QUICK_OVERAPPROX = False
    Settings.PRINT_OUTPUT = False
    res = nnenum.enumerate_network(init_box, network, spec)
    result_str = res.result_str
    print(mid, result_str)
    print("___________________________________________")
    return True if result_str=="safe" else False

d_lb = 0.50
d_ub = 0.51


t0 = time.time()
left = 0.0
right = 1.0
while right-left >= 0.01:
    mid = (left+right)/2.0
    if check_property(d_lb, d_ub, mid, "<="): # check if the outputs are guaranteed smaller or equal to "mid"
        right = mid
    else:
        left = mid
ub = right
print(ub)
print(time.time()-t0)


## search the lower bound (closed)
#left = 0.0
#right = ub
#while right-left >= 0.01:
#    mid = (left+right)/2.0
#    if check_property(d_lb, d_ub, mid, ">="): # check if the output are guaranteed greater or equal to "mid"
#        left = mid
#    else:
#        right = mid
#lb = left
#print(lb)