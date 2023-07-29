import numpy as np
import abcrown, arguments
import torch
from utils import load_model
import math
import logging
import time
from collections import defaultdict

def save_vnnlib(input_bounds, mid, sign, spec_path="./temp.vnnlib"):

    with open(spec_path, "w") as f:

        # Declare input variables.
        f.write("\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # Declare output variables.
        f.write("\n")
        f.write(f"(declare-const Y_0 Real)\n")
        f.write("\n")

        # Define input constraints.
        f.write(f"; Input constraints:\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(assert (<= X_{i} {input_bounds[i, 1]}))\n")
            f.write(f"(assert (>= X_{i} {input_bounds[i, 0]}))\n")
            f.write("\n")
        f.write("\n")

        # Define output constraints.
        f.write(f"; Output constraints:\n")
        f.write(f"(assert ({sign} Y_0 {mid}))\n")

class MultiStepVerifier:
    def __init__(self, d_lbs, d_ubs, v_lbs, v_ubs, step=1, latent_bounds=0.01, tolerance_ratio=0.1, binary_search_divide_ratio=0.5):
        self.d_lbs = d_lbs
        self.d_ubs = d_ubs
        self.v_lbs = v_lbs
        self.v_ubs = v_ubs
        assert latent_bounds >= 0
        self.latent_bounds = latent_bounds
        self.step = step
        self.tolerance_ratio = tolerance_ratio
        self.binary_search_divide_ratio = binary_search_divide_ratio    

    def check_property(self, init_box, mid, sign):
        neg_sign = "<=" if sign == ">=" else ">="
        save_vnnlib(init_box, mid, neg_sign)
        verified_status = abcrown.main()
        self.num_calls_alpha_beta_crown += 1
        if verified_status == "unsafe-pgd":
            return False
        elif verified_status == "safe":
            return True
        else:
            raise NotImplementedError
    
    def binary_search(self, lb_ub, ub_lb, init_box, index, tolerance):
        # lb_ub: the lower bound of the upper bound
        # ub_lb: the upper bound of the lower bound
        # init_box: the initial box
        # index: the index of the variable to be searched

        if index == 0:
            # if the lb of the distance is less or equal to zero, 
            # then the vehicle is already in the danger zone.
            # this case should be handled outside of this function
            assert lb_ub > 0.0

        ## search the lb
        ### check if the output are guaranteed greater or equal to 0
        logging.info("        binary search for lb")
        if self.check_property(init_box, 0.0, ">="):
            ### if yes, then the lb_lb is 0, and we can start to search the lb
            left = 0.0
            right = lb_ub
            while right-left >= tolerance:
                logging.info(f"            left: {left}, right: {right}")
                #mid = (left+right)/2.0
                mid = right - (right-left)*self.binary_search_divide_ratio
                if self.check_property(init_box, mid, ">="):
                    logging.info(f"            >=mid: {mid}, verified")
                    left = mid
                else:
                    logging.info(f"            >=mid: {mid}, not verified")
                    right = mid
            lb = left
        else:
            logging.info("        the lb is not guaranteed greater or equal to 0")
            ### if no, then the lb_lb can be value less than 0
            #### if searching for the lb of the distance, then the vehicle can in the danger zone
            if index == 0:
                return (-1, 0)
            #### if searching for the lb of the velocity, then the vehicle can be stopped
            elif index == 1:
                lb = 0.0
        
        ## search the ub
        ### since the distance and velocity are always decreasing,
        ### the ub_ub is the ub of the initial box
        left = ub_lb
        right = init_box[index][1]
        logging.info("        binary search for ub")
        while right-left >= tolerance or right<=0.0:
            logging.info(f"            left: {left}, right: {right}")
            #mid = (left+right)/2.0
            mid = left + (right-left)*self.binary_search_divide_ratio
            if self.check_property(init_box, mid, "<="): # check if the output are guaranteed less or equal to "mid"
                logging.info(f"            <=mid: {mid}, verified")
                right = mid
            else:
                logging.info(f"            <=mid: {mid}, not verified")
                left = mid
        ub = right

        return (lb, ub)
    
    def get_intervals(self, d_idx, v_idx):
        d_lb = self.d_lbs[d_idx]
        d_ub = self.d_ubs[d_idx]
        v_lb = self.v_lbs[v_idx]
        v_ub = self.v_ubs[v_idx]

        init_box = [[d_lb, d_ub],
                    [v_lb, v_ub]]
        init_box.extend([[-self.latent_bounds, self.latent_bounds]]*4*self.step)
        init_box = np.array(init_box, dtype=np.float32)

        # simulate the system
        samples = 10000
        inputs = []
        for bounds in init_box:
            inputs.append(np.random.uniform(bounds[0], bounds[1], samples).astype(np.float32))
        inputs = np.stack(inputs, axis=1)

        # distance 
        arguments.Config.all_args['model']['name'] = 'Customized("custom_model_data", "SingleStep", index=0)'
        # in order to save the gpu memory, we load the model for each simulation
        model_ori = load_model().cuda()
        model_ori.eval()
        outputs = model_ori(torch.from_numpy(inputs).cuda())
        del model_ori
        d_lb_sim = torch.min(outputs).item()
        d_ub_sim = torch.max(outputs).item()
        logging.info(f"    d_lb_sim: {d_lb_sim}, d_ub_sim: {d_ub_sim}")
        
        #assert d_ub_sim <= d_ub
        if d_lb_sim <= 0.0:
            # the vehicle is already in the danger zone, return (-1, 0, 0, 0)
            return (-1, 0, 0, 0)
        
        logging.info("    binary search for d_lb_ and d_ub_") 
        d_lb_, d_ub_ = self.binary_search(d_lb_sim, d_ub_sim, init_box, index=0, tolerance=(self.d_lbs[1]-self.d_lbs[0])*self.tolerance_ratio)
        logging.info(f"    d_lb_: {d_lb_}, d_ub_: {d_ub_}")
        if d_lb_ < 0:
            # the vehicle is already in the danger zone, return (-1, 0, 0, 0)
            return (-1, 0, 0, 0)

        # velocity
        arguments.Config.all_args['model']['name'] = 'Customized("custom_model_data", "SingleStep", index=1)'
        # in order to save the gpu memory, we load the model for each simulation
        model_ori = load_model().cuda()
        model_ori.eval()
        outputs = model_ori(torch.from_numpy(inputs).cuda())
        del model_ori
        v_lb_sim = torch.min(outputs).item()
        v_ub_sim = torch.max(outputs).item()
        logging.info(f"    v_lb_sim: {v_lb_sim}, v_ub_sim: {v_ub_sim}")
        assert v_ub_sim <= v_ub
        logging.info("    binary search for v_lb_ and v_ub_")
        v_lb_, v_ub_ = self.binary_search(v_lb_sim, v_ub_sim, init_box, index=1, tolerance=(self.v_lbs[1]-self.v_lbs[0])*self.tolerance_ratio)
        logging.info(f"    v_lb_: {v_lb_}, v_ub_: {v_ub_}")

        return (d_lb_, d_ub_, v_lb_, v_ub_)
    
    def get_overlapping_cells_from_intervals(self, d_bounds, v_bounds):
        """
        Get the overlapping cells from the intervals.
        :param d_bounds: distance bounds
        :param v_bounds: velocity bounds
        :return: overlapping cells
        """

        reachable_cells = set()

        # get the lower and upper bound indices of the output interval
        p_lb_idx = math.floor((d_bounds[0] - self.d_lbs[0])/(self.d_ubs[0]-self.d_lbs[0])) # floor
        p_ub_idx = math.ceil((d_bounds[1] - self.d_lbs[0])/(self.d_ubs[0]-self.d_lbs[0])) # ceil

        theta_lb_idx = math.floor((v_bounds[0] - self.v_lbs[0])/(self.v_ubs[0]-self.v_lbs[0])) # floor
        theta_ub_idx = math.ceil((v_bounds[1] - self.v_lbs[0])/(self.v_ubs[0]-self.v_lbs[0])) # ceil

        assert 0<=p_lb_idx<len(self.d_lbs)
        assert 1<=p_ub_idx<=len(self.d_ubs)
        assert 0<=theta_lb_idx<len(self.v_lbs)
        assert 1<=theta_ub_idx<=len(self.v_ubs)

        for p_idx in range(p_lb_idx, p_ub_idx):
            for theta_idx in range(theta_lb_idx, theta_ub_idx):
                reachable_cells.add((p_idx, theta_idx))

        return reachable_cells
    
    def compute_next_reachable_cells(self, d_idx, v_idx):
        result_dict = dict()

        time_dict = dict()
        self.num_calls_alpha_beta_crown = 0
        time_start = time.time()
        interval = self.get_intervals(d_idx, v_idx)
        time_end_get_intervals = time.time()
        logging.info(f"    Interval: {interval}")
        if interval[0] < 0:
            result_dict["reachable_cells"] = {(-2, -2)}
        
        elif interval[3] < 0:
            # in this case, the vehicle must already stopped
            result_dict["reachable_cells"] = {(-3, -3)}
        else: 
            time_start_get_overlapping_cells = time.time()
            reachable_cells = self.get_overlapping_cells_from_intervals(interval[:2], interval[2:])
            result_dict["reachable_cells"] = reachable_cells
            time_end_get_overlapping_cells = time.time()
            time_dict["get_overlapping_cells_time"] = time_end_get_overlapping_cells - time_start_get_overlapping_cells
        time_end = time.time()
        time_dict["whole_time"] = time_end - time_start
        time_dict["get_intervals_time"] = time_end_get_intervals - time_start
        result_dict["time"] = time_dict
        result_dict["num_calls_alpha_beta_crown"] = self.num_calls_alpha_beta_crown
        return result_dict

        
        
        
                    


        