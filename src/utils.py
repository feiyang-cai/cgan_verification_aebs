import numpy as np
import abcrown, arguments
import torch
from utils import load_model
import math
import logging
import time

from matplotlib import pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

class Plotter:
    def __init__(self, d_lbs, v_lbs) -> None:
        self.fig, self.ax = plt.subplots(figsize=(8,8), dpi=200)
        self.d_bounds = [np.inf, -np.inf]
        self.v_bounds = [np.inf, -np.inf]
        self.d_lbs = d_lbs
        self.v_lbs = v_lbs
        self.cell_width = (d_lbs[1] - d_lbs[0])
        self.cell_height = (v_lbs[1] - v_lbs[0])
        self.legend_label_list = []
        self.legend_list = []
    

    def add_patches(self, patches, color, label=None):
        for patch in patches:
            x = patch[0]
            y = patch[2]
            width = patch[1] - patch[0]
            height = patch[3] - patch[2]
            rec = plt.Rectangle((x, y), width, height, facecolor=color, edgecolor='none', alpha=1.0)
            self.ax.add_patch(rec)
            self.d_bounds[0] = min(self.d_bounds[0], x)
            self.d_bounds[1] = max(self.d_bounds[1], x+width)
            self.v_bounds[0] = min(self.v_bounds[0], y)
            self.v_bounds[1] = max(self.v_bounds[1], y+height)

        if label is not None:
            self.legend_label_list.append(label)
            self.legend_list.append(rec)
    
    def add_cells(self, cells, color, label=None, filled=False):
        for cell in cells:
            x = self.d_lbs[cell[0]]
            y = self.v_lbs[cell[1]]
            cell = plt.Rectangle((x, y), self.cell_width, self.cell_height, fill=filled, linewidth=2, facecolor=color, edgecolor='none', alpha=1)
            self.ax.add_patch(cell)
            self.d_bounds[0] = min(self.d_bounds[0], x)
            self.d_bounds[1] = max(self.d_bounds[1], x+self.cell_width)
            self.v_bounds[0] = min(self.v_bounds[0], y)
            self.v_bounds[1] = max(self.v_bounds[1], y+self.cell_height)

        if label is not None:
            self.legend_label_list.append(label)
            self.legend_list.append(cell)
    
    def add_simulations(self, ds, vs, color, label=None):
        scatter = self.ax.scatter(ds, vs, c=color, alpha=0.8, s=1)
        if label is not None:
            self.legend_label_list.append(label)
            self.legend_list.append(scatter)
    
    def save_figure(self, file_name, x_range=None, y_range=None):
        if x_range is not None and y_range is not None:
            self.ax.set_xlim(x_range[0], x_range[1])
            self.ax.set_ylim(y_range[0], y_range[1])
        else:    
            self.ax.set_xlim(self.d_bounds[0]-0.2, self.d_bounds[1]+0.2)
            self.ax.set_ylim(self.v_bounds[0]-0.2, self.v_bounds[1]+0.2)

        ## plot grids
        for d_lb in self.d_lbs:
            X = [d_lb, d_lb]
            Y = [self.v_lbs[0], self.v_lbs[-1]]
            self.ax.plot(X, Y, 'lightgray', alpha=0.2)

        for v_lb in self.v_lbs:
            Y = [v_lb, v_lb]
            X = [self.d_lbs[0], self.d_lbs[-1]]
            self.ax.plot(X, Y, 'lightgray', alpha=0.2)
        
        self.ax.set_xlabel(r"$d$ (m)")
        self.ax.set_ylabel(r"$v$ (m/s)")
        if len(self.legend_list) != 0:
            #self.ax.legend(self.legend_list, self.legend_label_list, loc='lower right')
            self.ax.legend(self.legend_list, self.legend_label_list, loc='upper right', bbox_to_anchor=(1.0, 1.05), borderaxespad=0.)
        self.fig.savefig(file_name)
        plt.close()


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
    def __init__(self, d_lbs, d_ubs, v_lbs, v_ubs, step=1, latent_bounds=0.01):
        self.d_lbs = d_lbs
        self.d_ubs = d_ubs
        self.v_lbs = v_lbs
        self.v_ubs = v_ubs
        assert latent_bounds >= 0
        self.latent_bounds = latent_bounds
        self.step = step
        arguments.Config.all_args['model']['input_shape'] = [-1, 2 + step * 4]

    def check_property(self, init_box, mid, sign):
        neg_sign = "<=" if sign == ">=" else ">="
        save_vnnlib(init_box, mid, neg_sign)
        for batch_size in [5000, 10]:
            arguments.Config.all_args['solver']['crown']['batch_size'] = batch_size
            try:
                logging.info(f"            batch_size: {batch_size}")
                torch.cuda.empty_cache()
                verified_status = abcrown.main()
                break
            except:
                continue
        self.num_calls_alpha_beta_crown += 1
        if verified_status == "unsafe-pgd":
            return False
        elif verified_status == "safe":
            return True
        else:
            raise NotImplementedError
    
    def get_overlapping_cells(self, lb_ub, ub_lb, init_box, ub_ub_idx, index):
        # lb_ub: the lower bound of the upper bound
        # ub_lb: the upper bound of the lower bound
        # init_box: the initial box
        # index: the index of the variable to be searched

        if index == 0:
            # if the lb of the distance is less or equal to zero, 
            # then the vehicle is already in the danger zone.
            # this case should be handled outside of this function
            assert lb_ub > 0.0
        
        ubs = self.d_ubs if index == 0 else self.v_ubs
        lbs = self.d_lbs if index == 0 else self.v_lbs
        
        ## search the lb
        logging.info(f"        search for lb for idx {index}")
        right_idx = math.floor((lb_ub - lbs[0])/(ubs[0]-lbs[0]))
        found_lb = False
        error_found_lb = False
        for i in range(right_idx, -1, -1):
            logging.info(f"            checking output >= {lbs[i]}: {i}")
            try:
                if self.check_property(init_box, lbs[i], ">="):
                    logging.info(f"            verified, the lb idx is {i}")
                    found_lb = True
                    lb_idx = i
                    break
                else:
                    pass
            except:
                self.error_during_verification = True
                error_found_lb = True
                logging.info(f"            error occurs when checking output >= {lbs[i]}: {i}")

        if not found_lb:
            logging.info(f"            the lb is not guaranteed greater or equal to {lbs[0]}")
            lb_idx = -1
            if error_found_lb:
                logging.info(f"            this bound cannot be verified due to error, return -2")
                lb_idx = -2
        
        ## search the ub
        logging.info(f"        search for ub for idx {index}")
        left_idx = math.ceil((ub_lb - ubs[0])/(ubs[0]-lbs[0]))
        left_idx = max(left_idx, 0)
        left_idx = min(left_idx, ub_ub_idx-1)
        logging.info(f"            left_idx: {left_idx}, right_idx: {ub_ub_idx-1}")
        if left_idx == ub_ub_idx-1:
            if left_idx == 0:
                logging.info(f"            checking output <= 0")
                if self.check_property(init_box, 0.0, "<="):
                    logging.info(f"            verified, the ub idx is {-1}")
                    ub_idx = -1
                else:
                    logging.info(f"            the ub is not guaranteed less or equal to {ubs[0]}")
                    ub_idx = -2
            else: 
                logging.info(f"            verification is not needed, the ub idx is {left_idx}")
                ub_idx = left_idx
            
        for i in range(left_idx, ub_ub_idx-1):
            logging.info(f"            checking output <= {ubs[i]}: {i}")
            try:
                if self.check_property(init_box, ubs[i], "<="):
                    logging.info(f"            verified, the ub idx is {i}")
                    ub_idx = i
                    break
                else:
                    pass
            except:
                self.error_during_verification = True
                ub_idx = ub_ub_idx-1
                logging.info(f"            error occurs when checking output <= {ubs[i]}: {i}")
        
        logging.info(f"        lb_idx: {lb_idx}, ub_idx: {ub_idx}") 
        return (lb_idx, ub_idx)
        
    
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
        arguments.Config.all_args['model']['name'] = f'Customized("custom_model_data", "MultiStep", index=0, num_steps={self.step})'
        # in order to save the gpu memory, we load the model for each simulation
        model_ori = load_model().cuda()
        model_ori.eval()
        outputs = model_ori(torch.from_numpy(inputs).cuda())
        del model_ori
        d_lb_sim = torch.min(outputs).item()
        d_ub_sim = torch.max(outputs).item()
        logging.info(f"    d_lb_sim: {d_lb_sim}, d_ub_sim: {d_ub_sim}")
        del outputs
        
        #assert d_ub_sim <= d_ub
        if d_lb_sim <= 0.0:
            # the vehicle is already in the danger zone, return (-1, 0, 0, 0)
            return (-1, 0, 0, 0)

        ub_ub_idx = d_idx+1
        d_lb_idx, d_ub_idx = self.get_overlapping_cells(d_lb_sim, d_ub_sim, init_box, ub_ub_idx, index=0)

        if d_lb_idx == -2 or d_ub_idx == -2:
            # the bounds cannot be verified due to error, return (-2, -2, -2, -2)
            return [-2, -2, -2, -2]
        
        if d_lb_idx == -1:
            # the vehicle is already in the danger zone, return (-1, 0, 0, 0)
            return [-1, 0, 0, 0]

        # velocity
        arguments.Config.all_args['model']['name'] = f'Customized("custom_model_data", "MultiStep", index=1, num_steps={self.step})'
        # in order to save the gpu memory, we load the model for each simulation
        model_ori = load_model().cuda()
        model_ori.eval()
        outputs = model_ori(torch.from_numpy(inputs).cuda())
        del model_ori
        v_lb_sim = torch.min(outputs).item()
        v_ub_sim = torch.max(outputs).item()
        logging.info(f"    v_lb_sim: {v_lb_sim}, v_ub_sim: {v_ub_sim}")
        assert v_ub_sim <= v_ub
        del outputs

        ub_ub_idx = v_idx+1
        v_lb_idx, v_ub_idx = self.get_overlapping_cells(v_lb_sim, v_ub_sim, init_box, ub_ub_idx, index=1)
        if v_lb_idx == -2 or v_ub_idx == -2:
            # the bounds cannot be verified due to error, return (-2, -2, -2, -2)
            return [-2, -2, -2, -2]
        if v_ub_idx == -1:
            return [d_lb_idx, d_ub_idx, -1, -1]

        return [d_lb_idx, d_ub_idx, v_lb_idx, v_ub_idx]
    
    def compute_next_reachable_cells(self, d_idx, v_idx):
        result_dict = dict()

        time_dict = dict()
        self.num_calls_alpha_beta_crown = 0
        self.error_during_verification = False
        time_start = time.time()
        interval = self.get_intervals(d_idx, v_idx)
        time_end_get_intervals = time.time()
        logging.info(f"    Interval index: {interval}")
        if interval == [-2, -2, -2, -2]:
            result_dict["error"] = True
            result_dict["reachable_cells"] = {(-1, -1)}
        
        elif interval[0] == -1:
            result_dict["reachable_cells"] = {(-2, -2)}
        
        elif interval[2] == -1 and interval[3] == -1:
            result_dict["reachable_cells"] = {(-3, -3)}
         
        else: 
            reachable_cells = set()
            
            assert 0 <= interval[0] < len(self.d_lbs)
            assert 0 <= interval[1] < len(self.d_ubs)
            if interval[2] == -1:
                # in this case, the vehicle must already stopped
                reachable_cells.add((-3, -3))
                interval[2] = 0
            assert 0 <= interval[2] < len(self.v_lbs)
            assert 0 <= interval[3] < len(self.v_ubs)

            for d_idx in range(interval[0], interval[1]+1):
                for v_idx in range(interval[2], interval[3]+1):
                    reachable_cells.add((d_idx, v_idx))            
            result_dict["reachable_cells"] = reachable_cells

        time_end = time.time()
        time_dict["whole_time"] = time_end - time_start
        time_dict["get_intervals_time"] = time_end_get_intervals - time_start
        result_dict["time"] = time_dict
        result_dict["num_calls_alpha_beta_crown"] = self.num_calls_alpha_beta_crown
        result_dict["error_during_verification"] = self.error_during_verification
        return result_dict