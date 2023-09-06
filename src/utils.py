import numpy as np
import abcrown, arguments
import torch
from load_model import load_model
import math
import logging
import time

from matplotlib import pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import pickle
from tqdm import tqdm
from collections import defaultdict
import subprocess
import gc
import os
import yaml

#def delete_all_models_on_gpu():
#    for i in range(torch.cuda.device_count()):
#        device = torch.device(f'cuda:{i}')
#        torch.cuda.set_device(device)  # Set the current device
#        for obj in gc.get_objects():
#            if isinstance(obj, torch.nn.Module):
#                if next(obj.parameters(), None) is not None and next(obj.parameters(), None).is_cuda:
#                    del obj
#                    torch.cuda.empty_cache()

def get_gpu_memory_usage():
    return torch.cuda.memory_allocated() / 1024**2


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
    def __init__(self, d_lbs, d_ubs, v_lbs, v_ubs, step=1, latent_bounds=0.01, simulation_samples=10000, reachable_cells_path=None, frequency=20):
        self.d_lbs = d_lbs
        self.d_ubs = d_ubs
        self.v_lbs = v_lbs
        self.v_ubs = v_ubs

        if reachable_cells_path is not None:
            self.reachable_cells = pickle.load(open(reachable_cells_path, 'rb'))
        else:
            self.simulation_samples = simulation_samples
            assert latent_bounds >= 0
            self.latent_bounds = latent_bounds
            self.step = step
            self.frequency = frequency

            arguments.Config.all_args['model']['input_shape'] = [-1, 2 + step * 4]
            arguments.Config.all_args['model']['path'] = f'./models/single_step_{frequency}Hz.pth'
            

    def load_abcrown_setting(self, setting_idx, output_idx, num_steps, controller_freq, vnnpath, spec_path="./temp.vnnlib"):
        # setting_idx: 0, 1, 2, 3, 4, 5, 6, 7
        settings = {
            'general': {
                'device': 'cuda',
                'conv_mode': 'matrix',
                'results_file': 'results.txt',
            },

            'model': {
                'name': f'Customized("custom_model_data", "MultiStep", index={output_idx}, num_steps={num_steps})',
                'path': f'./models/single_step_{controller_freq}Hz.pth',
                'input_shape': [-1, 2 + num_steps * 4],
            },
            
            'data': {
                'dataset': 'CGAN',
                'num_outputs': 1,
            },
            
            'specification': {
                'vnnlib_path': f"{vnnpath}",
            },

            'solver': {
                'batch_size': 1,
                'auto_enlarge_batch_size': True,
            },

            'attack': {
                'pgd_order': 'before',
                'pgd_restarts': 100
            },

            'bab': {
                'initial_max_domains': 100,
                'branching': {
                    'method': 'sb',
                    'sb_coeff_thresh': 0.01,
                    'input_split': {
                        'enable': True,
                        'catch_assertion': True,
                    },
                },
                'timeout': 300,
            }
        }
        
        batch_size = [4096, 1024, 512, 128, 32, 16, 8]
        if setting_idx == 0:
            settings['general']['enable_incomplete_verification'] = True
            settings['solver']['bound_prop_method'] = 'crown'
            settings['solver']['crown'] = {'batch_size': 512}
            
        else:
            # try complete verification with alpha-crown
            settings['general']['enable_incomplete_verification'] = False
            settings['solver']['bound_prop_method'] = 'alpha-crown'
            settings['solver']['crown'] = {'batch_size' : batch_size[setting_idx - 1]}
            settings['solver']['alpha-crown'] = {'lr_alpha': 0.25, 'iteration': 20, 'full_conv_alpha': False}
            settings['solver']['beta-crown'] = {'lr_alpha': 0.05, 'lr_beta': 0.1, 'iteration': 5}
        
        with open(spec_path, 'w') as f:
            yaml.dump(settings, f)
    
    def read_results(self, result_path):
        if os.path.exists(result_path):
            with open(result_path, "rb") as f:
                lines = pickle.load(f)
                results = lines['results'][0][0]
                #results = pickle.load(f)['results'][0][0]
            return results

        else:
            return "unknown"
    
    def check_property(self, init_box, mid, sign, idx):
        neg_sign = "<=" if sign == ">=" else ">="
        spec_path = "./temp.vnnlib"
        config_path = "./cgan.yaml"
        result_path = "./results.txt"
        ## clean the file first
        if os.path.exists(spec_path):
            os.remove(spec_path)
        assert not os.path.exists(spec_path)
        save_vnnlib(init_box, mid, neg_sign, spec_path="./temp.vnnlib")

        for setting_idx in range(8): # try different settings, starting from incomplete verification, and then complete verification with different batch sizes
            if os.path.exists(result_path):
                os.remove(result_path)
            if os.path.exists(config_path):
                os.remove(config_path)
            assert not os.path.exists(config_path)
            assert not os.path.exists(result_path)

            self.load_abcrown_setting(setting_idx, output_idx=idx, num_steps=self.step, controller_freq=self.frequency, vnnpath=spec_path, spec_path=config_path)

            logging.info(f"                using setting {setting_idx}")
            logging.info(f"                gpu memory usage: {get_gpu_memory_usage()}")

            ### verify the property
            #verified_status = abcrown.main()

            ### using subprocess to run the verification
            ### this is because the abcrown will change the gpu memory usage, and we need to clear the memory after each verification
            tool_dir = os.environ.get('TOOL_DIR')
            process = subprocess.Popen(["python", 
                                        os.path.join(tool_dir, "complete_verifier/abcrown.py"),
                                        "--config",
                                        "./cgan.yaml"])
            process.wait()

            verified_status = self.read_results(result_path)

            logging.info(f"                verification status: {verified_status}")
            if verified_status != "unknown" and verified_status != "timed out":
                break
            else:
                logging.info(f"                setting {setting_idx} failed")

        self.num_calls_alpha_beta_crown += 1

        if verified_status not in ["safe", "safe-incomplete", "unsafe-pgd", "unsafe-bab", "safe-incomplete (timed out)"]:
            self.setting_idx_for_each_call.append(-1)
        else:
            self.setting_idx_for_each_call.append(setting_idx)

        if verified_status == "unsafe-pgd" or verified_status == "unsafe-bab":
            return False
        elif verified_status == "safe" or verified_status == "safe-incomplete" or verified_status == "safe-incomplete (timed out)":
            return True
        elif verified_status == "unknown" or verified_status == "timed out":
            return None
        else:
            raise NotImplementedError(f"The verification status {verified_status} is not implemented")
    
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
            result = self.check_property(init_box, lbs[i], ">=", index)
            
            if result == None:
                self.error_during_verification = True

                
                error_found_lb = True
                logging.info(f"            error occurs when checking output >= {lbs[i]}: {i}")
                continue
            
            if result:
                logging.info(f"            verified, the lb idx is {i}")
                found_lb = True
                lb_idx = i
                break

        if not found_lb:
            logging.info(f"            the lb is not guaranteed greater or equal to {lbs[0]}")
            lb_idx = -1

            # sometimes the error not occurs during verification but the lb is still not found
            # example: the lb might be less than 0.0, the lb checking will only check till the "lb >= 0.0", and then skip.
            # if there is no error, it means the lb < 0.0
            # else the error occurs, it means the lb might be greater than 0.0
            if error_found_lb:
                logging.info(f"            this bound cannot be verified due to error, return -2")
                lb_idx = -2
        
        ## search the ub
        logging.info(f"        search for ub for idx {index}")
        ### TODO: since the ub_lb might smaller than 0, which will 
        ### cause the left_idx to be negative, 
        ### we need to check if the right_idx is negative first
        #temp = (ub_lb - ubs[0])/ (ubs[0]-lbs[0])
        checked_less_zero = False
        if ub_lb <= 0.0:
            logging.info(f"            checking output <= 0")
            result = self.check_property(init_box, 0.0, "<=", index)
            checked_less_zero = True
            if result == None:
            # if the error occurs, we set the upper bound to 0, this is over-approximation
            # example: if the real ub is less than 0.0, however, we cannot prove it due to error, we use idx=0 as the upper bound
                logging.info(f"            error occurs when checking output <= 0, set the upper bound to 0")
                self.error_during_verification = True
                
            else: 
                if result:
                    logging.info(f"            verified, the ub idx is {-1}")
                    return (-1, -1)
                else:
                    logging.info(f"            the ub is not guaranteed less or equal to {ubs[0]}")
            

        
        left_idx = math.ceil((ub_lb - ubs[0])/(ubs[0]-lbs[0]))
        left_idx = max(left_idx, 0)
        ## the left bound should be less than or equal to the right bound
        left_idx = min(left_idx, ub_ub_idx)

        logging.info(f"            left_idx: {left_idx}, right_idx: {ub_ub_idx}")

        # if the left_idx is equal to the right_idx
        if left_idx == ub_ub_idx:

            if left_idx == 0:
                ## if the left_idx is equal to zero, 
                ## we need to check if the ub is less or equal to zero
                if not checked_less_zero:
                    logging.info(f"            checking output <= 0")
                    result = self.check_property(init_box, 0.0, "<=", index)

                    if result == None:
                    # if the error occurs, we set the upper bound to 0, this is over-approximation
                    # example: if the real ub is less than 0.0, however, we cannot prove it due to error, we use idx=0 as the upper bound
                        logging.info(f"            error occurs when checking output <= 0, set the upper bound to 0")
                        self.error_during_verification = True
                        ub_idx = 0
                
                    else: 
                        if result:
                            logging.info(f"            verified, the ub idx is {-1}")
                            ub_idx = -1
                        else:
                            logging.info(f"            the ub is not guaranteed less or equal to {ubs[0]}")
                            ub_idx = 0
                else:
                    logging.info(f"            error occurs when checking output <= 0 before, set the upper bound to 0")
                    self.error_during_verification = True
                    ub_idx = 0
                    
                
            else: 
                ## if the left_idx is equal to the right_idx and not equal to zero,
                ## ub_idx is equal to the idx of current verified cell, the other operations are not needed
                logging.info(f"            verification is not needed, the ub idx is {left_idx}")
                ub_idx = left_idx
            
        # loop through the idx, the ub_ub_idx is not included because it is the upper bound of the upper bound idx
        for i in range(left_idx, ub_ub_idx):
            logging.info(f"            checking output <= {ubs[i]}: {i}")

            result = self.check_property(init_box, ubs[i], "<=", index)

            if result == None:
                self.error_during_verification = True
                ub_idx = ub_ub_idx
                logging.info(f"            error occurs when checking output <= {ubs[i]}: {i}")
                continue
            
            if result:
                logging.info(f"            verified, the ub idx is {i}")
                ub_idx = i
                break
        
        # it's possible that the ub is not found through the loop, then set the ub to the ub_ub_idx
        if 'ub_idx' not in locals():
            ub_idx = ub_ub_idx
        
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
        samples = self.simulation_samples
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
        torch.cuda.empty_cache()
        d_lb_sim = torch.min(outputs).item()
        d_ub_sim = torch.max(outputs).item()
        logging.info(f"    d_lb_sim: {d_lb_sim}, d_ub_sim: {d_ub_sim}")

        del outputs
        torch.cuda.empty_cache()
        
        #assert d_ub_sim <= d_ub
        if d_lb_sim <= 0.0:
            # the vehicle is already in the danger zone, return (-1, 0, 0, 0)
            return (-1, 0, 0, 0), (-1, 0, 0, 0)


        d_sim_lb_idx = math.floor((d_lb_sim - self.d_lbs[0])/(self.d_ubs[0]-self.d_lbs[0]))
        d_sim_ub_idx = math.ceil((d_ub_sim - self.d_lbs[0])/(self.d_ubs[0]-self.d_lbs[0]))

        ub_ub_idx = d_idx
        d_lb_idx, d_ub_idx = self.get_overlapping_cells(d_lb_sim, d_ub_sim, init_box, ub_ub_idx, index=0)

        # -2 is the error code. 
        # d_ub_idx cannot be -2, because if error occurs, the upper bound is set to the upper bound of the upper bound
        assert d_ub_idx != -2

        # if d_lb_idx == -2, it means that the lower bound cannot be verified due to error,
        # this means that the vehicle might be in the danger zone (overapproximation)
        if d_lb_idx == -2:
            return [-1, 0, 0, 0], [d_sim_lb_idx, d_sim_ub_idx, 0, 0]

        #if d_lb_idx == -2 or d_ub_idx == -2:
        #    # the bounds cannot be verified due to error, return (-2, -2, -2, -2)
        #    return [-2, -2, -2, -2]
        
        # if d_lb_idx == -1, it means that the vehicle is already in the danger zone
        if d_lb_idx == -1:
            # the vehicle is already in the danger zone, return (-1, 0, 0, 0)
            return [-1, 0, 0, 0], [d_sim_lb_idx, d_sim_ub_idx, 0, 0]

        # velocity
        arguments.Config.all_args['model']['name'] = f'Customized("custom_model_data", "MultiStep", index=1, num_steps={self.step})'
        # in order to save the gpu memory, we load the model for each simulation
        model_ori = load_model().cuda()
        model_ori.eval()
        outputs = model_ori(torch.from_numpy(inputs).cuda())
        del model_ori
        torch.cuda.empty_cache()
        v_lb_sim = torch.min(outputs).item()
        v_ub_sim = torch.max(outputs).item()
        logging.info(f"    v_lb_sim: {v_lb_sim}, v_ub_sim: {v_ub_sim}")
        assert v_ub_sim <= v_ub
        del outputs
        torch.cuda.empty_cache()

        v_sim_lb_idx = math.floor((v_lb_sim - self.v_lbs[0])/(self.v_ubs[0]-self.v_lbs[0]))
        v_sim_ub_idx = math.ceil((v_ub_sim - self.v_lbs[0])/(self.v_ubs[0]-self.v_lbs[0]))

        ub_ub_idx = v_idx
        v_lb_idx, v_ub_idx = self.get_overlapping_cells(v_lb_sim, v_ub_sim, init_box, ub_ub_idx, index=1)

        # -2 is the error code. 
        # v_ub_idx cannot be -2, because if error occurs, the upper bound is set to the upper bound of the upper bound
        assert v_ub_idx != -2

        # if v_lb_idx == -2, it means that the lower bound cannot be verified due to error,
        # this means that the vehicle might be in the safe zone (overapproximation)
        if v_lb_idx == -2:
            return [d_lb_idx, d_ub_idx, -1, v_ub_idx], [d_sim_lb_idx, d_sim_ub_idx, v_sim_lb_idx, v_sim_ub_idx]
        
        #if v_lb_idx == -2 or v_ub_idx == -2:
        #    # the bounds cannot be verified due to error, return (-2, -2, -2, -2)
        #    return [-2, -2, -2, -2]

        if v_ub_idx == -1:
            return [d_lb_idx, d_ub_idx, -1, -1], [d_sim_lb_idx, d_sim_ub_idx, v_sim_lb_idx, v_sim_ub_idx]

        return [d_lb_idx, d_ub_idx, v_lb_idx, v_ub_idx], [d_sim_lb_idx, d_sim_ub_idx, v_sim_lb_idx, v_sim_ub_idx]
    
    def compute_next_reachable_cells(self, d_idx, v_idx):
        result_dict = dict()

        if hasattr(self, 'reachable_cells'):
            result_dict["reachable_cells"] = self.reachable_cells[(d_idx, v_idx)]
            return result_dict
        

        time_dict = dict()
        self.num_calls_alpha_beta_crown = 0
        self.setting_idx_for_each_call = []
        self.error_during_verification = False

        time_start = time.time()
        interval, sim_interval = self.get_intervals(d_idx, v_idx)
        time_end_get_intervals = time.time()
        logging.info(f"    Interval index: {interval}")

        result_dict["sim_interval"] = sim_interval
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
        result_dict["setting_idx_for_each_call"] = self.setting_idx_for_each_call
        result_dict["error_during_verification"] = self.error_during_verification
        return result_dict


def compute_unsafe_cells(reachable_sets, d_lbs, d_ubs, v_lbs, v_ubs):
    isSafe = np.ones((len(d_lbs), len(v_lbs)))
    reversed_reachable_sets = defaultdict(set)
    helper = set()
    new_unsafe_state = []

    for d_idx in tqdm(range(len(d_lbs))):
        for v_idx in tqdm(range(len(v_lbs)), leave=False):
            
            if reachable_sets[(d_idx, v_idx)] == {(-4, -4)}:
                isSafe[d_idx, v_idx] = 0
                helper.add((d_idx, v_idx))
                new_unsafe_state.append((d_idx, v_idx))
                continue
                
            for reachable_cell in reachable_sets[(d_idx, v_idx)]:

                assert len(d_lbs)>=reachable_cell[0] >= 0, f"reachable_cell: {reachable_cell}"
                assert len(v_lbs)>=reachable_cell[1] >= 0, f"reachable_cell: {reachable_cell}"
                reversed_reachable_sets[reachable_cell].add((d_idx, v_idx))
    
    while len(new_unsafe_state)>0:
        temp = []
        for i, j in new_unsafe_state:
            for (_i, _j) in reversed_reachable_sets[(i, j)]:
                if (_i, _j) not in helper:
                    isSafe[_i, _j] = 0
                    temp.append((_i, _j))
                    helper.add((_i, _j))
        new_unsafe_state = temp
    
    return isSafe
