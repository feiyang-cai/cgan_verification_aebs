import argparse
import numpy as np
from src.utils import *
import os
import pickle
from collections import defaultdict


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Description of your program')

    # Add the arguments
    parser.add_argument('--d_range_lb', type=float, default=0.0, help='Lower bound for d_range')
    parser.add_argument('--d_range_ub', type=float, default=60.0, help='Upper bound for d_range')
    parser.add_argument('--d_num_bin', type=int, default=100, help='Number of bins for d')
    parser.add_argument('--v_range_lb', type=float, default=0.0, help='Lower bound for v_range')
    parser.add_argument('--v_range_ub', type=float, default=+30.0, help='Upper bound for v_range')
    parser.add_argument('--v_num_bin', type=int, default=100, help='Number of bins for v')
    parser.add_argument('--frequency', type=int, default=20, help='Frequency of the control loop')

    args = parser.parse_args()
    
    d_num_bin = args.d_num_bin
    v_num_bin = args.v_num_bin
    frequency = args.frequency

    # one_step_method
    step_1_results_path = f"./results/{frequency}Hz/reachable_sets_graph/step_1"
    whole_graph_pickle_file = os.path.join(step_1_results_path, "whole_graph.pkl")

    if os.path.exists(whole_graph_pickle_file):
        print(f"File {whole_graph_pickle_file} exists")
    else:
        # combine the files from different runs
        file_complete = True
        
        whole_graph_data = dict()
        for d_idx in range(d_num_bin):
            for v_idx in range(v_num_bin):
                pickle_file = os.path.join(step_1_results_path, f"results_d_idx_{d_idx}_v_idx_{v_idx}.pkl")
                if os.path.exists(pickle_file):
                    with open(pickle_file, "rb") as f:
                        data = pickle.load(f)[0]
                        reachable_set = data['reachable_cells']
                        whole_graph_data[(d_idx, v_idx)] = data
                else:
                    print(f"File {pickle_file} does not exist")
                    file_complete = False
                    break
        
        if file_complete:
            with open(whole_graph_pickle_file, "wb") as f:
                pickle.dump(whole_graph_data, f)

    # two_step_method
    step_2_results_path = f"./results/{frequency}Hz/reachable_sets_graph/step_2"
    whole_graph_pickle_file = os.path.join(step_2_results_path, "whole_graph.pkl")

    if os.path.exists(whole_graph_pickle_file):
        print(f"File {whole_graph_pickle_file} exists")
    else:
        # combine the files from different runs
        file_complete = True
        
        whole_graph_data = dict()
        for d_idx in range(d_num_bin):
            for v_idx in range(v_num_bin):
                pickle_file = os.path.join(step_2_results_path, f"results_d_idx_{d_idx}_v_idx_{v_idx}.pkl")
                if os.path.exists(pickle_file):
                    with open(pickle_file, "rb") as f:
                        data = pickle.load(f)[0]
                        reachable_set = data['reachable_cells']
                        error_during_verification = data['error_during_verification']
                        if error_during_verification:
                            print(reachable_set)
                            print(data['sim_interval'])
                            print("----------------")
                        whole_graph_data[(d_idx, v_idx)] = data
                else:
                    print(f"File {pickle_file} does not exist")
                    file_complete = False
                    break
        
        if file_complete:
            with open(whole_graph_pickle_file, "wb") as f:
                pickle.dump(whole_graph_data, f)

if __name__ == '__main__':
    main()