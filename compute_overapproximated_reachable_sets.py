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
    parser.add_argument('--d_lb', type=float, default=20.0, help='Lower bound of d')
    parser.add_argument('--d_ub', type=float, default=30.0, help='Upper bound of d')
    parser.add_argument('--v_lb', type=float, default=20.0, help='Lower bound of v')
    parser.add_argument('--v_ub', type=float, default=30.0, help='Upper bound of v')
    parser.add_argument('--d_range_lb', type=float, default=0.0, help='Lower bound for d_range')
    parser.add_argument('--d_range_ub', type=float, default=60.0, help='Upper bound for d_range')
    parser.add_argument('--d_num_bin', type=int, default=100, help='Number of bins for d')
    parser.add_argument('--v_range_lb', type=float, default=0.0, help='Lower bound for v_range')
    parser.add_argument('--v_range_ub', type=float, default=+30.0, help='Upper bound for v_range')
    parser.add_argument('--v_num_bin', type=int, default=100, help='Number of bins for v')
    parser.add_argument('--add_random_simulations', type=bool, default=True, help='Add random simulations')
    parser.add_argument('--add_eagerly_searching_simulations', type=bool, default=False, help='Add eagerly serching simulations')

    args = parser.parse_args()

    d_lb = args.d_lb
    d_ub = args.d_ub
    v_lb = args.v_lb
    v_ub = args.v_ub
    
    d_range_lb = args.d_range_lb
    d_range_ub = args.d_range_ub
    d_num_bin = args.d_num_bin
    v_range_lb = args.v_range_lb
    v_range_ub = args.v_range_ub
    v_num_bin = args.v_num_bin

    assert d_lb >= d_range_lb and d_ub <= d_range_ub
    assert v_lb >= v_range_lb and v_ub <= v_range_ub

    d_bins = np.linspace(d_range_lb, d_range_ub, d_num_bin+1, endpoint=True)
    d_lbs = np.array(d_bins[:-1],dtype=np.float32)
    d_ubs = np.array(d_bins[1:], dtype=np.float32)

    v_bins = np.linspace(v_range_lb, v_range_ub, v_num_bin+1, endpoint=True)
    v_lbs = np.array(v_bins[:-1],dtype=np.float32)
    v_ubs = np.array(v_bins[1:], dtype=np.float32)

    results_dir = f"./results/compute_overapproximated_reachable_sets/d_lb_{d_lb}_d_ub_{d_ub}_v_lb_{v_lb}_v_ub_{v_ub}_d_range_lb_{d_range_lb}_d_range_ub_{d_range_ub}_d_num_bin_{d_num_bin}_v_range_lb_{v_range_lb}_v_range_ub_{v_range_ub}_v_num_bin_{v_num_bin}"
    os.makedirs(results_dir, exist_ok=True)

    # one_step_method
    step_1_results_path = f"./results/reachable_sets_graph/step_1"
    reachable_sets_pickle_file = os.path.join(step_1_results_path, "reachable_sets.pkl")

    use_one_step_method = False
    if os.path.exists(reachable_sets_pickle_file):
        print(f"File {reachable_sets_pickle_file} exists")
        use_one_step_method = True
    else:
        # combine the files from different runs
        file_complete = True
        
        reachable_sets = defaultdict(set)
        for d_idx in range(d_num_bin):
            for v_idx in range(v_num_bin):
                pickle_file = os.path.join(step_1_results_path, f"results_d_idx_{d_idx}_v_idx_{v_idx}.pkl")
                if os.path.exists(pickle_file):
                    with open(pickle_file, "rb") as f:
                        reachable_set = pickle.load(f)[0]['reachable_cells']
                        reachable_sets[(d_idx, v_idx)] = reachable_set
                else:
                    print(f"File {pickle_file} does not exist")
                    file_complete = False
                    break
        
        if file_complete:
            with open(reachable_sets_pickle_file, "wb") as f:
                pickle.dump(reachable_sets, f)
            use_two_step_method = True

    # two_step_method
    step_2_results_path = f"./results/reachable_sets_graph/step_2"
    reachable_sets_pickle_file = os.path.join(step_2_results_path, "reachable_sets.pkl")

    use_two_step_method = False
    if os.path.exists(reachable_sets_pickle_file):
        print(f"File {reachable_sets_pickle_file} exists")
        use_two_step_method = True
    else:
        # combine the files from different runs
        file_complete = True
        
        reachable_sets = defaultdict(set)
        for d_idx in range(d_num_bin):
            for v_idx in range(v_num_bin):
                pickle_file = os.path.join(step_2_results_path, f"results_d_idx_{d_idx}_v_idx_{v_idx}.pkl")
                if os.path.exists(pickle_file):
                    with open(pickle_file, "rb") as f:
                        reachable_set = pickle.load(f)[0]['reachable_cells']
                        reachable_sets[(d_idx, v_idx)] = reachable_set
                else:
                    print(f"File {pickle_file} does not exist")
                    file_complete = False
                    break
        
        if file_complete:
            with open(reachable_sets_pickle_file, "wb") as f:
                pickle.dump(reachable_sets, f)
            use_two_step_method = True

if __name__ == '__main__':
    main()