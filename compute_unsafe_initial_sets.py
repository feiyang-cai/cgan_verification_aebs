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

    # Parse the command-line arguments
    args = parser.parse_args()

    d_range_lb = args.d_range_lb
    d_range_ub = args.d_range_ub
    d_num_bin = args.d_num_bin
    v_range_lb = args.v_range_lb
    v_range_ub = args.v_range_ub
    v_num_bin = args.v_num_bin
    frequency = args.frequency
    assert frequency == 20 or frequency == 10 or frequency == 5

    d_bins = np.linspace(d_range_lb, d_range_ub, d_num_bin+1, endpoint=True)
    d_lbs = np.array(d_bins[:-1],dtype=np.float32)
    d_ubs = np.array(d_bins[1:], dtype=np.float32)

    v_bins = np.linspace(v_range_lb, v_range_ub, v_num_bin+1, endpoint=True)
    v_lbs = np.array(v_bins[:-1],dtype=np.float32)
    v_ubs = np.array(v_bins[1:], dtype=np.float32)

    results_dir = f"./results/{frequency}Hz/unsafe_initial_cells/d_num_bin_{d_num_bin}_v_num_bin_{v_num_bin}"
    os.makedirs(results_dir, exist_ok=True)

    # one-step method
    reachable_sets = defaultdict(set)
    step_1_results_path = f"./results/{frequency}Hz/reachable_sets_graph/step_1"
    reachable_sets_pickle_file = os.path.join(step_1_results_path, "reachable_sets.pkl")
    one_step_verifier = MultiStepVerifier(d_lbs, d_ubs, v_lbs, v_ubs, reachable_cells_path=reachable_sets_pickle_file)

    reachable_sets = defaultdict(set)
    reachable_sets_one_step = defaultdict(set)

    for d_idx in range(len(d_lbs)):
        for v_idx in range(len(v_lbs)):
            results = one_step_verifier.compute_next_reachable_cells(d_idx, v_idx)
            if results['reachable_cells'] == {(-2, -2)}:
                reachable_sets[(d_idx, v_idx)] = {(-4, -4)} # this (-4, -4) means the cell is unsafe
                reachable_sets_one_step[(d_idx, v_idx)] = {(-4, -4)}
            else:
                reachable_sets_one_step[(d_idx, v_idx)] = results['reachable_cells']
                if (-3, -3) in reachable_sets_one_step[(d_idx, v_idx)]:
                    reachable_sets_one_step[(d_idx, v_idx)].remove((-3, -3))
                assert (-1, -1) not in reachable_sets_one_step[(d_idx, v_idx)]

    isSafe = compute_unsafe_cells(reachable_sets_one_step, d_lbs, d_ubs, v_lbs, v_ubs)
    unsafe_cells = []
    for p_idx in range(len(d_lbs)):
        for theta_idx in range(len(v_lbs)):
            if isSafe[p_idx, theta_idx] == 0:
                unsafe_cells.append((p_idx, theta_idx))
    print(f"Number of unsafe cells for one-step method: {len(unsafe_cells)}") 
    plotter = Plotter(d_lbs, v_lbs)
    plotter.add_cells(unsafe_cells, color='red', filled=True, label='Inconclusive cells')
    file_path = os.path.join(results_dir, f"unsafe_cells_one_step.png")
    plotter.save_figure(file_path, x_range=(0, 60), y_range=(0, 30))

    # two-step method
    reachable_sets_two_step = defaultdict(set)
    step_2_results_path = f"./results/{frequency}Hz/reachable_sets_graph/step_2"
    reachable_sets_pickle_file = os.path.join(step_2_results_path, "reachable_sets.pkl")
    two_step_verifier = MultiStepVerifier(d_lbs, d_ubs, v_lbs, v_ubs, reachable_cells_path=reachable_sets_pickle_file)

    for d_idx in range(len(d_lbs)):
        for v_idx in range(len(v_lbs)):
            if reachable_sets[(d_idx, v_idx)] == {(-4, -4)}:
                reachable_sets_two_step[(d_idx, v_idx)] = {(-4, -4)}
                continue
            results = two_step_verifier.compute_next_reachable_cells(d_idx, v_idx)
            if results['reachable_cells'] == {(-2, -2)}:
                reachable_sets[(d_idx, v_idx)] = {(-4, -4)} # this (-4, -4) means the cell is unsafe
                reachable_sets_two_step[(d_idx, v_idx)] = {(-4, -4)}
            else:
                reachable_sets_two_step[(d_idx, v_idx)] = results['reachable_cells']
                if (-3, -3) in reachable_sets_two_step[(d_idx, v_idx)]:
                    reachable_sets_two_step[(d_idx, v_idx)].remove((-3, -3))
                assert (-1, -1) not in reachable_sets_two_step[(d_idx, v_idx)], f"({d_idx}, {v_idx}) has (-1, -1)"

    
    isSafe = compute_unsafe_cells(reachable_sets_two_step, d_lbs, d_ubs, v_lbs, v_ubs)
    unsafe_cells = []
    for p_idx in range(len(d_lbs)):
        for theta_idx in range(len(v_lbs)):
            if isSafe[p_idx, theta_idx] == 0:
                unsafe_cells.append((p_idx, theta_idx))
    print(f"Number of unsafe cells for two-step method: {len(unsafe_cells)}") 
    plotter = Plotter(d_lbs, v_lbs)
    plotter.add_cells(unsafe_cells, color='red', filled=True, label='Inconclusive cells')
    file_path = os.path.join(results_dir, f"unsafe_cells_two_step.png")
    plotter.save_figure(file_path, x_range=(0, 60), y_range=(0, 30))
    #file_path = os.path.join(results_dir, f"unsafe_cells.png")
    
    # three-step method
    reachable_sets_three_step = defaultdict(set)
    step_3_results_path = f"./results/{frequency}Hz/reachable_sets_graph/step_3"
    reachable_sets_pickle_file = os.path.join(step_3_results_path, "reachable_sets.pkl")
    three_step_verifier = MultiStepVerifier(d_lbs, d_ubs, v_lbs, v_ubs, reachable_cells_path=reachable_sets_pickle_file)

    for d_idx in range(len(d_lbs)):
        for v_idx in range(len(v_lbs)):
            if reachable_sets[(d_idx, v_idx)] == {(-4, -4)}:
                reachable_sets_three_step[(d_idx, v_idx)] = {(-4, -4)}
                continue
            results = three_step_verifier.compute_next_reachable_cells(d_idx, v_idx)
            if results['reachable_cells'] == {(-2, -2)}:
                reachable_sets[(d_idx, v_idx)] = {(-4, -4)} # this (-4, -4) means the cell is unsafe
                reachable_sets_three_step[(d_idx, v_idx)] = {(-4, -4)}
            else:
                reachable_sets_three_step[(d_idx, v_idx)] = results['reachable_cells']
                assert (-1, -1) not in reachable_sets_three_step[(d_idx, v_idx)]
                if (-3, -3) in reachable_sets_three_step[(d_idx, v_idx)]:
                    reachable_sets_three_step[(d_idx, v_idx)].remove((-3, -3))
    
    isSafe = compute_unsafe_cells(reachable_sets_three_step, d_lbs, d_ubs, v_lbs, v_ubs)
    unsafe_cells = []
    for p_idx in range(len(d_lbs)):
        for theta_idx in range(len(v_lbs)):
            if isSafe[p_idx, theta_idx] == 0:
                unsafe_cells.append((p_idx, theta_idx))
    print(f"Number of unsafe cells for two-step method: {len(unsafe_cells)}") 
    plotter = Plotter(d_lbs, v_lbs)
    plotter.add_cells(unsafe_cells, color='red', filled=True, label='Inconclusive cells')
    file_path = os.path.join(results_dir, f"unsafe_cells_three_step.png")
    plotter.save_figure(file_path, x_range=(0, 60), y_range=(0, 30))
if __name__ == '__main__':
    main()