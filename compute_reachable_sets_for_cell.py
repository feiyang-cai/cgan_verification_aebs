import numpy as np
import os
from datetime import datetime
import logging
from tqdm import tqdm
from collections import defaultdict
import pickle
import arguments


def main():
    # Create the argument parser
    args = arguments.Config
    h = ['system parameters']
    args.add_argument('--d_idx', type=int, default=50, help='Index of d.', hierarchy=h + ['d_idx'])
    args.add_argument('--v_idx', type=int, default=50, help='Index of v.', hierarchy=h + ['v_idx'])
    args.add_argument('--d_range_lb', type=float, default=0.0, help='Lower bound for d.', hierarchy=h + ['d_lb'])
    args.add_argument('--d_range_ub', type=float, default=+60.0, help='Upper bound for d.', hierarchy=h + ['d_ub'])
    args.add_argument('--d_num_bin', type=int, default=100, help='Number of bins for d.', hierarchy=h + ['d_num_bin'])
    args.add_argument('--v_range_lb', type=float, default=0.0, help='Lower bound for v.', hierarchy=h + ['v_lb'])
    args.add_argument('--v_range_ub', type=float, default=+30.0, help='Upper bound for v.', hierarchy=h + ['v_ub'])
    args.add_argument('--v_num_bin', type=int, default=100, help='Number of bins for v.', hierarchy=h + ['v_num_bin'])
    args.add_argument('--reachability_steps', type=int, default=1, help='Number of reachability steps.', hierarchy=h + ['reachability_steps'])
    args.add_argument('--latent_bounds', type=float, default=0.01, help='Bounds for latent variables.', hierarchy=h + ['latent_bounds'])
    args.add_argument('--tolerance_ratio', type=float, default=0.1, help='Tolerance ratio for binary search.', hierarchy=h + ['tolerance_ratio'])
    args.add_argument('--binary_search_divide_ratio', type=float, default=0.5, help='Divide ratio for binary search.', hierarchy=h + ['binary_search_divide_ratio'])
    args.parse_config()

    d_idx = args['system parameters']['d_idx']
    v_idx = args['system parameters']['v_idx']
    d_range_lb = args['system parameters']['d_lb']
    d_range_ub = args['system parameters']['d_ub']
    d_num_bin = args['system parameters']['d_num_bin']
    v_range_lb = args['system parameters']['v_lb']
    v_range_ub = args['system parameters']['v_ub']
    v_num_bin = args['system parameters']['v_num_bin']
    latent_bounds = args['system parameters']['latent_bounds']
    reachability_steps = args['system parameters']['reachability_steps']
    tolerance_ratio = args['system parameters']['tolerance_ratio']
    binary_search_divide_ratio = args['system parameters']['binary_search_divide_ratio']

    result_path = f"./results/reachable_sets_cell/step_{reachability_steps}/"
    os.makedirs(result_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(result_path, f"log_{timestamp}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_filename,
        filemode='a'
    )
    from src.utils import MultiStepVerifier


    d_bins = np.linspace(d_range_lb, d_range_ub, d_num_bin+1, endpoint=True)
    d_lbs = np.array(d_bins[:-1],dtype=np.float32)
    d_ubs = np.array(d_bins[1:], dtype=np.float32)

    v_bins = np.linspace(v_range_lb, v_range_ub, v_num_bin+1, endpoint=True)
    v_lbs = np.array(v_bins[:-1],dtype=np.float32)
    v_ubs = np.array(v_bins[1:], dtype=np.float32)

    verifier = MultiStepVerifier(d_lbs, d_ubs, v_lbs, v_ubs, reachability_steps, latent_bounds, tolerance_ratio, binary_search_divide_ratio)
    logging.info(f"Computing reachable set for cell ({d_idx}, {v_idx})")
    results = verifier.compute_next_reachable_cells(d_idx, v_idx)
    reachable_cells = results["reachable_cells"]
    time = results["time"]["whole_time"]
    num_calls_alpha_beta_crown = results["num_calls_alpha_beta_crown"]
    logging.info(f"    Taking {time} seconds and calling alpha-beta-crown {num_calls_alpha_beta_crown} times")
    logging.info(f"    Reachable cells: {reachable_cells}")

if __name__ == '__main__':
    main()