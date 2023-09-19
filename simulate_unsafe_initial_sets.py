import argparse
import numpy as np
from src.utils import *
import os
import pickle
from collections import defaultdict
from tqdm import tqdm
from load_model import load_model
import arguments


def main():
    # Create the argument parser
    args = arguments.Config
    h = ['system parameters']
    args.add_argument('--d_range_lb', type=float, default=0.0, help='Lower bound for d.', hierarchy=h + ['d_lb'])
    args.add_argument('--d_range_ub', type=float, default=+60.0, help='Upper bound for d.', hierarchy=h + ['d_ub'])
    args.add_argument('--d_num_bin', type=int, default=100, help='Number of bins for d.', hierarchy=h + ['d_num_bin'])
    args.add_argument('--v_range_lb', type=float, default=0.0, help='Lower bound for v.', hierarchy=h + ['v_lb'])
    args.add_argument('--v_range_ub', type=float, default=+30.0, help='Upper bound for v.', hierarchy=h + ['v_ub'])
    args.add_argument('--v_num_bin', type=int, default=100, help='Number of bins for v.', hierarchy=h + ['v_num_bin'])
    args.add_argument('--latent_bounds', type=float, default=0.01, help='Bounds for latent variables.', hierarchy=h + ['latent_bounds'])
    args.add_argument('--simulation_samples', type=int, default=5000, help='Number of simulation samples.', hierarchy=h + ['simulation_samples'])
    args.add_argument('--frequency', type=int, default=20, help='Frequency of the control loop.', hierarchy=h + ['frequency'])
    args.add_argument('--ViT', action='store_true', help='Use ViT to compute the reachable set.', hierarchy=h + ['ViT'])
    args.parse_config()

    d_range_lb = args['system parameters']['d_lb']
    d_range_ub = args['system parameters']['d_ub']
    d_num_bin = args['system parameters']['d_num_bin']
    v_range_lb = args['system parameters']['v_lb']
    v_range_ub = args['system parameters']['v_ub']
    v_num_bin = args['system parameters']['v_num_bin']
    latent_bounds = args['system parameters']['latent_bounds']
    simulation_samples = args['system parameters']['simulation_samples']
    frequency = args['system parameters']['frequency']
    is_ViT = args['system parameters']['ViT']
    assert frequency == 20 or frequency == 10 or frequency == 5

    d_bins = np.linspace(d_range_lb, d_range_ub, d_num_bin+1, endpoint=True)
    d_lbs = np.array(d_bins[:-1],dtype=np.float32)
    d_ubs = np.array(d_bins[1:], dtype=np.float32)

    v_bins = np.linspace(v_range_lb, v_range_ub, v_num_bin+1, endpoint=True)
    v_lbs = np.array(v_bins[:-1],dtype=np.float32)
    v_ubs = np.array(v_bins[1:], dtype=np.float32)

    if is_ViT:
        results_dir = f"./results/ViT/{frequency}Hz/unsafe_initial_cells/d_num_bin_{d_num_bin}_v_num_bin_{v_num_bin}"
    else:
        results_dir = f"./results/{frequency}Hz/unsafe_initial_cells/d_num_bin_{d_num_bin}_v_num_bin_{v_num_bin}"
    os.makedirs(results_dir, exist_ok=True)

    isSafe = np.ones((len(d_lbs), len(v_lbs)))
    if is_ViT:
        arguments.Config.all_args['model']['path'] = f'./models/single_step_vit_{frequency}Hz.pth'
    else:
        arguments.Config.all_args['model']['path'] = f'./models/single_step_{frequency}Hz.pth'

    # load model
    if is_ViT:
        arguments.Config.all_args['model']['name'] = f'Customized("custom_model_data", "MultiStepViT", index=0, num_steps=1)'
    else:
        arguments.Config.all_args['model']['name'] = f'Customized("custom_model_data", "MultiStep", index=0, num_steps=1)'
    model_dist = load_model().cuda()
    model_dist.eval()

    if is_ViT:
        arguments.Config.all_args['model']['name'] = f'Customized("custom_model_data", "MultiStepViT", index=1, num_steps=1)'
    else:
        arguments.Config.all_args['model']['name'] = f'Customized("custom_model_data", "MultiStep", index=1, num_steps=1)'
    model_vel = load_model().cuda()
    model_vel.eval()

    for d_idx in tqdm(range(len(d_lbs))):
        for v_idx in tqdm(range(len(v_lbs))):
            d_lb = d_lbs[d_idx]
            d_ub = d_ubs[d_idx]
            v_lb = v_lbs[v_idx]
            v_ub = v_ubs[v_idx]
            ds = np.random.uniform(d_lb, d_ub, (simulation_samples, 1)).astype(np.float32)
            vs = np.random.uniform(v_lb, v_ub, (simulation_samples, 1)).astype(np.float32)
            x = np.concatenate([ds, vs], axis=1)
            x = torch.from_numpy(x).cuda() # (simulation_samples, 2)
            dones = np.zeros((simulation_samples, 1))

            step = 0
            while True:
                z = torch.from_numpy(np.random.uniform(-latent_bounds, latent_bounds, (simulation_samples, 4)).astype(np.float32)).cuda()
                inputs = torch.cat([x, z], dim=1)
                with torch.no_grad():
                    ds_ = model_dist(inputs)
                    vs_ = model_vel(inputs)
                    done_d = torch.nonzero(ds_ < 0.0).cpu().numpy()
                    done_v = torch.nonzero(vs_ < 0.0).cpu().numpy()
                unsafe = np.any(np.logical_not(dones[done_d[:, 0]]))
                if unsafe:
                    isSafe[d_idx, v_idx] = 0
                    break
                dones[done_d[:, 0]] = 1
                dones[done_v[:, 0]] = 1
                step += 1
                if np.all(dones):
                    break
                x = torch.cat([ds_, vs_], dim=1)
        np.save(os.path.join(results_dir, f"isSafe_simulated_{d_idx}.npy"), isSafe)
    np.save(os.path.join(results_dir, f"isSafe_simulated.npy"), isSafe)

    unsafe_cells = []
    for p_idx in range(len(d_lbs)):
        for theta_idx in range(len(v_lbs)):
            if isSafe[p_idx, theta_idx] == 0:
                unsafe_cells.append((p_idx, theta_idx))
    print(f"Number of unsafe cells for simulated method: {len(unsafe_cells)}") 
    plotter = Plotter(d_lbs, v_lbs)
    plotter.add_cells(unsafe_cells, color='red', filled=True, label='Inconclusive cells')
    file_path = os.path.join(results_dir, f"unsafe_cells_simulated.png")
    plotter.save_figure(file_path, x_range=(0, 60), y_range=(0, 30))
    #file_path = os.path.join(results_dir, f"unsafe_cells.png")
    
if __name__ == '__main__':
    main()