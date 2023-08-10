from load_model import load_model
import arguments
import numpy as np
import torch
from matplotlib import pyplot as plt

def main():
    # Create the argument parser
    args = arguments.Config
    args.parse_config()

    # distance model
    arguments.Config.all_args['model']['name'] = f'Customized("custom_model_data", "MultiStep", index=0, num_steps=1)'
    model_d = load_model().cuda()
    model_d.eval()

    # velocity model
    arguments.Config.all_args['model']['name'] = f'Customized("custom_model_data", "MultiStep", index=1, num_steps=1)'
    model_v = load_model().cuda()
    model_v.eval()

    num_samples = 1000
    d_0 = np.random.uniform(59.4, 60, num_samples)
    v_0 = np.random.uniform(0, 0.3, num_samples)
    z0 = np.random.uniform(-0.01, 0.01, (num_samples, 4))
    z1 = np.random.uniform(-0.01, 0.01, (num_samples, 4))

    d_1 = model_d(torch.from_numpy(np.concatenate([d_0.reshape(-1, 1), v_0.reshape(-1, 1), z0], axis=1)).float().cuda()).cpu().detach().numpy()
    v_1 = model_v(torch.from_numpy(np.concatenate([d_0.reshape(-1, 1), v_0.reshape(-1, 1), z0], axis=1)).float().cuda()).cpu().detach().numpy()

    d_2 = model_d(torch.from_numpy(np.concatenate([d_1, v_1, z1], axis=1)).float().cuda()).cpu().detach().numpy()
    v_2 = model_v(torch.from_numpy(np.concatenate([d_1, v_1, z1], axis=1)).float().cuda()).cpu().detach().numpy()

    # plot 0 step
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    cell_width = 0.6
    cell_height = 0.3

    safe_rect = plt.Rectangle((58.0, -1.0), 2.0, 1.0, color='g', fill=True, edgecolor='none', alpha=0.2)
    ax.add_patch(safe_rect)
    
    
    ## add cells
    cells = [(59.4, 0.0)]
    for cell in cells:
        rect = plt.Rectangle(cell, cell_width, cell_height, color='r', fill=True, edgecolor='none')
        ax.add_patch(rect)

    ## add simulations
    ax.scatter(d_0, v_0, s=0.5, c='b')
    xticks = [58.2, 58.8, 59.4, 60.0, 60.6]
    yticks = [-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    
    ## plot grids
    for x in xticks:
        X = [x, x]
        Y = [-1.0, 1.0]
        ax.plot(X, Y, color='lightgray', alpha=0.2)
    
    for y in yticks:
        X = [58.0, 61.0]
        Y = [y, y]
        ax.plot(X, Y, color='lightgray', alpha=0.2)

    plt.xlim(58.0, 60.0)
    plt.ylim(-1.0, 1.0)
    
    ax.set_title('step 0') 
    ax.set_xlabel(r"$d$ (m)")
    ax.set_ylabel(r"$v$ (m/s)")
    fig.savefig('0_step.png')

    # plot 1 step
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    cell_width = 0.6
    cell_height = 0.3

    safe_rect = plt.Rectangle((58.0, -1.0), 2.0, 1.0, color='g', fill=True, edgecolor='none', alpha=0.2)
    ax.add_patch(safe_rect)

    ## add cells
    cells = [(59.4, 0.0), (59.4, -0.3), (58.8, 0.0), (58.8, -0.3)]
    for cell in cells:
        rect = plt.Rectangle(cell, cell_width, cell_height, facecolor='r', fill=True, edgecolor='g')
        ax.add_patch(rect)

    ## add simulations
    ax.scatter(d_1, v_1, s=0.5, c='b')
    xticks = [58.2, 58.8, 59.4, 60.0, 60.6]
    yticks = [-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    
    ## plot grids
    for x in xticks:
        X = [x, x]
        Y = [-1.0, 1.0]
        ax.plot(X, Y, color='lightgray', alpha=0.2)
    
    for y in yticks:
        X = [58.0, 61.0]
        Y = [y, y]
        ax.plot(X, Y, color='lightgray', alpha=0.2)
    
    ax.set_title('step 1 (one-step method)') 
    ax.set_xlabel(r"$d$ (m)")
    ax.set_ylabel(r"$v$ (m/s)")
    plt.xlim(58.0, 60.0)
    plt.ylim(-1.0, 1.0)
    fig.savefig('1_step.png')


    # plot 2 step for one-step method
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    cell_width = 0.6
    cell_height = 0.3

    safe_rect = plt.Rectangle((58.0, -1.0), 2.0, 1.0, color='g', fill=True, edgecolor='none', alpha=0.2)
    ax.add_patch(safe_rect)

    ## add one-step method cells
    cells = [(59.4, 0.0), (59.4, -0.3), (59.4, -0.6), (58.8, 0.0), (58.8, -0.3), (58.8, -0.6), (58.2, 0.0), (58.2, -0.3), (58.2, -0.6)]
    for cell in cells:
        rect = plt.Rectangle(cell, cell_width, cell_height, facecolor='r', fill=True, edgecolor='g')
        ax.add_patch(rect)

    ### add two-step method cells
    #cells = [(59.4, -0.6), (59.4, -0.3), (58.8, -0.6), (58.8, -0.3)]
    #for cell in cells:
    #    rect = plt.Rectangle(cell, cell_width, cell_height, facecolor='r', fill=True, edgecolor='g')
    #    ax.add_patch(rect)

    ## add simulations
    ax.scatter(d_2, v_2, s=0.5, c='b')
    xticks = [58.2, 58.8, 59.4, 60.0, 60.6]
    yticks = [-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    
    ## plot grids
    for x in xticks:
        X = [x, x]
        Y = [-1.0, 1.0]
        ax.plot(X, Y, color='lightgray', alpha=0.2)
    
    for y in yticks:
        X = [58.0, 61.0]
        Y = [y, y]
        ax.plot(X, Y, color='lightgray', alpha=0.2)
    
    ax.set_title('step 2 (one-step method)') 
    ax.set_xlabel(r"$d$ (m)")
    ax.set_ylabel(r"$v$ (m/s)")
    plt.xlim(58.0, 60.0)
    plt.ylim(-1.0, 1.0)
    fig.savefig('2_step_single_step_method.png')


    # plot 2 step for two-step method
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    cell_width = 0.6
    cell_height = 0.3

    safe_rect = plt.Rectangle((58.0, -1.0), 2.0, 1.0, color='g', fill=True, edgecolor='none', alpha=0.2)
    ax.add_patch(safe_rect)


    ### add two-step method cells
    cells = [(59.4, -0.6), (59.4, -0.3), (58.8, -0.6), (58.8, -0.3)]
    for cell in cells:
        rect = plt.Rectangle(cell, cell_width, cell_height, facecolor='r', fill=True, edgecolor='g')
        ax.add_patch(rect)

    ## add simulations
    ax.scatter(d_2, v_2, s=0.5, c='b')
    xticks = [58.2, 58.8, 59.4, 60.0, 60.6]
    yticks = [-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    
    ## plot grids
    for x in xticks:
        X = [x, x]
        Y = [-1.0, 1.0]
        ax.plot(X, Y, color='lightgray', alpha=0.2)
    
    for y in yticks:
        X = [58.0, 61.0]
        Y = [y, y]
        ax.plot(X, Y, color='lightgray', alpha=0.2)
    
    ax.set_title('step 2 (two-step method)') 
    ax.set_xlabel(r"$d$ (m)")
    ax.set_ylabel(r"$v$ (m/s)")
    plt.xlim(58.0, 60.0)
    plt.ylim(-1.0, 1.0)
    fig.savefig('2_step_two_step_method.png')


if __name__ == '__main__':
    main()