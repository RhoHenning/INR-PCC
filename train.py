import os
import argparse
import yaml
import torch
import torch.optim as optim
from logger import Logger
from model import Representation
from sampler import GeometrySampler, AttributeSampler

def focal_loss(probs, occupancies, gamma=2, eps=1e-5, alpha=None):
    if alpha is None:
        alpha = 1 - torch.mean(occupancies)
    probs = probs.squeeze(dim=-1)
    alpha_t = alpha * occupancies + (1 - alpha) * (1 - occupancies)
    probs_t = probs * occupancies + (1 - probs) * (1 - occupancies)
    loss = -alpha_t * ((1 - probs_t) ** gamma) * torch.log(probs_t + eps)
    return loss.mean()

def mse_loss(reconstructed_colors, colors):
    loss = torch.sum((reconstructed_colors - colors) ** 2, dim=-1)
    return loss.mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('train.py')
    parser.add_argument('--cloud_path', type=str, required=True, metavar='<path>',
                        help='the path to the original point cloud file')
    parser.add_argument('--geometry_path', default=None, type=str, metavar='<path>',
                        help='the path to the reconstructed geometry; if omitted, the ground-truth geometry will be applied')
    parser.add_argument('--exp_dir', type=str, required=True, metavar='<dir>',
                        help='the path to the directory containing all files needed for the experiment')
    parser.add_argument('--device', default=None, type=int, metavar='<int>',
                        help='the selected device')
    args = parser.parse_args()

    config_path = os.path.join(args.exp_dir, 'config.yaml')
    model_path = os.path.join(args.exp_dir, 'model.pt')
    log_path = os.path.join(args.exp_dir, 'log.txt')

    if not os.path.exists(args.cloud_path):
        raise FileNotFoundError
    if args.geometry_path is not None and not os.path.exists(args.geometry_path):
        raise FileNotFoundError
    if not os.path.exists(config_path):
        raise FileNotFoundError

    with open(config_path) as file:
        config_dict = yaml.safe_load(file)
        for name, value in config_dict.items():
            setattr(args, name, value)

    assert torch.cuda.is_available()
    device = torch.device('cuda')
    if args.device:
        torch.cuda.set_device(args.device)

    logger = Logger('train.py', log_path)
    logger.log('*' * 32 + ' train.py ' + '*' * 32)
    logger.log('Arguments:')
    for arg_str in vars(args):
        logger.log(f'    {arg_str}: {getattr(args, arg_str)}')

    if args.component == 'geometry':
        output_dim = 1
        sampler = GeometrySampler(args.cloud_path, args.depth, args.block_depth, args.batch_size, device, args.occupied_ratio)
        distortion_fn = lambda probs, occupancies: focal_loss(probs, occupancies, alpha=sampler.alpha)
    elif args.component == 'attribute':
        output_dim = 3
        sampler = AttributeSampler(args.cloud_path, args.geometry_path, args.depth, args.block_depth, args.batch_size, device)
        distortion_fn = mse_loss
    else:
        raise ValueError

    model = Representation(args.num_freqs, args.block_dim, args.hidden_dim, output_dim, args.num_blocks, args.layer_norm, args.short_cut).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_step_size = args.num_steps // args.scheduler_steps
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=0.1)

    penalty_list = []
    distortion_list = []
    loss_list = []

    model.train()
    for step in range(args.num_steps):
        samples, targets = sampler.sample()
        normalized_inputs = samples / (1 << (args.depth - 1)) - 1
        outputs = model(normalized_inputs)
        distortion = distortion_fn(outputs, targets)
        penalty = model.l1_penalty() / sampler.num_points
        loss = distortion + args.lmbda * penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        penalty_list.append(penalty.item())
        distortion_list.append(distortion.item())
        loss_list.append(loss.item())

        if (step + 1) % args.check_interval == 0:
            mean_penalty = sum(penalty_list) / len(penalty_list)
            mean_distortion = sum(distortion_list) / len(distortion_list)
            mean_loss = sum(loss_list) / len(loss_list)

            check = (step + 1) // args.check_interval
            num_checks = args.num_steps // args.check_interval
            logger.log(f'[Check {check}/{num_checks}] ' + 
                       f'lr: {scheduler.get_last_lr()[0]:.1} ' + 
                       f'loss: {mean_loss:.4} ' +
                       f'distortion: {mean_distortion:.4} ' +
                       f'penalty: {mean_penalty:.4}')

            penalty_list.clear()
            distortion_list.clear()
            loss_list.clear()

        scheduler.step()
    
    torch.save(model.state_dict(), model_path)