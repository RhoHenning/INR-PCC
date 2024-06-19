import os
import csv
import argparse
import yaml
import deepCABAC
import torch
import numpy as np
import cloud
import pc_error
from logger import Logger
from model import Representation

result_header = [
    'cloud',
    'quantization steps',
    'threshold',
    'scaling ratio',
    'bits',
    'bits per point',
    'D1 PSNR',
    'D2 PSNR',
    'Y PSNR',
    'YUV PSNR'
]

def encode_model(model, step_size):
    encoder = deepCABAC.Encoder()
    for param in model.state_dict().values():
        param = param.cpu().numpy()
        encoder.encodeWeightsRD(param, 0.0, step_size, 0.0)
    stream = encoder.finish().tobytes()

    decoder = deepCABAC.Decoder()
    decoder.getStream(np.frombuffer(stream, dtype=np.uint8))
    state_dict = model.state_dict()
    for name in state_dict.keys():
        param = decoder.decodeWeights()
        state_dict[name] = torch.tensor(param)
    decoder.finish()
    model.load_state_dict(state_dict)
    return stream, model

def local_voxels(block_width):
    base = torch.arange(block_width, dtype=torch.float32, device=device).unsqueeze(1)
    i = base.repeat_interleave(block_width ** 2, dim=0)
    j = base.repeat_interleave(block_width, dim=0).repeat(block_width, 1)
    k = base.repeat(block_width ** 2, 1)
    voxels = torch.cat([i, j, k], dim=-1)
    return voxels

@torch.no_grad()
def reconstruct_geometry(model, blocks):
    model.eval()
    reconstructed_points = []
    for block in blocks:
        voxels = local_voxels(block_width) + block * block_width
        num_iters = (len(voxels) + args.batch_size - 1) // args.batch_size
        for iter in range(num_iters):
            start = iter * args.batch_size
            end = min(start + args.batch_size, len(voxels))
            inputs = voxels[start:end]
            normalized_inputs = inputs / (1 << (args.depth - 1)) - 1
            outputs = model(normalized_inputs)
            mask = outputs.squeeze(dim=-1) > args.threshold
            reconstructed_points.append(inputs[mask])
    reconstructed_points = torch.cat(reconstructed_points, dim=0)
    return reconstructed_points

@torch.no_grad()
def reconstruct_attribute(model, voxels):
    model.eval()
    reconstructed_colors = []
    num_iters = (len(voxels) + args.batch_size - 1) // args.batch_size
    for iter in range(num_iters):
        start = iter * args.batch_size
        end = min(start + args.batch_size, len(voxels))
        inputs = voxels[start:end]
        normalized_inputs = inputs / (1 << (args.depth - 1)) - 1
        outputs = model(normalized_inputs)
        reconstructed_colors.append(outputs)
    reconstructed_colors = torch.cat(reconstructed_colors, dim=0)
    return reconstructed_colors

if __name__ == '__main__':
    parser = argparse.ArgumentParser('test.py')
    parser.add_argument('--cloud_path', type=str)
    parser.add_argument('--geometry_path', default=None, type=str)
    parser.add_argument('--exp_dir', type=str)
    parser.add_argument('--device', default=None, type=int)
    args = parser.parse_args()

    config_path = os.path.join(args.exp_dir, 'config.yaml')
    model_path = os.path.join(args.exp_dir, 'model.pt')
    log_path = os.path.join(args.exp_dir, 'log.txt')
    reconstructed_path = os.path.join(args.exp_dir, 'cloud.ply')
    result_path = os.path.join(args.exp_dir, 'result.csv')

    if not os.path.exists(args.cloud_path):
        raise FileNotFoundError
    if args.geometry_path is not None and not os.path.exists(args.geometry_path):
        raise FileNotFoundError
    if not os.path.exists(config_path):
        raise FileNotFoundError
    if not os.path.exists(model_path):
        raise FileNotFoundError

    config_dict = yaml.safe_load(open(config_path))
    for name, value in config_dict.items():
        setattr(args, name, value)

    assert torch.cuda.is_available()
    device = torch.device('cuda')
    if args.device:
        torch.cuda.set_device(args.device)

    if not os.path.exists(result_path):
        with open(result_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(result_header)

    logger = Logger('test.py', log_path)
    logger.log('*' * 32 + ' test.py ' + '*' * 32)
    logger.log(str(args))
    
    results = {}
    results['cloud'] = args.cloud_path
    results['quantization steps'] = args.quantization_steps

    if args.component == 'geometry':
        output_dim = 1
    elif args.component == 'attribute':
        output_dim = 3
    else:
        raise ValueError

    block_width = 1 << (args.depth - args.block_depth)
    points, _, _ = cloud.load_ply_cloud(args.cloud_path)
    blocks = cloud.partition_blocks(points, args.depth, args.block_depth)
    blocks = torch.from_numpy(blocks).to(device)

    model = model = Representation(args.num_freqs, args.block_dim, args.hidden_dim, output_dim, args.num_blocks, args.layer_norm, args.short_cut).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    step_size = 1 / args.quantization_steps
    stream, model = encode_model(model, step_size)
    num_bits = len(stream) * 8
    if args.component == 'geometry':
        num_bits += args.block_depth * 3 * (len(blocks) + 1) + 32
    logger.log(f'Compressed size: {num_bits} bits ({num_bits / len(points):.6} bpp)')
    results['bits'] = num_bits
    results['bits per point'] = num_bits / len(points)

    if args.component == 'geometry':
        reconstructed_points = reconstruct_geometry(model, blocks)
        reconstructed_colors = None
        assert len(reconstructed_points) > 0
        logger.log(f'Original points: {len(points)}')
        logger.log(f'Reconstructed points: {len(reconstructed_points)}')
        logger.log(f'Scaling ratio: {len(reconstructed_points) / len(points):.6}')
        results['threshold'] = args.threshold
        results['scaling ratio'] = len(reconstructed_points) / len(points)
    elif args.component == 'attribute':
        if args.geometry_path is None:
            voxels = points
        else:
            voxels, _, _ = cloud.load_ply_cloud(args.geometry_path)
        reconstructed_points = torch.from_numpy(voxels).to(device)
        reconstructed_colors = reconstruct_attribute(model, reconstructed_points)
    reconstructed_points = reconstructed_points.cpu().numpy()
    if reconstructed_colors is not None:
        reconstructed_colors = reconstructed_colors.cpu().numpy()
    cloud.write_ply_cloud(reconstructed_path, reconstructed_points, reconstructed_colors)

    has_colors = args.component == 'attribute'
    distortion_results = pc_error.distortion(args.cloud_path, reconstructed_path, (1 << args.depth) - 1, has_colors)
    for metric in ['D1', 'D2', 'Y', 'YUV']:
        metric_psnr = f'{metric} PSNR'
        if metric in distortion_results and metric_psnr in distortion_results:
            logger.log(f'{metric}: {distortion_results[metric]:.6} PSNR: {distortion_results[metric_psnr]:.6}')
            if metric_psnr in result_header:
                results[metric_psnr] = f'{distortion_results[metric_psnr]:.6}'

    with open(result_path, 'a') as file:
        writer = csv.DictWriter(file, fieldnames=result_header)
        writer.writerow(results)