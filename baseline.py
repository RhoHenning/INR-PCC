import argparse
import os
import csv
import cloud
import pc_error
import tmc3
from logger import Logger

result_header = [
    'cloud',
    'pqs (octree)',
    'qp (RAHT)',
    'bits',
    'bits per point',
    'D1 PSNR',
    'D2 PSNR',
    'Y PSNR',
    'YUV PSNR'
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser('baseline.py')
    parser.add_argument('--cloud_path', type=str, required=True, metavar='<path>',
                        help='the path to the original point cloud file')
    parser.add_argument('--result_dir', type=str, required=True, metavar='<dir>',
                        help='the path to the directory containing all output files')
    parser.add_argument('--depth', default=10, type=int, metavar='<int>',
                        help='voxelization depth of the original point cloud [default: 10]')
    parser.add_argument('--encode_colors', action='store_true')
    parser.add_argument('--pqs', default=0.75, type=float, metavar='<float>',
                        help='position quantization scale used by G-PCC (octree) [default: 0.75]')
    parser.add_argument('--qp', default=34, type=int, metavar='<int>',
                        help='quantization parameter used by G-PCC (RAHT) [default: 34]')
    args = parser.parse_args()

    bin_path = os.path.join(args.result_dir, 'encoded.bin')
    reconstructed_path = os.path.join(args.result_dir, 'cloud.ply')
    result_path = os.path.join(args.result_dir, 'result.csv')

    if not os.path.exists(args.cloud_path):
        raise FileNotFoundError
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(result_path):
        with open(result_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(result_header)

    logger = Logger('baseline.py')
    logger.log('*' * 32 + ' baseline.py ' + '*' * 32)
    logger.log('Arguments:')
    for arg_str in vars(args):
        logger.log(f'    {arg_str}: {getattr(args, arg_str)}')

    results = {}
    results['cloud'] = args.cloud_path
    results['pqs (octree)'] = args.pqs
    if args.encode_colors:
        results['qp (RAHT)'] = args.qp

    points, _, _ = cloud.load_ply_cloud(args.cloud_path)
    encode_results = tmc3.encode(args.cloud_path, bin_path, args.pqs, args.qp, args.encode_colors)
    num_bits = encode_results['total'] * 8
    logger.log(f'Compressed size: {num_bits} bits ({num_bits / len(points):.6} bpp)')
    results['bits'] = num_bits
    results['bits per point'] = f'{num_bits / len(points):.6}'
    tmc3.decode(bin_path, reconstructed_path)

    distortion_results = pc_error.distortion(args.cloud_path, reconstructed_path, (1 << args.depth) - 1, args.encode_colors)
    for metric in ['D1', 'D2', 'Y', 'YUV']:
        metric_psnr = f'{metric} PSNR'
        if metric in distortion_results and metric_psnr in distortion_results:
            logger.log(f'{metric}: {distortion_results[metric]:.6} PSNR: {distortion_results[metric_psnr]:.6}')
            if metric_psnr in result_header:
                results[metric_psnr] = f'{distortion_results[metric_psnr]:.6}'

    with open(result_path, 'a') as file:
        writer = csv.DictWriter(file, fieldnames=result_header)
        writer.writerow(results)
    # os.remove(bin_path)
    # os.remove(reconstructed_path)