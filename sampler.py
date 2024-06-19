import torch
import cloud

class Bitset:
    def __init__(self, bit_size, device):
        self.bit_size = bit_size
        self.int8_size = bit_size // 8 + 1
        self.bits = torch.zeros((self.int8_size,), dtype=torch.int8, device=device)

    def add(self, indices):
        if (indices >= self.bit_size).any():
            raise IndexError
        mask = torch.div(indices, 8, rounding_mode='floor')
        bit_pos = (indices - mask * 8).to(torch.int8)
        self.bits.index_add_(0, mask, torch.ones_like(bit_pos) << bit_pos)

    def test(self, indices):
        if (indices >= self.bit_size).any():
            raise IndexError
        mask = torch.div(indices, 8, rounding_mode='floor')
        bit_pos = indices - mask * 8
        results = (self.bits[mask] >> bit_pos) & 1
        return results

class GeometrySampler:
    def __init__(self, cloud_path, depth, block_depth, batch_size, device, occupied_ratio=None):
        self.device = device
        self.depth = depth
        self.block_depth = block_depth
        self.block_width = 1 << (depth - block_depth)
        points, _, _ = cloud.load_ply_cloud(cloud_path)
        blocks = cloud.partition_blocks(points, depth, block_depth)
        self.points = torch.from_numpy(points).to(device)
        self.blocks = torch.from_numpy(blocks).to(device)
        self.num_points = len(self.points)
        self.num_blocks = len(self.blocks)
        self.num_voxels = len(self.blocks) * (self.block_width ** 3)

        self.batch_size = batch_size
        self.occupied_ratio = occupied_ratio
        occupied_fraction = self.num_points / self.num_voxels
        if occupied_ratio:
            point_fraction = (occupied_ratio - occupied_fraction) / (1 - occupied_fraction)
            point_batch_size = int(batch_size * point_fraction)
            voxel_batch_size = batch_size - point_batch_size
            if point_batch_size <= 0 or voxel_batch_size <= 0:
                raise ValueError
            self.point_batch_size = point_batch_size
            self.voxel_batch_size = voxel_batch_size
            self.alpha = 1 - occupied_ratio
        else:
            self.alpha = 1 - occupied_fraction
        
        self.bitset = Bitset(self.num_voxels, device)
        num_iters = (self.num_points + batch_size - 1) // batch_size
        for iter in range(num_iters):
            start = iter * batch_size
            end = min(start + batch_size, self.num_points)
            self.bitset.add(self.flatten_voxels(self.points[start:end]))

    def flatten1(self, local_voxels):
        return (local_voxels[:, 0] * self.block_width + local_voxels[:, 1]) * self.block_width + local_voxels[:, 2]

    def flatten2(self, local_voxel_indices, block_indices):
        return block_indices * (self.block_width ** 3) + local_voxel_indices

    def flatten_voxels(self, voxels):
        voxel_blocks = voxels.to(torch.int64) // self.block_width
        dist = torch.sum((self.blocks.repeat(len(voxel_blocks), 1, 1) - voxel_blocks.unsqueeze(1)) ** 2, dim=-1)
        pair = torch.nonzero(dist == 0)
        block_indices = torch.zeros_like(voxel_blocks[:, 0]).to(torch.int64)
        block_indices[pair[:, 0]] = pair[:, 1]
        local_voxels = voxels.to(torch.int64) % self.block_width
        local_voxel_indices = self.flatten1(local_voxels)
        flatten = self.flatten2(local_voxel_indices, block_indices)
        return flatten
    
    def sample(self):
        if self.occupied_ratio:
            point_indices = torch.randint(0, self.num_points, (self.point_batch_size,), device=self.device)
            points = self.points[point_indices]
            flatten_point = self.flatten_voxels(points)
            block_indices = torch.randint(0, self.num_blocks, (self.voxel_batch_size,), device=self.device)
            blocks = self.blocks[block_indices]
            local_voxels = torch.randint(0, self.block_width, (self.voxel_batch_size, 3), device=self.device)
            local_voxel_indices = self.flatten1(local_voxels)
            voxels = local_voxels + self.block_width * blocks
            flatten_voxel = self.flatten2(local_voxel_indices, block_indices)
            samples = torch.cat((points, voxels), dim = 0)
            flatten = torch.cat((flatten_point, flatten_voxel), dim = 0)
            indices = torch.randperm(self.batch_size, device=self.device)
            samples = samples[indices]
            flatten = flatten[indices]
        else:
            block_indices = torch.randint(0, self.num_blocks, (self.batch_size,), device=self.device)
            blocks = self.blocks[block_indices]
            local_voxels = torch.randint(0, self.block_width, (self.batch_size, 3), device=self.device)
            local_voxel_indices = self.flatten1(local_voxels)
            samples = local_voxels + self.block_width * blocks
            flatten = self.flatten2(local_voxel_indices, block_indices)

        occupancies = self.bitset.test(flatten)
        return samples, occupancies

class AttributeSampler:
    def __init__(self, cloud_path, geometry_path, depth, block_depth, batch_size, device):
        self.device = device
        self.depth = depth
        self.block_depth = block_depth
        self.block_width = 1 << (depth - block_depth)
        points, colors, _ = cloud.load_ply_cloud(cloud_path)
        self.num_points = len(points)

        self.batch_size = batch_size
        
        if geometry_path is None:
            voxels = points
            target_colors = colors
        else:
            voxels, _, _ = cloud.load_ply_cloud(geometry_path)
            indices = cloud.nearest_neighbor_indices(points, voxels)
            target_colors = colors[indices]

        self.voxels = torch.from_numpy(voxels).to(device)
        self.target_colors = torch.from_numpy(target_colors).to(device)
        self.num_voxels = len(voxels)
    
    def sample(self):
        indices = torch.randint(0, self.num_voxels, (self.batch_size,))
        samples = self.voxels[indices]
        target_colors = self.target_colors[indices]
        return samples, target_colors