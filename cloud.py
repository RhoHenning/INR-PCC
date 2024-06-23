import numpy as np
import open3d as o3d

def load_ply_cloud(path):
    pcd = o3d.io.read_point_cloud(path)
    points = np.array(pcd.points).astype(np.float32)
    colors, normals = None, None
    if pcd.has_colors():
        colors = np.array(pcd.colors).astype(np.float32)
    if pcd.has_normals():
        normals = np.array(pcd.normals).astype(np.float32)
    return points, colors, normals

def write_ply_cloud(path, points, colors=None, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(path, pcd, write_ascii=True)
    with open(path, 'r') as file:
        cloud = file.read()
    with open(path, 'w') as file:
        file.write(cloud.replace('double', 'float'))
    del cloud

def partition_blocks(points, depth, block_depth):
    block_occupancy = np.zeros((1 << block_depth,) * 3)
    block_width = 1 << (depth - block_depth)
    for point in points:
        block = divmod(point.astype(np.int64), block_width)[0]
        block_occupancy[tuple(block)] = 1
    blocks = np.argwhere(block_occupancy)
    return blocks

def estimate_normals(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()
    normals = np.array(pcd.normals).astype(np.float32)
    return normals

def nearest_neighbor_indices(points, voxels):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    kd_tree = o3d.geometry.KDTreeFlann(pcd)
    indices = np.zeros((len(voxels),), dtype=np.int64)
    for i, voxel in enumerate(voxels):
        indices[i] = kd_tree.search_knn_vector_3d(voxel, 1)[1][0]
    return indices