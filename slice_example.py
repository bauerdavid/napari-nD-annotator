import numpy as np
from scipy.spatial.transform.rotation import Rotation
from scipy.interpolate import griddata
from skimage.data import cells3d
import matplotlib.pyplot as plt
import itertools

image_map = cells3d()

image_map = image_map[:, 1, :, :]
grid = np.mgrid[:image_map.shape[0], :image_map.shape[1], :image_map.shape[2]].T.reshape(-1, 3)
image_map = image_map.T.reshape(-1)
coords = np.meshgrid(np.arange(50), np.arange(50), 0, indexing='xy')

rotation = Rotation.from_euler('xyz', [30, 12, 20], degrees=True)
concatenated = np.concatenate([c[np.newaxis] for c in coords]).T.reshape(-1, 3)
transformed = rotation.apply(concatenated)
transformed[:, 1] += 50
transformed[:, 2] += 50
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], marker="x")
# plt.show()
lower_bound = np.floor(transformed.min(0)).astype(int)
upper_bound = np.ceil(transformed.max(0)).astype(int)
mask = np.all(np.logical_and(grid >= lower_bound, grid <= upper_bound), 1)

import time
print("cropping only")
start = time.time()
crop = griddata(grid[mask], image_map[mask], transformed, method='linear', fill_value=0)
end = time.time()
print("done in %d seconds" % (end-start))
# plt.figure()
# plt.imshow(crop.reshape(50, 50))
# plt.show()

print("selecting neighboring pixels")
start = time.time()
prod = np.asarray(list(itertools.product([False, True], repeat=3)))
filtered_grid = grid[mask]
filtered_image_map = image_map[mask]
# corner_1 = np.floor(transformed)
# corner_2 = np.ceil(transformed)
# neighbor_points = np.where(np.tile(prod, (len(corner_1), 1)), np.repeat(corner_1, 8, 0), np.repeat(corner_2, 8, 0)).astype(np.int32)
# neighbor_points = np.unique(neighbor_points, axis=0)
# ax.scatter(neighbor_points[:, 0], neighbor_points[:, 1], neighbor_points[:, 2], marker="x", color="r")
# mask = np.all(np.isin(grid, neighbor_points), 1)
idx = []
for p in list(transformed):
    idx.extend(np.argwhere((np.abs(filtered_grid-p) <=1).all(1)).reshape(-1))
# idx = np.apply_along_axis(lambda p: np.argwhere((np.abs(grid-p) <=1).all(1)).reshape(-1), 1, transformed)


crop = griddata(filtered_grid[idx], filtered_image_map[idx], transformed, method='linear', fill_value=0)
end = time.time()

print("done in %d seconds" % (end-start))
# ax.scatter(grid[idx][:, 0], grid[idx][:, 1], grid[idx][:, 2], marker="x", color="r")
# plt.show()
plt.figure()
plt.imshow(crop.reshape(50, 50))
plt.show()
exit()
# map = tifffile.imread(cells_path, aszarr=True)
# map = zarr.open(map, mode="r")
# t, h, w, c = map.shape
# print(map.shape)