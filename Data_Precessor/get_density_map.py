import numpy as np
import scipy
from scipy.ndimage import filters




def get_density_map(im, points, model='kdtree',neighbour_size=10):
    im_density = np.zeros_like(im, dtype=np.float32)
    [h, w] = im.shape  # ?????????
    gt_count = points.shape[0]
    k_tree_size = min(gt_count, neighbour_size)

    if gt_count == 0:
        return im_density

    if model == 'constant_sigma':
        sigma = 4.0
        if gt_count == 1:
            x1 = max(1, min(w, round(points[0, 0])))-1
            y1 = max(1, min(h, round(points[0, 1])))-1
            im_density[y1, x1] = 1.0
            # im_density[y1, x1] = 255
            return
        #generate density..
        for pt in points:
            ptx = int(max(1, min(w, round(pt[0])))-1)
            pty = int(max(1, min(h, round(pt[1])))-1)
            pt2d = np.zeros_like(im_density, dtype=np.float32)
            pt2d[pty, ptx] = 1.0
            im_density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    else:
        #build kdtree
        leafsize = 4086
        tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
        distances, locations = tree.query(points, k=k_tree_size, eps=10.)

        #generate density
        for i, pt in enumerate(points):
            ptx = int(max(0, min(w - 1, round(pt[0]))))
            pty = int(max(0, min(h - 1, round(pt[1]))))
            pt2d = np.zeros_like(im_density, dtype=np.float32)
            pt2d[pty, ptx] = 1.0
            if gt_count > 1:
                sigma = 0.1*np.mean(distances[i, 1:k_tree_size])
            else:
                sigma = np.average(np.array(im.shape))/2.0/2.0
            im_density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')

    density_count = np.sum(im_density)
    print('density count in original ground true {}; in density map {}'.format(gt_count, density_count))

    return im_density






