import torch
import os
from glob import glob
import numpy as np
from torch.nn import functional as F
import time
import collections
from tqdm import tqdm


class Evaluator(object):
    def __init__(self, model, dataset, exp_name, checkpoint=None, device=torch.device("cuda")):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.checkpoint_path = os.path.dirname(__file__) + '/experiments/{}/checkpoints/'.format(exp_name)
        self.load_checkpoint(checkpoint)
        self.dataset = dataset

    def load_checkpoint(self, checkpoint):
        checkpoints = glob(self.checkpoint_path + '/*')
        if checkpoint is None:
            if len(checkpoints) == 0:
                print('No checkpoints found at {}'.format(self.checkpoint_path))
                return 0, 0

            checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in checkpoints]
            checkpoints = np.array(checkpoints, dtype=int)
            checkpoints = np.sort(checkpoints)
            path = self.checkpoint_path + 'checkpoint_{}.tar'.format(checkpoints[-1])
        else:
            path = self.checkpoint_path + 'checkpoint_{}.tar'.format(checkpoint)
        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)

        def remove_module(key):
            k_split = key.split('.')[1:]
            short_key = '.'.join(k_split)
            return short_key

        # modify_check = collections.OrderedDict(
        #     [(remove_module(k), v) for k, v in checkpoint['model_state_dict'].items()])

        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.model.load_state_dict(modify_check)
        epoch = checkpoint['epoch']
        return epoch

    def eval_registration(self, data_num):
        self.dataset.set_data_eval_num(data_num)
        data_loader = self.dataset.get_loader()

        pred_holder = []
        gt_holder = []
        origin_holder = []
        with tqdm(data_loader, smoothing=0.2) as desc:
            for batch in desc:
                ra_m = batch.get('ra_matrix')
                ct_m = batch.get('ct_matrix')
                ra_trans = batch.get('ra_trans')
                ct_align = batch.get('ct_align')

                ra_sample_m = batch.get('ra_sample_matrix').to(self.device)
                ra_sample_loc = batch.get('ra_sample_loc')
                ra_sample_loc_origin = batch.get('ra_sample_loc_origin')
                loc_pred = self.model(ra_sample_m)

                pred_holder.append(loc_pred.detach().cpu().numpy())
                gt_holder.append(ra_sample_loc.detach().cpu().numpy())
                origin_holder.append(ra_sample_loc_origin.detach().cpu().numpy())

        pred_holder = np.concatenate(pred_holder, axis=0) * 128
        gt_holder = np.concatenate(gt_holder, axis=0)
        origin_holder = np.concatenate(origin_holder, axis=0)
        name_back = self.dataset.data_eval_name()
        return pred_holder, gt_holder, origin_holder, name_back, ra_m[0, :, :], ct_m[0, :, :], ra_trans[0, :, :], ct_align[0, :, :]


def l2_norm(x, y):
    """Calculate l2 norm (distance) of `x` and `y`.
    Args:
        x (numpy.ndarray or cupy): (batch_size, num_point, coord_dim)
        y (numpy.ndarray): (batch_size, num_point, coord_dim)
    Returns (numpy.ndarray): (batch_size, num_point,)
    """
    return ((x - y) ** 2).sum(axis=1)


def farthest_point_sampling(pts, k, initial_idx=None, metrics=l2_norm,
                            skip_initial=False, indices_dtype=np.int32):
    """Batch operation of farthest point sampling
    Code referenced from below link by @Graipher
    https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
    Args:
        pts (numpy.ndarray or cupy.ndarray): 2-dim array (num_point, coord_dim)
            or 3-dim array (batch_size, num_point, coord_dim)
            When input is 2-dim array, it is treated as 3-dim array with
            `batch_size=1`.
        k (int): number of points to sample
        initial_idx (int): initial index to start farthest point sampling.
            `None` indicates to sample from random index,
            in this case the returned value is not deterministic.
        metrics (callable): metrics function, indicates how to calc distance.
        skip_initial (bool): If True, initial point is skipped to store as
            farthest point. It stabilizes the function output.
        xp (numpy or cupy):
        indices_dtype (): dtype of output `indices`
        distances_dtype (): dtype of output `distances`
    Returns (tuple): `indices` and `distances`.
        indices (numpy.ndarray or cupy.ndarray): 2-dim array (batch_size, k, )
            indices of sampled farthest points.
            `pts[indices[i, j]]` represents `i-th` batch element of `j-th`
            farthest point.
        distances (numpy.ndarray or cupy.ndarray): 3-dim array
            (batch_size, k, num_point)
    """
    num_point, coord_dim = pts.shape
    indices = np.zeros((k, ), dtype=indices_dtype)

    # distances[i, j] is distance between i-th farthest point `pts[i]`
    # and j-th input point `pts[j]`.
    if initial_idx is None:
        indices[0] = np.random.randint(len(pts))
    else:
        indices[0] = initial_idx

    farthest_point = pts[indices[0], :]
    # minimum distances to the sampled farthest point
    try:
        min_distances = metrics(farthest_point, pts)
    except Exception as e:
        import IPython; IPython.embed()

    if skip_initial:
        # Override 0-th `indices` by the farthest point of `initial_idx`
        indices[0] = np.argmax(min_distances)
        farthest_point = pts[indices[0], :]
        min_distances = metrics(farthest_point, pts)

    for i in range(1, k):
        indices[i] = np.argmax(min_distances)
        farthest_point = pts[indices[i], :]
        dist = metrics(farthest_point, pts)
        min_distances = np.minimum(min_distances, dist)
    return indices



