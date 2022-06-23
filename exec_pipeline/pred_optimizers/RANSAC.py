import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.linalg import sqrtm
from post_DL_utils.rigid_matrix_from_predictions import gen_rigid_matrix


def matrix_RANSAC_optimizer(origin, pred, iters=3000, accept_P=0.999, rand_pick_num=6, error_threshold=10, minimum_final_pick_num=20):
    point_set_size = len(origin)
    best_matrix = None
    best_picked_num = rand_pick_num
    for it in range(iters):
        maybe_inliers = list(np.random.choice(point_set_size, rand_pick_num, replace=False))
        try:
            maybe_matrix = gen_rigid_matrix(origin[maybe_inliers, :],
                                            pred[maybe_inliers, :])
        except np.linalg.LinAlgError:
            it -= 1
            continue
        also_inliers = []
        for pt in range(point_set_size):
            if pt not in maybe_inliers \
                    and pt_estimate_error(maybe_matrix, origin[pt, :], pred[pt, :]) < error_threshold:
                also_inliers.append(pt)
        combine_possible_inliers = maybe_inliers + also_inliers
        # if len(combine_possible_inliers) == point_set_size:
        #     raise ValueError
        if len(combine_possible_inliers) >= max(minimum_final_pick_num, best_picked_num+1):
            # this_error = pt_estimate_error(better_matrix,
            #                                origin[combine_possible_inliers, :],
            #                                pred[combine_possible_inliers, :],
            #                                opt_mean=True)
            # better_matrix = gen_rigid_matrix(origin[combine_possible_inliers, :],
            #                                  pred[combine_possible_inliers, :])
            # best_matrix = better_matrix.copy()
            best_matrix = maybe_matrix.copy()
            best_picked_num = len(combine_possible_inliers)
    if best_matrix is None:
        raise ValueError
    # recommend_iters = math.log(1 - accept_P) / math.log(1 - pow(best_picked_num / point_set_size, rand_pick_num))
    return best_matrix


def matrix_RANSAC_filter(origin, pred, iters=3000, accept_P=0.999, rand_pick_num=6, error_threshold=10, minimum_final_pick_num=20):
    point_set_size = len(origin)
    best_matrix = None
    best_picked_num = rand_pick_num
    for it in range(iters):
        maybe_inliers = list(np.random.choice(point_set_size, rand_pick_num, replace=False))
        try:
            maybe_matrix = gen_rigid_matrix(origin[maybe_inliers, :],
                                            pred[maybe_inliers, :])
        except np.linalg.LinAlgError:
            it -= 1
            continue
        also_inliers = []
        for pt in range(point_set_size):
            if pt not in maybe_inliers \
                    and pt_estimate_error(maybe_matrix, origin[pt, :], pred[pt, :]) < error_threshold:
                also_inliers.append(pt)
        combine_possible_inliers = maybe_inliers + also_inliers
        # if len(combine_possible_inliers) == point_set_size:
        #     raise ValueError
        if len(combine_possible_inliers) >= max(minimum_final_pick_num, best_picked_num+1):
            # this_error = pt_estimate_error(better_matrix,
            #                                origin[combine_possible_inliers, :],
            #                                pred[combine_possible_inliers, :],
            #                                opt_mean=True)
            # better_matrix = gen_rigid_matrix(origin[combine_possible_inliers, :],
            #                                  pred[combine_possible_inliers, :])
            # best_matrix = better_matrix.copy()
            best_matrix = maybe_matrix.copy()
            best_picked_num = len(combine_possible_inliers)
            best_pts = np.asarray(combine_possible_inliers).copy()
    if best_matrix is None:
        raise ValueError
    # recommend_iters = math.log(1 - accept_P) / math.log(1 - pow(best_picked_num / point_set_size, rand_pick_num))
    return best_pts


def pt_estimate_error(trans_matrix, origin, pred, opt_mean=False, norm_ord=2):
    if len(origin.shape) < 2:
        origin = np.expand_dims(origin, 0)
        pred = np.expand_dims(pred, 0)
    addon_ones = np.ones((origin.shape[0], 1))
    origin = np.concatenate((origin, addon_ones), axis=1)
    trans_origin = np.matmul(trans_matrix, np.transpose(origin, axes=[1, 0]))
    pt_estimate_error = np.linalg.norm(trans_origin[0:3, :] - np.transpose(pred, axes=[1, 0]), axis=0, ord=norm_ord)
    if opt_mean:
        return np.mean(pt_estimate_error)
    return pt_estimate_error
