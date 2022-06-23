import numpy as np
from scipy.ndimage import interpolation


def resample(image, old_spacing, new_spacing=[1, 1, 1]):
    resize_factor = old_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = old_spacing / real_resize_factor

    image = interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


def resample_based_scale(image, old_spacing, lim_size, base_scale):
    resize_factor = old_spacing * base_scale
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    image = interpolation.zoom(image, real_resize_factor, mode='nearest')

    im_x, im_y, im_z = image.shape
    fill_x = min(im_x, lim_size)
    fill_y = min(im_y, lim_size)
    filled_image = np.ones([lim_size] * 3) * -2048.
    if im_z <= lim_size:
        filled_image[:fill_x, :fill_y, (lim_size-im_z):] = image[:fill_x, :fill_y, :]
    else:
        filled_image[:fill_x, :fill_y, :] = image[:fill_x, :fill_y, (im_z-lim_size):]
    shift_z = lim_size - im_z

    return filled_image, shift_z

