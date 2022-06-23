from glob import glob
from basic_utils.show_data import load_dcm
import numpy as np
import pandas as pd

if __name__ == '__main__':
    gt_path = '/home2/reg/dataset/GpR/*/CT/'
    f_list = list(glob(gt_path))
    
    actual_x_holder = []
    actual_y_holder = []
    actual_z_holder = []
    pixel_x_holder = []
    pixel_y_holder = []
    pixel_z_holder = []
    spacing_x_holder = []
    spacing_y_holder = []
    spacing_z_holder = []
    name_holder = []
    for f in f_list:
        print('Finished {}'.format(f))
        f_name = f.split('/')[5]
        pixels, _, _, spacings = load_dcm(f)
        actual_size = (np.asarray(pixels.shape) - 1) * spacings
        name_holder.append(f_name)
        actual_x_holder.append(actual_size[0])
        actual_y_holder.append(actual_size[1])
        actual_z_holder.append(actual_size[2])
        pixel_x_holder.append(pixels.shape[0])
        pixel_y_holder.append(pixels.shape[1])
        pixel_z_holder.append(pixels.shape[2])
        spacing_x_holder.append(spacings[0])
        spacing_y_holder.append(spacings[1])
        spacing_z_holder.append(spacings[2])

    data_save = {'Data ID': name_holder,
                 'Actual_X(mm)':  actual_x_holder,
                 'Actual_Y(mm)':  actual_y_holder,
                 'Actual_Z(mm)':  actual_z_holder,
                 'Pixel_X(px)':   pixel_x_holder,
                 'Pixel_Y(px)':   pixel_y_holder,
                 'Pixel_Z(px)':   pixel_z_holder,
                 'Spacing_X(mm)': spacing_x_holder,
                 'Spacing_Y(mm)': spacing_y_holder,
                 'Spacing_Z(mm)': spacing_z_holder}
    df = pd.DataFrame(data_save)
    df.to_excel('CT_stats_more.xlsx')
