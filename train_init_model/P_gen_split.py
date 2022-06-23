from glob import glob
import numpy as np


def gen_5splits(data_path, out_path):
    data_list = list(glob(data_path + '/*'))
    data_list = [path.split('/')[-1] for path in data_list]
    size_each = len(data_list) // 5
    split_0 = data_list[0: size_each]
    split_1 = data_list[size_each: size_each * 2]
    split_2 = data_list[size_each * 2: size_each * 3]
    split_3 = data_list[size_each * 3: size_each * 4]
    split_4 = data_list[size_each * 4:]
    np.savez(out_path, split_0=split_0, split_1=split_1, split_2=split_2, split_3=split_3, split_4=split_4)
    c = 1


if __name__ == '__main__':
    data_path = '/media/apis/WDSSD/Igarashi_Lab_Projs/Proj_MRA_Reg/Dataset/GpR_mhd_center_128/'
    out_path = '/media/apis/WDSSD/Igarashi_Lab_Projs/Proj_MRA_Reg/Dataset/Split_folder/CT_3DRA_split.npz'
    gen_5splits(data_path, out_path)
