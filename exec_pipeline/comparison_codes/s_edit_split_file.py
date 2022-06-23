import numpy as np

data_path = '/home2/reg/dataset/CT_3DRA_split.npz'
file_in = np.load(data_path)
s_0 = file_in['split_0']
s_1 = file_in['split_1']
s_2 = file_in['split_2']
s_3 = file_in['split_3']
s_4 = file_in['split_4']
s_2_new = np.delete(s_2, 4)
new_dict = {'split_0': s_0,
            'split_1': s_1,
            'split_2': s_2_new,
            'split_3': s_3,
            'split_4': s_4}
np.savez('/home2/reg/dataset/CT_3DRA_split_no46.npz', **new_dict)
c = 1