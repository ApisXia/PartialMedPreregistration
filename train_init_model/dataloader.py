from torch.utils.data import Dataset
import torch
import numpy as np
import traceback
import os


class CT3DRA_dataset(Dataset):
    def __init__(self, test_fold_idx, mode, data_path, split_file, batch_size, num_workers, res=128, eval_opt=False):
        self.path = data_path
        self.split = np.load(split_file)
        self.mode = mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.res = res
        self.ra_sample_num = 150
        self.eval_opt = eval_opt
        self.data_eval_num = 0

        self.expand_pixels = 0

        self.test_idx = test_fold_idx
        if type(self.test_idx) is int:
            self.test_idx = [self.test_idx]

        self.train_idx = [0, 1, 2, 3, 4]
        for item in self.test_idx:
            self.train_idx.remove(item)

        self.data = []
        if self.mode == 'train':
            for num in self.train_idx:
                self.data.append(self.split['split_{}'.format(num)])
        elif self.mode == 'test':
            for num in self.test_idx:
                self.data.append(self.split['split_{}'.format(num)])
        else:
            raise NameError('Wrong mode name!')
        self.data = np.concatenate(self.data)

        self.lower_bound = -500.
        self.upper_bound = 3000.
        assert self.lower_bound < self.upper_bound

    def __len__(self):
        if self.eval_opt:
            return self.ra_sample_num
        return len(self.data) * self.ra_sample_num

    def set_data_eval_num(self, num):
        self.data_eval_num = num

    def data_eval_name(self):
        return self.data[self.data_eval_num]

    def data_eval_len(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.eval_opt:
            try:
                instance_name = self.data[self.data_eval_num]
                matrix_path = self.path + '/{}/ct_3dra_scaled.npz'.format((instance_name))
                ra_sample_path = self.path + '/%s/sampled_3dra_%02d.npz' % (instance_name, idx)
                full_data = np.load(matrix_path)
                ct_matrix = full_data['images_ct']
                ct_align = full_data['matrix_ct_aligned']
                ra_matrix = full_data['images_3dra']
                ra_trans = full_data['matrix_3dra']

                ra_sample = np.load(ra_sample_path)
                ra_sample_matrix = ra_sample['images_3dra']
                ra_sample_loc = ra_sample['center_3dra']
                ra_sample_loc_origin = ra_sample['center_3dra_origin']

                return {'ct_matrix': ct_matrix,
                        'ct_align': ct_align,
                        'ra_matrix': ra_matrix,
                        'ra_trans': ra_trans,
                        'ra_sample_matrix': np.array(self.scale_input(ra_sample_matrix), dtype=np.float32),
                        'ra_sample_loc': ra_sample_loc,
                        'ra_sample_loc_origin': ra_sample_loc_origin}
            except:
                print('Error with {}: {}'.format(instance_name, traceback.format_exc()))
                raise

        real_idx = idx // self.ra_sample_num
        ra_sample_index = idx % self.ra_sample_num
        try:
            instance_name = self.data[real_idx]

            matrix_path = self.path + '/{}/ct_3dra_scaled.npz'.format((instance_name))

            ra_sample_path = self.path + '/%s/sampled_3dra_%02d.npz' % (instance_name, ra_sample_index)

            ct_matrix = np.load(matrix_path)['images_ct']
            ra_sample = np.load(ra_sample_path)
            ra_sample_matrix = ra_sample['images_3dra']
            ra_sample_loc = ra_sample['center_3dra']

        except:
            print('Error with {}: {}'.format(instance_name, traceback.format_exc()))
            raise

        return {'ct_matrix': np.array(self.scale_input(ct_matrix), dtype=np.float32),
                'ra_matrix': np.array(self.scale_input(ra_sample_matrix), dtype=np.float32),
                'ra_loc': np.squeeze(np.array((ra_sample_loc + self.expand_pixels) / (self.res + 2 * self.expand_pixels - 1), dtype=np.float32)),
                'instance_name': instance_name}

    def scale_input(self, matrix):
        matrix[matrix < self.lower_bound] = self.lower_bound
        matrix[matrix > self.upper_bound] = self.upper_bound
        return (matrix - self.lower_bound) / (self.upper_bound - self.lower_bound) - 0.5

    def get_loader(self, shuffle=True):

        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
            worker_init_fn=self.worker_init_fn)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)


if __name__ == "__main__":
    data_path = '/media/apis/WDSSD/Igarashi_Lab_Projs/Proj_MRA_Reg/Dataset/GpR_mhd_center_128'
    split_file = '/media/apis/WDSSD/Igarashi_Lab_Projs/Proj_MRA_Reg/Dataset/Split_folder/CT_3DRA_split.npz'
    data_loader = CT3DRA_dataset(0, 'train', data_path, split_file, 8, 12, eval_opt=True)
    temp = data_loader[0]
    c = 1
