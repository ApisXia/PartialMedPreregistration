import model as model
from dataloader import CT3DRA_dataset
import train_utils
import torch


net = model.Simple_ConvMODEL()

data_path = '/home2/reg/dataset/GpR_mhd_center_aligned_128_16_new'
split_file = '/home2/reg/dataset/CT_3DRA_split_no46.npz'
train_dataset = CT3DRA_dataset(1, 'train', data_path, split_file, 64, 8)
test_dataset = CT3DRA_dataset(1, 'test', data_path, split_file, 64, 8)

exp_name = 'Simple_ConvMODEL_LScale_p16_f1_new'

trainer = train_utils.Trainer(net,
                              torch.device("cuda"),
                              train_dataset,
                              test_dataset,
                              exp_name)
 
trainer.train_model(3000)
