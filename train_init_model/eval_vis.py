from glob import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data_path = '/media/apis/WDSSD/Igarashi_Lab_Projs/Proj_MRA_Reg/preprocess/experiments_group/simplest_model/eval_result/p16_fold{}/diffs_statistics_5s.npz'
tra_holder = []
rot_holder = []
for i in range(5):
    with np.load(data_path.format(i)) as np_loaded:
        tra_holder.append(np_loaded['translation_diff'])
        rot_holder.append(np_loaded['rotation_diff'])
pt1 = sns.boxplot(data=tra_holder, palette="Set3")
pt1.set(xlabel="Folders", ylabel="Translation (px)")
plt.show()
print(np.mean(np.concatenate(tra_holder)))
print(np.median(np.concatenate(tra_holder)))
pt2 = sns.boxplot(data=rot_holder, palette="Set3")
pt2.set(xlabel="Folds", ylabel="Rotation (Angle)")
plt.show()
print(np.mean(np.concatenate(rot_holder)))
print(np.median(np.concatenate(rot_holder)))
