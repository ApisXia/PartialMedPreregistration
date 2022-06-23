import seaborn as sns
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_format = 'pt_preds_stats/*.xlsx'
data_list = list(glob(data_format))
no_R_holder = []
R_holder = []
rm_R_holder = []
for case in data_list:
    dp_pred_ori = pd.read_excel(case, sheet_name='pred_gt')
    R_id_load = pd.read_excel(case, sheet_name='R ID')
    loss_array = np.sqrt((dp_pred_ori['pred_x'].to_numpy() - dp_pred_ori['gt_x'].to_numpy())**2 +
                         (dp_pred_ori['pred_y'].to_numpy() - dp_pred_ori['gt_y'].to_numpy())**2 +
                         (dp_pred_ori['pred_z'].to_numpy() - dp_pred_ori['gt_z'].to_numpy())**2)
    R_id = R_id_load['RANSAC ID'].to_numpy()
    # if len(R_id) > 160:
    #     continue
    no_R_holder.append(loss_array)
    R_holder.append(loss_array[R_id])
    rm_R_holder.append(np.delete(loss_array, R_id))
no_R_holder = np.concatenate(no_R_holder, axis=0)
R_holder = np.concatenate(R_holder, axis=0)
rm_R_holder = np.concatenate(rm_R_holder, axis=0)

# d = {'w/o RANSAC': no_R_holder,
#      'w RANSAC': R_holder,
#      'rm RANSAC': rm_R_holder}
d = {'rm RANSAC': rm_R_holder,
     'w RANSAC': R_holder}

ddf = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))

# ddf = pd.DataFrame({'w/o RANSAC': no_R_holder,
#                     'w RANSAC': R_holder})

# colors = ['#ff5447', '#a259a3', '#455eff']
colors = ['#f4cc70', '#20948b']
sns.set_palette(sns.color_palette(colors))

f, ax = plt.subplots(figsize=(12, 7))
sns.despine(f)

sns.histplot(
    ddf,
    edgecolor=".3",
    linewidth=.5,
    log_scale=True,
    bins=30,
    multiple="stack"
)

# ax = sns.displot(ddf, kind="kde", hist_kws={'log': True})
ax.set_xlabel("Euclidean distance (mm)", fontsize=20)
ax.set_ylabel("Count", fontsize=20)
# ax.legend(['rm RANSAC', 'w RANSAC', 'w/o RANSAC'], fontsize=20)
ax.legend(['with RANSAC', 'Outliers'], fontsize=20)
ax.tick_params(labelsize=20)
plt.show()
c = 1