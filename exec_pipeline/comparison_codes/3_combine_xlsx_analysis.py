import pandas as pd


def read_entire_trial_files(path_format, trial_num):
    folds = [0, 1, 2, 3, 4]
    file_folder = []
    for f in folds:
        xlsx_f_path = path_format.format(f, trial_num)
        file_folder.append(pd.read_excel(xlsx_f_path))
    return pd.concat(file_folder)


if __name__ == "__main__":
    test_id = [0]
    prior_identifier = 'nmi'
    post_identifier = 'sup'
    x_path_format = 'results_%s_f{}_{}_%s.xlsx' % (prior_identifier, post_identifier)
    x_save_path = 'stats_%s_%s.xlsx' % (prior_identifier, post_identifier)
    test_holder = []
    for z in test_id:
        test_holder.append(read_entire_trial_files(x_path_format, z))

    writer = pd.ExcelWriter(x_save_path, engine='xlsxwriter')
    # Write each dataframe to a different worksheet.
    for z, df in zip(test_id, test_holder):
        df.to_excel(writer, sheet_name='Trial_{}'.format(z), index=False)
    writer.save()
