import scipy.io as sio
import os
import h5py


def load_mat(mat_file):
    # only for matlab of lower versions
    return sio.loadmat(mat_file)


def load_mat_h5py(mat_file):
    f = h5py.File(mat_file)
    cm_data = f['CM']
    cnh_data = f['CNH']
    hog_data = f['HOG']
    lbp_data = f['LBP']
    print(cm_data)


if __name__ == '__main__':
    dataset_folder = r'E:\violent_reference\VSD\VSD_2014_December_official_release'
    visual_mat = os.path.join(dataset_folder, 'Hollywood-dev', 'features', 'Armageddon_visual.mat')
    load_mat_h5py(visual_mat)
