import h5py
import tifffile as tiff
from shutil import copy

def cvppp_ins_to_h5(pred_root, pred_file_name):

    template = 'submission_example.h5'  # This can be downloaded from the official cvppp challenge website.

    pred_h5 = pred_file_name + '.h5'

    copy(pred_root+ template, pred_root + pred_h5)

    file = h5py.File(pred_root + pred_h5, 'r+')

    a1_pred = file['A1']

    for data in a1_pred.keys():  # loop datasets in the current group
        # get label image
        fullkey =  'A1/' + data + '/label'
        tiff_path =pred_root +  pred_file_name + '/' + data + '_rgb.tif'
        pred_tiff = tiff.imread(tiff_path)

        inLabel = file[fullkey]
        inLabel[...] = pred_tiff

    file.close()



if __name__ == "__main__":
    pred_root = ''
    pred_file_name = ''  # The name of the folder containing your instance segmentation prediction in '.tif' format.

    cvppp_ins_to_h5(pred_root, pred_file_name)