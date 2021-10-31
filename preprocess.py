import os
import argparse
import warnings
import numpy as np
import nilearn.image
import nibabel as nib

warnings.filterwarnings("ignore")

def main():
    input_dir, output_dir = get_args()
    print('Processing...')
    os.makedirs(output_dir, exist_ok=True)
    for roots,dirs,files in os.walk(input_dir):
        for f in files:
            img = nib.load(roots+'/'+f)
            print('loaded',f)
            preprocess_nifti(img).to_filename(os.path.join(output_dir, f))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory containing dicom image directories')
    parser.add_argument('output', help='output directory for nifti files')
    args = parser.parse_args()
    return os.path.expanduser(args.input), os.path.expanduser(args.output)

def preprocess_nifti(img, downsample=2, size=256, clip=(0, 150), dtype=np.int16):
    new_affine = img.affine.copy()
    new_affine[:3, :3] = np.matmul(img.affine[:3, :3], np.diag((downsample, downsample, 1)))
    min_value = img.get_fdata().min()
    tmp_img = nilearn.image.resample_img(img, target_affine=new_affine,
        target_shape=(size, size, img.shape[2]), fill_value=min_value)
    data = tmp_img.get_fdata()
    if clip:
        data = data.clip(min=clip[0], max=clip[1])
    return nib.Nifti1Image(data.astype(dtype), tmp_img.affine)

if __name__ == '__main__':
    main()
