# SAH Segmentation

### Introduction
This document describes the usage of the SAH segmentation tool developed by HUS and CGI.
Basic skills in working with Python scripts are required to use the tool.

### Steps
1. The program requires Python 3, and it has been tested with version 3.7.
Install the required packages:

        pip install -r requirements.txt
Note: If you have a machine with a GPU and CUDA 10.0, you can replace `tensorflow-cpu` with `tensorflow-gpu` in the requirements file for significantly faster inference.

2. We assume that the original CT images are in DICOM format, inside the `<dicom_dir>` directory, with each patient in its own subdirectory.
The algorithm takes the images in NIFTI format, downsampled to 256&times;256 pixels and windowed to 0&ndash;150 HU.
Run the preprocessing script to convert the files and save the results to `<nifti_dir>`:

        python preprocess.py <dicom_dir> <nifti_dir>

3. Add the NIFTI image path to `job/config.ini` as the `path_to_search` value:

        [img]
        path_to_search = <nifti_dir>

4. Run the algorithm:

        python run.py --task inference --config job/config.ini --job-dir job

5. The segmentations are now available as NIFTI files in `job/predictions`.
