import numpy as np
import persim
import cripser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.pyplot as plt
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import nibabel as nib # common way of importing nibabel
from skimage import io
from skimage import filters
import skimage.io
import skimage.color
import skimage.filters
from skimage.filters import median
from skimage.morphology import disk

def vec(ph,mx,my,mz,dims=None, max_life=-1):
    if dims is None:
        if mz == 1:
            dims = [0,1]
        else:
            dims = [0,1,2]

    if max_life<0:
        max_life = np.quantile(ph[:,2]-ph[:,1],0.8)
        #print(max_life)
    life_vec = np.zeros((len(dims),mx,my,mz))
    for c in ph:
        d = int(c[0]) # dim
        if d in dims:
            life = min(c[2]-c[1],max_life)
            di = dims.index(d)
            x,y,z=int(c[3]),int(c[4]),int(c[5]) # location
            life_vec[di,x,y,z] = max(life_vec[di,x,y,z],life)

    return(np.squeeze(life_vec))

def extract_patch(img, centroid, patch_size):
    h, w = img.shape
    x, y = centroid
    x, y = int(x), int(y)
    patch = img[x-patch_size//2:x+patch_size//2, y-patch_size//2:y+patch_size//2]
    return patch



import time
start_time = time.time()

mri_file2 = 'E:/Dataset/MICCAI_BraTS2020_TrainingData/BraTS20_Training_308/BraTS20_Training_308_seg.nii'
#mri_file2= 'E:/Dataset/aga khan/BraTS2021_00030/00000046_brain_flair.nii'
img2 = nib.load(mri_file2)
#img1 = nib.load(mri_file1)
img_data2 = img2.get_fdata()


folder_path = 'D:/BRATS/BRATS_Original/'

# Loop over subfolders and files
for subdir, dirs, files in os.walk(folder_path):
    for file in files:
    # Check if the file name contains 'flair.nii'
        if 't2.nii' in file:
            # Construct the file path
            file_path = os.path.join(subdir, file)
            print(file_path)
            # Load the file as a numpy array
            test_image_t1ce = nib.load(file_path).get_fdata()

            # Apply processing to the array
            # Example processing: normalize the values to be between 0 and 1
            volume=test_image_t1ce
            t1ce_data = volume

# assuming the 3D centroid is stored as a tuple of coordinates (x, y, z)
            dimension = 2
            slices = t1ce_data.shape[dimension]

            # Parameters for patch extraction
            patch_size = 128

            # Loop through all slices and extract patches
            skip_slices = []
            for i in range(t1ce_data.shape[2]):
                image = t1ce_data[:, :, i]
                sum_of_values = image.sum()
                if sum_of_values == 0:
                    skip_slices.append(i)

# assuming the 3D centroid is stored as a tuple of coordinates (x, y, z)
                    #patch = extract_patch(image, centroid, patch_size)

                    # Save the patch as an NPY file
                   # output_file = f"E:/Dataset/BraTS2020_TrainingDataa/t1ce_patch/slice_{file}_{i}.npy"
                    #np.save(output_file, patch)

                    #print(f"Saved slice {i} as {output_file}")
                else:
                    print(f"Skipping slice {i} (Sum of pixel values is zero)")
    for file in files:
        # Check if the file name contains 'flair.nii'
        if 'flair.nii' in file:
            # Construct the file path
            file_path = os.path.join(subdir, file)

            # Load the file as a numpy array
            test_image_flair = nib.load(file_path).get_fdata()

            # Apply processing to the array
            # Example processing: normalize the values to be between 0 and 1
            img_data=test_image_flair
            #test_mask = test_mask[56:184, 56:184, 13:141]
            # Choose the dimension (e.g., dimension 2) and extract all slices
            dimension = 2
            slices = img_data.shape[dimension]

            # Parameters for patch extraction
            patch_size = 128

            # Loop through all slices and extract patches
            for i in range(img_data.shape[2]):
                if i in skip_slices:
                    print(f"Skipping slice {i} in 't1.nii'")
                    continue

                # Get the current slice
                image1 = img_data[:, :, i]
                img = skimage.filters.gaussian(image1, sigma=3, output=None, mode='nearest', cval=0, preserve_range=False, truncate=3.0)
                fig = skimage.filters.threshold_yen(img)
                binary = image1 > fig
                oimage = binary * image1

                    # Perform further processing and patch extraction
                    # ...
                    # The rest of your code goes here

                    # Calculate the centroid and extract the patch
                centroid = np.mean(np.nonzero(oimage), axis=1)
            else:
                print(f"Skipping slice {i} (Sum of pixel values is zero)")

# Loop over subfolders and files
    for file in files:
        # Check if the file name contains 'flair.nii'
        if 't2.nii' in file:
            # Construct the file path
            file_path = os.path.join(subdir, file)
            print(file_path)
            # Load the file as a numpy array
            test_image_t1ce = nib.load(file_path).get_fdata()

            # Apply processing to the array
            # Example processing: normalize the values to be between 0 and 1
            t2_data=test_image_t1ce
            dimension = 2
            slices = t2_data.shape[dimension]

            # Parameters for patch extraction
            patch_size = 128

            # Loop through all slices and extract patches
            for i in range(t2_data.shape[2]):
                if i in skip_slices:
                    print(f"Skipping slice {i} in 't2.nii'")
                    continue

                # Get the current slice
                image = t2_data[:, :, i]
            # assuming the 3D centroid is stored as a tuple of coordinates (x, y, z)
                patch = extract_patch(image, centroid, patch_size)

                    # Save the patch as an NPY file
                output_file = f"D:/2D patches/2D Patches Axial/t2_patch/slice_{file}_{i}.npy"
                np.save(output_file, patch)

                print(f"Saved slice {i} as {output_file}")
            else:
                print(f"Skipping slice {i} (Sum of pixel values is zero)")
    for file in files:
        # Check if the file name contains 'flair.nii'
        if 't1ce.nii' in file:
            # Construct the file path
            file_path = os.path.join(subdir, file)
            print(file_path)
            # Load the file as a numpy array
            test_image_t1ce = nib.load(file_path).get_fdata()

            # Apply processing to the array
            # Example processing: normalize the values to be between 0 and 1
            volume=test_image_t1ce
            t1ce_data = volume

# assuming the 3D centroid is stored as a tuple of coordinates (x, y, z)
            dimension = 2
            slices = t1ce_data.shape[dimension]

            # Parameters for patch extraction
            patch_size = 128

            # Loop through all slices and extract patches
            for i in range(t1ce_data.shape[2]):
                if i in skip_slices:
                    print(f"Skipping slice {i} in 't1ce.nii'")
                    continue

                # Get the current slice
                image = t1ce_data[:, :, i]
            # assuming the 3D centroid is stored as a tuple of coordinates (x, y, z)
                patch = extract_patch(image, centroid, patch_size)

                    # Save the patch as an NPY file
                output_file = f"D:/2D patches/2D Patches Axial/t1ce_patch/slice_{file}_{i}.npy"
                np.save(output_file, patch)

                print(f"Saved slice {i} as {output_file}")
            else:
                print(f"Skipping slice {i} (Sum of pixel values is zero)")
    for file in files:
        if 't1.nii' in file:
            # Construct the file path
            file_path = os.path.join(subdir, file)
            print(file_path)
            # Load the file as a numpy array
            test_image_t1ce = nib.load(file_path).get_fdata()

            # Apply processing to the array
            # Example processing: normalize the values to be between 0 and 1
            volume=test_image_t1ce
            t1_data = volume
            dimension = 2
            slices = t1_data.shape[dimension]

            # Parameters for patch extraction
            patch_size = 128

            # Loop through all slices and extract patches
            for i in range(t1_data.shape[2]):
                if i in skip_slices:
                    print(f"Skipping slice {i} in 't1.nii'")
                    continue

                # Get the current slice
                image = t1_data[:, :, i]
            # assuming the 3D centroid is stored as a tuple of coordinates (x, y, z)
                patch = extract_patch(image, centroid, patch_size)

                    # Save the patch as an NPY file
                output_file = f"D:/2D patches/2D Patches Axial/t1_patch/slice_{file}_{i}.npy"
                np.save(output_file, patch)

                print(f"Saved slice {i} as {output_file}")
            else:
                print(f"Skipping slice {i} (Sum of pixel values is zero)")
# assuming the 3D centroid is stored as a tuple of coordinates (x, y, z)
    
    for file in files:
        if 'flair.nii' in file:
            # Construct the file path
            file_path = os.path.join(subdir, file)
            print(file_path)
            # Load the file as a numpy array
            test_image_t1ce = nib.load(file_path).get_fdata()

            # Apply processing to the array
            # Example processing: normalize the values to be between 0 and 1
            volume=test_image_t1ce
            flair_data = volume

    # assuming the 3D centroid is stored as a tuple of coordinates (x, y, z)
            dimension = 2
            slices = flair_data.shape[dimension]

            # Parameters for patch extraction
            patch_size = 128

            # Loop through all slices and extract patches
            for i in range(flair_data.shape[2]):
                if i in skip_slices:
                    print(f"Skipping slice {i} in 'falir.nii'")
                    continue

                # Get the current slice
                image = flair_data[:, :, i]
            # assuming the 3D centroid is stored as a tuple of coordinates (x, y, z)
                patch = extract_patch(image, centroid, patch_size)

                    # Save the patch as an NPY file
                output_file = f"D:/2D patches/2D Patches Axial/flair_patch/slice_{file}_{i}.npy"
                np.save(output_file, patch)

                print(f"Saved slice {i} as {output_file}")
            else:
                print(f"Skipping slice {i} (Sum of pixel values is zero)")
        
    for file in files:
        if 'seg.nii' in file:
            # Construct the file path
            file_path = os.path.join(subdir, file)
            print(file_path)
            # Load the file as a numpy array
            test_image_t1ce = nib.load(file_path).get_fdata()

            # Apply processing to the array
            # Example processing: normalize the values to be between 0 and 1
            volume=test_image_t1ce
            seg_data = volume

# assuming the 3D centroid is stored as a tuple of coordinates (x, y, z)
            dimension = 2
            slices = seg_data.shape[dimension]

            # Parameters for patch extraction
            patch_size = 128

            # Loop through all slices and extract patches
            for i in range(seg_data.shape[2]):
                if i in skip_slices:
                    print(f"Skipping slice {i} in 'seg.nii'")
                    continue

                # Get the current slice
                image = seg_data[:, :, i]
            # assuming the 3D centroid is stored as a tuple of coordinates (x, y, z)
                patch = extract_patch(image, centroid, patch_size)

                    # Save the patch as an NPY file
                output_file = f"D:/2D patches/2D Patches Axial/seg_patch/slice_{file}_{i}.npy"
                np.save(output_file, patch)

                print(f"Saved slice {i} as {output_file}")
            else:
                print(f"Skipping slice {i} (Sum of pixel values is zero)")
    




