import numpy as np
import skimage.filters 
import skimage
from skimage import measure
import SimpleITK as sitk
import numpy as np
import os
import numpy as np
import nibabel as nib
import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tifffile import imsave

import time
start_time = time.time()
folder_path = "D:/brats_training_data/"

# Loop over subfolders and files
for subdir, dirs, files in os.walk(folder_path):
    for file in files:
        # Check if the file name contains 'flair.nii'
        if 'flair.nii.gz' in file:
            # Construct the file path
            file_path = os.path.join(subdir, file)

            # Load the file as a numpy array
            test_image_flair = sitk.ReadImage(file_path)
            flair_data = sitk.GetArrayFromImage(test_image_flair)
            volume=flair_data
#test_mask = test_mask[56:184, 56:184, 13:141]
            blurred_volume = skimage.filters.gaussian(volume, sigma=1.0, multichannel=False)
            new_vol= blurred_volume[56:184, 56:184,56:184]
            thresh = skimage.filters.threshold_yen(new_vol)


            binary_volume = blurred_volume > thresh
            labels, num_labels = skimage.measure.label(binary_volume,connectivity=2, return_num=True)
            labels1 = measure.label(labels, background=0)

# Sort the connected components by size
            regions = measure.regionprops(labels1)
            sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)


            biggest_region = sorted_regions[0]
            biggest_component_mask = (labels == biggest_region.label).astype('uint8') * 1
            np.unique(biggest_component_mask )
            image= biggest_component_mask * flair_data
            centroid = np.mean(np.nonzero(image), axis=1) # find the centroid
            centroid1 =np.rint(centroid).astype(int)
            

# Loop over subfolders and files
    for file in files:
        # Check if the file name contains 'flair.nii'
        if 't2.nii' in file:
            # Construct the file path
            file_path = os.path.join(subdir, file)
            print(file_path)
            # Load the file as a numpy array
            test_image_t2 = sitk.ReadImage(file_path)
            t2_data = sitk.GetArrayFromImage(test_image_t2)
            # Apply processing to the array
            # Example processing: normalize the values to be between 0 and 1
            volume= t2_data
            data = volume


            print(centroid1)
            centroid = centroid1
                        
            data_shape = data.shape

# calculate the starting and ending indices for the patch
            start_idx = np.array(centroid) - np.array([64, 64, 64])
            end_idx = np.array(centroid) + np.array([64, 64, 64])


            for i in range(3):
                if start_idx[i] < 0:
                    end_idx[i] -= start_idx[i]
                    start_idx[i] = 0
                if end_idx[i] > data_shape[i]:
                    start_idx[i] -= end_idx[i] - data_shape[i]
                    end_idx[i] = data_shape[i]

            patch_size = end_idx - start_idx

            patch = np.zeros(patch_size)

            patch_start_idx = np.array([0, 0, 0])
            patch_end_idx = patch_start_idx + patch_size

            data_start_idx = start_idx
            data_end_idx = end_idx

            patch[patch_start_idx[0]:patch_end_idx[0], patch_start_idx[1]:patch_end_idx[1], patch_start_idx[2]:patch_end_idx[2]] = data[data_start_idx[0]:data_end_idx[0], data_start_idx[1]:data_end_idx[1], data_start_idx[2]:data_end_idx[2]]

    # create a 3D numpy array with random values
            data = patch

    # create a SimpleITK image from the numpy array
            sitk_image = sitk.GetImageFromArray(data)

    # set the origin and spacing of the image (optional)
            sitk_image.SetOrigin((0, -239, 0))
            sitk_image.SetSpacing((1, 1, 1))
            sitk.WriteImage(sitk_image, file_path)

            
    for file in files:
        # Check if the file name contains 'flair.nii'
        if 'seg.nii.gz' in file:
            # Construct the file path
            file_path = os.path.join(subdir, file)
            print(file_path)
            # Load the file as a numpy array
            test_image_seg = sitk.ReadImage(file_path)
            seg_data = sitk.GetArrayFromImage(test_image_seg)
            # Apply processing to the array
            # Example processing: normalize the values to be between 0 and 1
            seg_data[seg_data==4] = 3
            volume=seg_data
            data = volume

# assuming the 3D centroid is stored as a tuple of coordinates (x, y, z)
            print(centroid1)
            centroid = centroid1
                        

            data_shape = data.shape

            start_idx = np.array(centroid) - np.array([64, 64, 64])
            end_idx = np.array(centroid) + np.array([64, 64, 64])


            for i in range(3):
                if start_idx[i] < 0:
                    end_idx[i] -= start_idx[i]
                    start_idx[i] = 0
                if end_idx[i] > data_shape[i]:
                    start_idx[i] -= end_idx[i] - data_shape[i]
                    end_idx[i] = data_shape[i]

    # calculate the actual size of the patch
            patch_size = end_idx - start_idx

   
            patch = np.zeros(patch_size)

  
            patch_start_idx = np.array([0, 0, 0])
            patch_end_idx = patch_start_idx + patch_size

    # calculate the starting and ending indices of the patch within the original data
            data_start_idx = start_idx
            data_end_idx = end_idx

    # extract the patch from the original data and store it in the patch array
            patch[patch_start_idx[0]:patch_end_idx[0], patch_start_idx[1]:patch_end_idx[1], patch_start_idx[2]:patch_end_idx[2]] = data[data_start_idx[0]:data_end_idx[0], data_start_idx[1]:data_end_idx[1], data_start_idx[2]:data_end_idx[2]]

    # create a 3D numpy array with random values
            data = patch

    # create a SimpleITK image from the numpy array
            sitk_image = sitk.GetImageFromArray(data)

    # set the origin and spacing of the image (optional)
            sitk_image.SetOrigin((0, -239, 0))
            sitk_image.SetSpacing((1, 1, 1))
            sitk.WriteImage(sitk_image, file_path)

            # Save the modified array back to the same file
            #np.save(file_path, arr_normalized)
    for file in files:
        # Check if the file name contains 'flair.nii'
        if 't1ce.nii.gz' in file:
            # Construct the file path
            file_path = os.path.join(subdir, file)
            print(file_path)
            # Load the file as a numpy array
            test_image_t1ce = sitk.ReadImage(file_path)
            t1ce_data = sitk.GetArrayFromImage(test_image_t1ce)
            # Apply processing to the array
            # Example processing: normalize the values to be between 0 and 1
            volume=t1ce_data
            data = volume

# assuming the 3D centroid is stored as a tuple of coordinates (x, y, z)
            print(centroid1)
            centroid = centroid1
                        

# assuming the original data is stored in a numpy array called 'data'
            data_shape = data.shape

# calculate the starting and ending indices for the patch
            start_idx = np.array(centroid) - np.array([64, 64, 64])
            end_idx = np.array(centroid) + np.array([64, 64, 64])


            for i in range(3):
                if start_idx[i] < 0:
                    end_idx[i] -= start_idx[i]
                    start_idx[i] = 0
                if end_idx[i] > data_shape[i]:
                    start_idx[i] -= end_idx[i] - data_shape[i]
                    end_idx[i] = data_shape[i]

    # calculate the actual size of the patch
            patch_size = end_idx - start_idx

    # create an empty array to store the patch
            patch = np.zeros(patch_size)

   
            patch_start_idx = np.array([0, 0, 0])
            patch_end_idx = patch_start_idx + patch_size

            data_start_idx = start_idx
            data_end_idx = end_idx

            patch[patch_start_idx[0]:patch_end_idx[0], patch_start_idx[1]:patch_end_idx[1], patch_start_idx[2]:patch_end_idx[2]] = data[data_start_idx[0]:data_end_idx[0], data_start_idx[1]:data_end_idx[1], data_start_idx[2]:data_end_idx[2]]

    # create a 3D numpy array with random values
            data = patch

    # create a SimpleITK image from the numpy array
            sitk_image = sitk.GetImageFromArray(data)

    # set the origin and spacing of the image (optional)
            sitk_image.SetOrigin((0, -239, 0))
            sitk_image.SetSpacing((1, 1, 1))
            sitk.WriteImage(sitk_image, file_path)

            # Save the modified array back to the same file
            #np.save(file_path, arr_normalized)
    for file in files:
        if 't1.nii' in file:
            # Construct the file path
            file_path = os.path.join(subdir, file)
            print(file_path)
            # Load the file as a numpy array
            test_image_t1 = sitk.ReadImage(file_path)
            t1_data = sitk.GetArrayFromImage(test_image_t1)
            # Apply processing to the array
            # Example processing: normalize the values to be between 0 and 1
            volume=t1_data
            data = volume

# assuming the 3D centroid is stored as a tuple of coordinates (x, y, z)
            print(centroid1)
            centroid = centroid1
                        

# assuming the original data is stored in a numpy array called 'data'
            data_shape = data.shape

# calculate the starting and ending indices for the patch
            start_idx = np.array(centroid) - np.array([64, 64, 64])
            end_idx = np.array(centroid) + np.array([64, 64, 64])

# adjust the starting and ending indices if they go beyond the boundaries of the original data
            for i in range(3):
                if start_idx[i] < 0:
                    end_idx[i] -= start_idx[i]
                    start_idx[i] = 0
                if end_idx[i] > data_shape[i]:
                    start_idx[i] -= end_idx[i] - data_shape[i]
                    end_idx[i] = data_shape[i]

    # calculate the actual size of the patch
            patch_size = end_idx - start_idx

    # create an empty array to store the patch
            patch = np.zeros(patch_size)

            patch_start_idx = np.array([0, 0, 0])
            patch_end_idx = patch_start_idx + patch_size

  
            data_start_idx = start_idx
            data_end_idx = end_idx

  
            patch[patch_start_idx[0]:patch_end_idx[0], patch_start_idx[1]:patch_end_idx[1], patch_start_idx[2]:patch_end_idx[2]] = data[data_start_idx[0]:data_end_idx[0], data_start_idx[1]:data_end_idx[1], data_start_idx[2]:data_end_idx[2]]

    # create a 3D numpy array with random values
            data = patch

    # create a SimpleITK image from the numpy array
            sitk_image = sitk.GetImageFromArray(data)

    # set the origin and spacing of the image (optional)
            sitk_image.SetOrigin((0, -239, 0))
            sitk_image.SetSpacing((1, 1, 1))
            sitk.WriteImage(sitk_image, file_path)

            # Save the modified array back to the same file
            #np.save(file_path, arr_normalized)
    for file in files:
        if 'flair.nii' in file:
            # Construct the file path
            file_path = os.path.join(subdir, file)
            print(file_path)
            # Load the file as a numpy array
            test_image_f = sitk.ReadImage(file_path)
            f_data = sitk.GetArrayFromImage(test_image_f)
            # Apply processing to the array
            # Example processing: normalize the values to be between 0 and 1
            volume=f_data
            data = volume

# assuming the 3D centroid is stored as a tuple of coordinates (x, y, z)
            print(centroid1)
            centroid = centroid1
                        

# assuming the original data is stored in a numpy array called 'data'
            data_shape = data.shape

# calculate the starting and ending indices for the patch
            start_idx = np.array(centroid) - np.array([64, 64, 64])
            end_idx = np.array(centroid) + np.array([64, 64, 64])


            for i in range(3):
                if start_idx[i] < 0:
                    end_idx[i] -= start_idx[i]
                    start_idx[i] = 0
                if end_idx[i] > data_shape[i]:
                    start_idx[i] -= end_idx[i] - data_shape[i]
                    end_idx[i] = data_shape[i]

            patch_size = end_idx - start_idx

            patch = np.zeros(patch_size)

    
            patch_start_idx = np.array([0, 0, 0])
            patch_end_idx = patch_start_idx + patch_size

            data_start_idx = start_idx
            data_end_idx = end_idx


            patch[patch_start_idx[0]:patch_end_idx[0], patch_start_idx[1]:patch_end_idx[1], patch_start_idx[2]:patch_end_idx[2]] = data[data_start_idx[0]:data_end_idx[0], data_start_idx[1]:data_end_idx[1], data_start_idx[2]:data_end_idx[2]]

    # create a 3D numpy array with random values
            data = patch

    # create a SimpleITK image from the numpy array
            sitk_image = sitk.GetImageFromArray(data)

    # set the origin and spacing of the image (optional)
            sitk_image.SetOrigin((0, -239, 0))
            sitk_image.SetSpacing((1, 1, 1))
            sitk.WriteImage(sitk_image, file_path)

            # Save the modified array back to the same file
            #np.save(file_path, arr_normalized)
end_time = time.time()

# Calculate and print the total processing time
total_time = end_time - start_time
print(f"Total processing time: {total_time} seconds") 
