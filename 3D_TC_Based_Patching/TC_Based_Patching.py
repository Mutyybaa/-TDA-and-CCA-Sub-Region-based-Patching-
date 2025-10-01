import numpy as np

def extract_patch(volume, centroid, patch_size):
    half_patch = patch_size // 2
    
    x_c, y_c, z_c = centroid
    
    # Calculate the start and end indices for each dimension, ensuring equal padding
    x_start = x_c - half_patch
    x_end = x_c + half_patch
    y_start = y_c - half_patch
    y_end = y_c + half_patch
    z_start = z_c - half_patch
    z_end = z_c + half_patch
    
    # Ensure the start and end indices are within the volume boundaries
    x_start = max(0, x_start)
    x_end = min(volume.shape[0] - 1, x_end)
    y_start = max(0, y_start)
    y_end = min(volume.shape[1] - 1, y_end)
    z_start = max(0, z_start)
    z_end = min(volume.shape[2] - 1, z_end)
    
    # Calculate the dimensions of the extracted_patch
    patch_shape = (patch_size, patch_size, patch_size)
    
    # Create an empty patch filled with zeros with the calculated shape
    extracted_patch = np.zeros(patch_shape, dtype=volume.dtype)
    
    # Calculate the padding needed for each dimension
    x_padding = patch_size - (x_end - x_start + 1)
    y_padding = patch_size - (y_end - y_start + 1)
    z_padding = patch_size - (z_end - z_start + 1)
    
    # Calculate the padding on each side
    x_padding_start = x_padding // 2
    x_padding_end = x_padding - x_padding_start
    y_padding_start = y_padding // 2
    y_padding_end = y_padding - y_padding_start
    z_padding_start = z_padding // 2
    z_padding_end = z_padding - z_padding_start
    
    # Copy the valid portion of the volume to the center of the patch
    extracted_patch[
        x_padding_start:x_padding_start + (x_end - x_start + 1),
        y_padding_start:y_padding_start + (y_end - y_start + 1),
        z_padding_start:z_padding_start + (z_end - z_start + 1)
    ] = volume[x_start:x_end + 1, y_start:y_end + 1, z_start:z_end + 1]

  import os
import numpy as np
import nibabel as nib
from skimage import filters, measure
import skimage.filters
import pandas as pd  # Import pandas for Excel operations
from skimage import exposure
import cv2

# ... (Rest of your code)

# Initialize a list to store Dice scores
dice_scores = []


# Set your base directory
base_dir = 'D:\MICCAI1'
kernel_size = 25
def DICE_COE(mask1, mask2):
    intersect = np.sum(mask1*mask2)
    fsum = np.sum(mask1)
    ssum = np.sum(mask2)
    dice = (2 * intersect ) / (fsum + ssum)
    dice = np.mean(dice)
    dice = round(dice, 3) # for easy reading
    return dice
# Iterate through subdirectories
for subdir in os.listdir(base_dir):
    sub_dir_path = os.path.join(base_dir, subdir)

    # Check if the subdirectory contains the necessary files
    flair_path = os.path.join(sub_dir_path, f"{subdir}_flair.nii")
    t1ce_path = os.path.join(sub_dir_path, f"{subdir}_t1ce.nii")
    seg_path = os.path.join(sub_dir_path, f"{subdir}_seg.nii")
    t1_path = os.path.join(sub_dir_path, f"{subdir}_t1.nii")
    t2_path = os.path.join(sub_dir_path, f"{subdir}_t2.nii")

    if os.path.exists(flair_path) and os.path.exists(t1ce_path) and os.path.exists(seg_path) and os.path.exists(t1_path) and os.path.exists(t2_path):
        # Load the MRI images and masks
        flair_img = nib.load(flair_path)
        t1ce_img = nib.load(t1ce_path)
        seg_img = nib.load(seg_path)
        t1_img = nib.load(t1_path)
        t2_img = nib.load(t2_path)

        flair_data = flair_img.get_fdata()
        t1ce_data = t1ce_img.get_fdata()
        seg_data = seg_img.get_fdata()
        t1_data = t1_img.get_fdata()
        t2_data = t2_img.get_fdata()

        # Rest of your image processing operations
        # Rest of your image processing operations

        data_min = np.min(t1ce_data)
        data_max = np.max(t1ce_data)
        data = (t1ce_data - data_min) / (data_max - data_min)

# Perform local histogram equalization
        equalized_data = np.zeros_like(t1ce_data)
        for slice_index in range(t1ce_data.shape[0]):
            equalized_data[:, :, slice_index] = exposure.equalize_adapthist(data[:, :, slice_index], kernel_size=kernel_size)
            equalized_data[:, slice_index, :] = exposure.equalize_adapthist(data[:, slice_index, :], kernel_size=kernel_size)
            equalized_data[slice_index, :, :] = exposure.equalize_adapthist(data[slice_index, :, :], kernel_size=kernel_size)



        # Create a new NIfTI image with the equalized data
        equalized_img = nib.Nifti1Image(equalized_data, t1ce_img.affine, t1ce_img.header)

        volume1=equalized_img.get_fdata()
        #volume= t1ce_data
        #test_mask = test_mask[56:184, 56:184, 13:141]
        blurred_volume1 = skimage.filters.gaussian(volume1, sigma=0.2, multichannel=False)
        new_vol1= blurred_volume1[48:80, 48:80,48:80]
        thresh = skimage.filters.threshold_yen(new_vol1)


        binary_volume2 = blurred_volume1 > thresh
        #restored_volume = np.zeros((128, 128, 128))

        #restored_volume[48:80, 48:80,48:80] = binary_volume2
        #labels, num_labels = skimage.measure.label(binary_volume2,connectivity=2, return_num=True)
        
        #for label in range(1, num_labels):
            #component_size = np.sum(labels == label)
            #if component_size < 1000:
                #labels = np.where(labels == label, 0, labels)

# Convert the labeled image back to binary
        #labels[labels != 0] = 1
        labels1 = measure.label(binary_volume2, background=0)


        regions1 = measure.regionprops(labels1)
        sorted_regions1 = sorted(regions1, key=lambda x: x.area, reverse=True)


        biggest_region1 = sorted_regions1[0]
        biggest_component_mask2 = (labels1 == biggest_region1.label).astype('uint8') * 1
        image= biggest_component_mask2  * volume1
        centroid = np.mean(np.nonzero(image), axis=1)
        centroid1 = np.rint(centroid).astype(int)
        
        for file in os.listdir(sub_dir_path):
            file_path = os.path.join(sub_dir_path, file)

    # Process each file using the calculated centroid1
            if file.endswith('flair.nii'):
                # Load and process 'flair.nii' file using centroid1
                test_image_flair = nib.load(file_path).get_fdata()
                print(centroid1)
                # Replace this with your actual MRI volume data, centroid, and desired patch size
                volume = test_image_flair  # Example MRI volume (replace with actual data)
                centroid = centroid1   # Example centroid coordinates (replace with actual centroid)
                patch_size = 128  # Example patch size

                patch_around_centroid = extract_patch(volume, centroid, patch_size)
                data = patch_around_centroid


                sitk_image = sitk.GetImageFromArray(data)


                sitk_image.SetOrigin((0, -239, 0))
                sitk_image.SetSpacing((1, 1, 1))
                new_file_path = file_path.replace('.nii', '_modified.nii.gz')  # Create new file path with .nii.gz extension
                sitk.WriteImage(sitk_image, new_file_path)
                # ...
            elif file.endswith('t1ce.nii'):
                # Load and process 't1ce.nii' file using centroid1
                test_image_t1ce = nib.load(file_path).get_fdata()
                print(centroid1)
                volume = test_image_t1ce  # Example MRI volume (replace with actual data)
                centroid = centroid1   # Example centroid coordinates (replace with actual centroid)
                patch_size = 128  # Example patch size

                patch_around_centroid = extract_patch(volume, centroid, patch_size)
                data = patch_around_centroid


                sitk_image = sitk.GetImageFromArray(data)


                sitk_image.SetOrigin((0, -239, 0))
                sitk_image.SetSpacing((1, 1, 1))
                new_file_path = file_path.replace('.nii', '_modified.nii.gz')  # Create new file path with .nii.gz extension
                sitk.WriteImage(sitk_image, new_file_path)
                pass# ...
            elif file.endswith('t1.nii'):
                # Load and process 't1.nii' file using centroid1
                test_image_t1 = nib.load(file_path).get_fdata()
                print(centroid1)
                volume = test_image_t1  # Example MRI volume (replace with actual data)
                centroid = centroid1   # Example centroid coordinates (replace with actual centroid)
                patch_size = 128  # Example patch size

                patch_around_centroid = extract_patch(volume, centroid, patch_size)
                data = patch_around_centroid


                sitk_image = sitk.GetImageFromArray(data)


                sitk_image.SetOrigin((0, -239, 0))
                sitk_image.SetSpacing((1, 1, 1))
                new_file_path = file_path.replace('.nii', '_modified.nii.gz')  # Create new file path with .nii.gz extension
                sitk.WriteImage(sitk_image, new_file_path)
                pass# ...
            elif file.endswith('t2.nii'):
                # Load and process 't2.nii' file using centroid1
                test_image_t2 = nib.load(file_path).get_fdata()
                print(centroid1)
                volume = test_image_t2  # Example MRI volume (replace with actual data)
                centroid = centroid1   # Example centroid coordinates (replace with actual centroid)
                patch_size = 128  # Example patch size

                patch_around_centroid = extract_patch(volume, centroid, patch_size)
                data = patch_around_centroid


                sitk_image = sitk.GetImageFromArray(data)


                sitk_image.SetOrigin((0, -239, 0))
                sitk_image.SetSpacing((1, 1, 1))
                new_file_path = file_path.replace('.nii', '_modified.nii.gz')  # Create new file path with .nii.gz extension
                sitk.WriteImage(sitk_image, new_file_path)
                pass# ...
            elif file.endswith('seg.nii'):
                # Load and process 'seg.nii' file using centroid1
                test_image_seg = nib.load(file_path).get_fdata()
                print(centroid1)
                volume = test_image_seg  # Example MRI volume (replace with actual data)
                centroid = centroid1   # Example centroid coordinates (replace with actual centroid)
                patch_size = 128  # Example patch size

                patch_around_centroid = extract_patch(volume, centroid, patch_size)
                data = patch_around_centroid


                sitk_image = sitk.GetImageFromArray(data)


                sitk_image.SetOrigin((0, -239, 0))
                sitk_image.SetSpacing((1, 1, 1))
                new_file_path = file_path.replace('.nii', '_modified.nii.gz')  # Create new file path with .nii.gz extension
                sitk.WriteImage(sitk_image, new_file_path)
                pass# ...
    return extracted_patch
