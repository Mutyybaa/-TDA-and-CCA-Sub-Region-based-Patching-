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
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import skimage.measure
import skimage.filters
from skimage import measure

start_time = time.time()
folder_path = "D:/MICCAI1/"

# Create an empty list to store centroid coordinates
centroid_coordinates = []

# Loop over subfolders and files
for subdir, dirs, files in os.walk(folder_path):
    for file in files:
        # Check if the file name contains 'flair.nii'
        if 'flair.nii' in file:
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
            
            # Append the centroid coordinates to the list
            centroid_coordinates.append(centroid1)

# Create a DataFrame from the centroid coordinates list
df = pd.DataFrame(centroid_coordinates, columns=['x', 'y', 'z'])

# Save the DataFrame to an Excel file
df.to_excel('centroid_coordinates.xlsx', index=False)

print("Centroid coordinates saved to centroid_coordinates.xlsx")
