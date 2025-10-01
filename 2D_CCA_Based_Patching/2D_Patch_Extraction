folder_path = 'D:/MICCAI1/'

# Loop over subfolders and files
for subdir, dirs, files in os.walk(folder_path):
    for file in files:
        # Check if the file name contains 'flair.nii'
        if 'flair.nii' in file:
            # Construct the file path
            file_path = os.path.join(subdir, file)

            # Load the file as a numpy array
            test_image_flair = nib.load(file_path).get_fdata()

            volume=test_image_flair
            #here you can give your 2D numpy array for further processing
            volume= test_image_flair[:,:,65]
            blurred_volume = skimage.filters.gaussian(volume, sigma=3, multichannel=False)
            new_vol= blurred_volume[56:184, 56:184]
            thresh = skimage.filters.threshold_yen(blurred_volume)


            binary_volume = blurred_volume > thresh
            labels = measure.label(binary_volume, background=0)

# Sort the connected components by size
            regions = measure.regionprops(labels)
            sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)


            biggest_region = sorted_regions[0]
            biggest_component_mask = (labels == biggest_region.label).astype('uint8') * 1
            np.unique(biggest_component_mask )
            image= biggest_component_mask * volume
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
            test_image_t1ce = nib.load(file_path).get_fdata()

            # Apply processing to the array
            # The coresponding 2D array for patch extraction
            volume=test_image_t1ce[:,:,65]
            data = volume


            print(centroid1)
            centroid = centroid1
                        
            data_shape = data.shape

# calculate the starting and ending indices for the patch
            start_idx = np.array(centroid) - np.array([64, 64])
            end_idx = np.array(centroid) + np.array([64, 64])


            for i in range(2):
                if start_idx[i] < 0:
                    end_idx[i] -= start_idx[i]
                    start_idx[i] = 0
                if end_idx[i] > data_shape[i]:
                    start_idx[i] -= end_idx[i] - data_shape[i]
                    end_idx[i] = data_shape[i]

            patch_size = end_idx - start_idx

            patch = np.zeros(patch_size)

            patch_start_idx = np.array([0, 0])
            patch_end_idx = patch_start_idx + patch_size

            data_start_idx = start_idx
            data_end_idx = end_idx

            patch[patch_start_idx[0]:patch_end_idx[0], patch_start_idx[1]:patch_end_idx[1]] = data[data_start_idx[0]:data_end_idx[0], data_start_idx[1]:data_end_idx[1]]

    # create a 2D numpy array with random values
            data = patch
    # save it in a format that you need for further processing
    
            #sitk_image = sitk.GetImageFromArray(data)
            #sitk.WriteImage(sitk_image, file_path)
