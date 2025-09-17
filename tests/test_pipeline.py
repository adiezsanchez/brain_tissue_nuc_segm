from pathlib import Path
import napari
import tifffile
import warnings
import os
import gc
import numpy as np
from tqdm import tqdm
from Stardist.utils_stardist import get_gpu_details, list_images, check_files, read_image, extract_nuclei_stack, get_stardist_model, segment_nuclei, remove_labels_touching_roi_edge

get_gpu_details()

# Absolute path to the pixi_test_data folder
project_root = Path(__file__).parent.parent.resolve()
directory_path = project_root / "assets" / "pixi_test_data"

# Construct ROI path from directory_path above
roi_path = directory_path / "ROIs"

# Iterate through the .czi and .nd2 files in the pixi_test_data directory
# If your images have a different file format (i.e. .tif), change the function below like this: list_images(directory_path, format=".tif")
images = list_images(directory_path, format=".tif")

# Image size reduction (downsampling) to improve processing times (slicing, not lossless compression)
# Now, in addition to xy, you can downsample across your z-stack
slicing_factor_xy = None # Use 2 or 4 for downsampling in xy (None for lossless)
slicing_factor_z = None # Use 2 to select 1 out of every 2 z-slices

# Define the nuclei and markers of interest channel order ('Remember in Python one starts counting from zero')
nuclei_channel = 3

# The n_tiles parameter defines the number of tiles the input volume/image will be divided into along each dimension (z, y, x) during prediction. 
# This is useful for processing large images that may not fit into memory at once.
# While tiling can handle memory limitations, chopping the image into smaller chunks increases
# the processing time for stitching the predictions back together. 
# Use n_tiles=(1, 1, 1) if the input volume fits in memory without tiling to minimize processing overhead.
n_tiles=(1,1,1)

# Segmentation type ("2D" or "3D").
# Choose 2D if your input image has no z-dimension (not a 3D-stack, but a single plane 2D multichannel image) 
# 2D also takes a z-stack as input, performs MIP (Maximum Intensity Projection) and predicts nuclei from the resulting projection (faster, useful for single layers of cells)
# 3D is more computationally expensive. Predicts 3D nuclear volumes, useful for multilayered structures
segmentation_type = "3D"

# Nuclear segmentation model type ("Stardist")
# Choose your Stardist fine-tuned model (model_name) from stardist_models folder
# If no custom model is present, type "test" and a standard pre-trained model will be loaded
model_name = "MEC0.1" # Type "test" if you don't have a custom model trained

# Model loading 
model = get_stardist_model(segmentation_type, name=model_name, basedir='./Stardist/stardist_models')

# Check if (all) ROIs are present before starting the nuclei prediction
check_files(images, directory_path, segmentation_type, model_name, filetype='roi')

# Initialize napari viewer
viewer = napari.Viewer(ndisplay=3)

# List of subfolder names
try:
    roi_names = [folder.name for folder in roi_path.iterdir() if folder.is_dir()]
    print(f"The following regions of interest will be analyzed: {roi_names}")
except FileNotFoundError:
    roi_names = ["full_image"]

for image in tqdm(images):
    for roi_name in roi_names:
        
        # Check if the prediction has already been generated
        file_path = Path(image)
        filename = file_path.stem
        pred_file = directory_path / "nuclei_preds" / segmentation_type / model_name / roi_name / f"{filename}.tiff"
        
        if pred_file.exists():
            print(f"\nWARNING: Nuclei predictions already found for: {filename} ROI: {roi_name}")
            print("Make sure nuclei labels were generated using the same settings.")
            continue  # Skip to the next roi_name if the prediction exists
        # Proceed to generate predictions if the file is not found

        # Read image, apply slicing if needed and return filename and img as a np array
        img, filename = read_image(image, slicing_factor_xy, slicing_factor_z)

        # Visualize image in Napari
        viewer.add_image(img, name=filename)

        # If 3D-segmentation input nuclei_img is a 3D-stack
        if segmentation_type == "3D":
            # Slice the nuclei stack
            nuclei_img = extract_nuclei_stack(img, nuclei_channel)
            print(f"Generating {segmentation_type} nuclei predictions for {roi_name} ROI")

        # If 2D-segmentation input nuclei_img is a max itensity projection of said 3D-stack
        elif segmentation_type == "2D":
            # Slice the nuclei stack
            nuclei_img = extract_nuclei_stack(img, nuclei_channel)
            nuclei_img = np.max(nuclei_img, axis=0)
            print(f"Generating {segmentation_type} nuclei predictions for {roi_name} ROI")

        # Construct path to read ROI
        roi_path = directory_path / "ROIs" / roi_name / f"{filename}.tiff"

        try:
            # Read the .tiff files containing the user-defined ROIs
            roi = tifffile.imread(roi_path)

            if np.all(roi == 0):
                warnings.warn("ROI file is empty (all zeros). Nuclei prediction running across the entire image", UserWarning)
                roi = np.ones_like(roi, dtype=np.uint8)  # Replace with ones

            # We will create a mask where roi is greater than or equal to 1
            mask = (roi >= 1).astype(np.uint8)

            # 3D segmentation logic, extend 2D mask across the entire stack volume
            if segmentation_type == "3D":
                # Extract the number of z-slices to extend the mask
                slice_nr = img.shape[1]

                # Extend the mask across the entire volume
                mask = np.tile(mask, (slice_nr, 1, 1))

                # Apply the mask to nuclei_img and marker_img, setting all other pixels to 0
                masked_nuclei_img = np.where(mask, nuclei_img, 0)
                
            elif segmentation_type == "2D":
                # Apply the mask to nuclei_img and marker_img, setting all other pixels to 0
                masked_nuclei_img = np.where(mask, nuclei_img, 0)

            # Clean up variables to free memory
            del mask
            gc.collect()

        except FileNotFoundError:
            # If no ROI is saved the script will predict nuclei in the entire nuclei_img input
            masked_nuclei_img = nuclei_img

        # Segment nuclei and return labels
        nuclei_labels = segment_nuclei(masked_nuclei_img, segmentation_type, model, n_tiles)

        # Remove labels touching ROI edge
        try:
            print("Removing nuclei labels touching ROI edge")
            nuclei_labels = remove_labels_touching_roi_edge(nuclei_labels, roi)
        except NameError: # Generate ROI around the entire image
            roi = np.ones(nuclei_labels.shape[-2:], dtype=np.int8)
            nuclei_labels = remove_labels_touching_roi_edge(nuclei_labels, roi)
        # Clean up variables to free memory
        del roi

        # Add labels to Napari
        viewer.add_labels(nuclei_labels, name=f"{filename}_nuclei")

        # Save nuclei labels as .tiff files to reuse them later
        # Create nuclei_predictions directory if it does not exist
        try:
            os.makedirs(directory_path / "nuclei_preds" / segmentation_type / model_name / roi_name)
        except FileExistsError:
            pass

        # Construct path to store
        nuclei_preds_path = directory_path / "nuclei_preds" / segmentation_type / model_name / roi_name / f"{filename}.tiff"

        # Save nuclei labels as .tiff
        tifffile.imwrite(nuclei_preds_path, nuclei_labels)

# Keep the viewer open
napari.run()

print("\nNuclei prediction completed")