from csbdeep.utils import normalize
from tensorflow.python.client import device_lib
from pathlib import Path
import czifile
import nd2
import tifffile
import napari
import numpy as np
import os
import pandas as pd
from skimage import exposure, filters, measure
from scipy.ndimage import binary_erosion
import pyclesperanto_prototype as cle

cle.select_device("RTX")

def get_gpu_details():
    devices = device_lib.list_local_devices()
    for device in devices:
        if device.device_type == 'GPU':
            print(f"Device name: {device.name}")
            print(f"Device type: {device.device_type}")
            print(f"GPU model: {device.physical_device_desc}")

def list_images (directory_path):

    # Create an empty list to store all image filepaths within the dataset directory
    images = []

    # Iterate through the .czi and .nd2 files in the directory
    for file_path in directory_path.glob("*.czi"):
        images.append(str(file_path))
        
    for file_path in directory_path.glob("*.nd2"):
        images.append(str(file_path))

    return images


def read_image (image, slicing_factor_xy, slicing_factor_z):
    """Read raw image microscope files, apply downsampling if needed and return filename and a numpy array"""
    # Read path storing raw image and extract filename
    file_path = Path(image)
    filename = file_path.stem

    # Extract file extension
    extension = file_path.suffix

    # Read the image file (either .czi or .nd2)
    if extension == ".czi":
        # Stack from .czi (ch, z, x, y)
        img = czifile.imread(image)
        # Remove singleton dimensions
        img = img.squeeze()

    elif extension == ".nd2":
        # Stack from .nd2 (z, ch, x, y)
        img = nd2.imread(image)
        # Transpose to output (ch, z, x, y)
        img = img.transpose(1, 0, 2, 3)

    else:
        print ("Implement new file reader")

    print(f"\n\nImage analyzed: {filename}")
    print(f"Original Array shape: {img.shape}")

    # Apply slicing trick to reduce image size (xy resolution)
    try:
        img = img[:, ::slicing_factor_z, ::slicing_factor_xy, ::slicing_factor_xy]
    except IndexError as e:
        print(f"Slicing Error: {e}")
        print(f"Slicing parameters: Slicing_XY:{slicing_factor_xy} Slicing_Z:{slicing_factor_z} ")

    # Feedback for researcher
    print(f"Compressed Array shape: {img.shape}")

    return img, filename

def maximum_intensity_projection (img):

    # Perform MIP on all channels 
    img_mip = np.max(img, axis=1)

    return  img_mip

def extract_nuclei_stack (img, nuclei_channel):

    # Extract nuclei stack from a multichannel z-stack (ch, z, x, y)
    nuclei_img = img[nuclei_channel, :, :, :]

    return nuclei_img

def get_stardist_model(segmentation_type, name, basedir='stardist_models'):
    """
    Load a StarDist model based on the specified segmentation type.

    Parameters:
    segmentation_type (str): Either '2D' or '3D', specifying the type of model to load.
    name (str): The name of the trained model to load.
    basedir (str, optional): Directory where custom trained models are stored. Default is 'stardist_models'.

    Returns:
    StarDist2D or StarDist3D model: The loaded StarDist model.

    If the specified model is not found, a pretrained model is loaded instead:
    - '2D_versatile_fluo' for 2D models
    - '3D_demo' for 3D models

    Raises:
    ValueError: If an invalid segmentation_type is provided.
    """

    if segmentation_type == "2D":
        from stardist.models import StarDist2D
        print(f"Loading {segmentation_type} segmentation model")
        try:
            return StarDist2D(None, name, basedir)
        except FileNotFoundError:
            print(f"Model {name} not found. Loading pretrained Stardist model '2D_versatile_fluo'.")
            return StarDist2D.from_pretrained("2D_versatile_fluo")

    elif segmentation_type == "3D":
        from stardist.models import StarDist3D
        print(f"Loading {segmentation_type} segmentation model")
        try:
            return StarDist3D(None, name, basedir)
        except FileNotFoundError:
            print(f"Model {name} not found. Loading pretrained Stardist model '3D_demo'.")
            return StarDist3D.from_pretrained("3D_demo")

    else:
        raise ValueError("segmentation_type must be '2D' or '3D'")

def segment_nuclei_3d(nuclei_img, model, n_tiles=None):
    
    normalized = normalize(nuclei_img)

    nuclei_labels, _ = model.predict_instances(normalized, n_tiles=n_tiles, show_tile_progress=True)

    return nuclei_labels

def segment_nuclei(nuclei_img, segmentation_type, model, n_tiles=None):

    if segmentation_type == "2D":
        # Ignore the z-dimension of the n_tiles tuple (x, y, z)
        n_tiles = n_tiles[-2:]
    
    normalized = normalize(nuclei_img)

    nuclei_labels, _ = model.predict_instances(normalized, n_tiles=n_tiles, show_tile_progress=True)

    return nuclei_labels

def save_rois(viewer, directory_path, filename):

    """Code snippet to save cropped regions (ROIs) defined by labels as .tiff files"""

    # Initialize empty list to store the label name and Numpy arrays so we can loop across the different ROIs
    layer_names = []
    layer_labels = []

    if len(viewer.layers) == 1:

        print("No user-defined ROIs have been stored")

    else:

        for layer in viewer.layers:

            # Extract the label names
            label_name = layer.name
            # Ignore img_mip since it is not a user defined label
            if label_name == "img_mip":
                pass
            else:
                # Store label names
                layer_names.append(label_name)
                # Get the label data as a NumPy array to mask the image
                label = layer.data 
                layer_labels.append(label)

        # Print the defined ROIs that will be analyzed
        print(f"The following labels will be analyzed: {layer_names}")

    # Save user-defined ROIs in a ROI folder under directory_path/ROI as .tiff files
    # Subfolders for each user-defined label region
    # Store using the same filename as the input image to make things easier

    for label_name, label_array in zip(layer_names, layer_labels):

        # Perform maximum intensity projection (MIP) from the label stack
        label_mip = np.max(label_array, axis=0)

        # We will create a mask where label_mip is greater than or equal to 1
        mask = (label_mip >= 1).astype(np.uint8)

        # Create ROI directory if it does not exist
        try:
            os.makedirs(directory_path / "ROIs" / label_name)
        except FileExistsError:
            pass

        # Construct path to store
        roi_path = directory_path / "ROIs" / label_name / f"{filename}.tiff"

        # Save mask (binary image)
        tifffile.imwrite(roi_path, mask)


def process_labels (viewer, directory_path, filename):
    """Stores user-defined labels in memory for masking input image and saves them as .tiff files"""

    # Initialize empty list to store the label name and Numpy arrays so we can loop across the different ROIs
    layer_names = []
    layer_labels = []

    if len(viewer.layers) == 1:

        # Extract the xy dimensions of the input image
        img_shape = viewer.layers[0].data.shape
        img_xy_dims = img_shape[-2:]

        # Create a label covering the entire image
        label = np.ones(img_xy_dims)

        # Add a name and the label to its corresponding list
        layer_names.append("full_image")
        layer_labels.append(label)

    else:

        for layer in viewer.layers:

            # Extract the label names
            label_name = layer.name
            # Ignore img_mip since it is not a user defined label
            if label_name == "img_mip":
                pass
            else:
                # Store label names
                layer_names.append(label_name)
                # Get the label data as a NumPy array to mask the image
                label = layer.data 
                layer_labels.append(label)

        # Print the defined ROIs that will be analyzed
        print(f"The following labels will be analyzed: {layer_names}")

    # Save user-defined ROIs in a ROI folder under directory_path/ROI as .tiff files
    # Subfolders for each user-defined label region
    # Store using the same filename as the input image to make things easier

    for label_name, label_array in zip(layer_names, layer_labels):

        if label_name == "full_image":
            print("Full image analyzed, no need to store ROIs")
            pass

        else:

            # Perform maximum intensity projection (MIP) from the label stack
            label_mip = np.max(label_array, axis=0)

            # We will create a mask where label_mip is greater than or equal to 1
            mask = (label_mip >= 1).astype(np.uint8)

            # Create ROI directory if it does not exist
            try:
                os.makedirs(directory_path / "ROIs" / label_name)
            except FileExistsError:
                pass

            # Construct path to store
            roi_path = directory_path / "ROIs" / label_name / f"{filename}.tiff"

            # Save mask (binary image)
            tifffile.imwrite(roi_path, mask)

    return layer_names, layer_labels

def segment_marker_positive_nuclei (nuclei_labels, marker_input, min_max_range, erosion_factor):

    if len(marker_input.shape) == 3:
        # Perform maximum intensity projection from the stack
        marker_mip = np.max(marker_input, axis=0)

    elif len(marker_input.shape) == 2:
        # Input is already a maximum intensity projection (MIP)
        marker_mip = marker_input

    # Convert nuclei_masks to boolean mask
    nuclei_masks_bool = nuclei_labels.astype(bool)

    # Find nuclei that intersect with the marker signal defined range
    nuclei_and_marker = nuclei_masks_bool & (min_max_range[0] < marker_mip) & (marker_mip <= min_max_range[1])


    # Create a 3D structuring element (cuboid)
    structuring_element = np.ones((erosion_factor, erosion_factor, erosion_factor), dtype=bool)
    eroded_nuclei_and_marker = binary_erosion(nuclei_and_marker, structure=structuring_element)

    # Label the eroded nuclei and marker mask
    labeled_nuclei, num_labels = measure.label(nuclei_labels, return_num=True)

    # Use NumPy's advanced indexing to identify labels that intersect with the eroded marker mask
    intersecting_labels = np.unique(labeled_nuclei[eroded_nuclei_and_marker])
    intersecting_labels = intersecting_labels[intersecting_labels != 0]  # Remove background label

    # Create an empty array for the final labeled nuclei
    processed_region_labels = np.zeros_like(labeled_nuclei, dtype=int)

    # Recover the full extent of nuclei that intersect with the marker mask
    for idx, label in enumerate(intersecting_labels):
        # Recover the entire region of the original nuclei_mask that has this label
        processed_region_labels[labeled_nuclei == label] = label

    return nuclei_and_marker, eroded_nuclei_and_marker, marker_mip, processed_region_labels

def check_filenames(images, rois):

    # Extract the base filenames without extensions using Path.stem
    images_base = [Path(file).stem for file in images]
    rois_base = [Path(file).stem for file in rois]

    # Check for missing files in images list
    missing_in_images = [file for file in rois_base if file not in images_base]
    if missing_in_images:
        for file in missing_in_images:
            print(f"Missing in images list: {file}")
    else:
        print("No files missing in images list.")

    # Check for missing files in rois list
    missing_in_rois = [file for file in images_base if file not in rois_base]
    if missing_in_rois:
        for file in missing_in_rois:
            print(f"Missing in rois list: {file}")
    else:
        print("No files missing in rois list.")

def simulate_cytoplasm(nuclei_labels, dilation_radius=2, erosion_radius=0):

    if erosion_radius >= 1:

        # Erode nuclei_labels to maintain a closed cytoplasmic region when labels are touching (if needed)
        eroded_nuclei_labels = cle.erode_labels(nuclei_labels, radius=erosion_radius)
        eroded_nuclei_labels = cle.pull(eroded_nuclei_labels)
        nuclei_labels = eroded_nuclei_labels

    # Dilate nuclei labels to simulate the surrounding cytoplasm
    cyto_nuclei_labels = cle.dilate_labels(nuclei_labels, radius=dilation_radius)
    cytoplasm = cle.pull(cyto_nuclei_labels)

    # Create a binary mask of the nuclei
    nuclei_mask = nuclei_labels > 0

    # Set the corresponding values in the cyto_nuclei_labels array to zero
    cytoplasm[nuclei_mask] = 0

    return cytoplasm

def simulate_cell(nuclei_labels, dilation_radius=2, erosion_radius=0):

    if erosion_radius >= 1:

        # Erode nuclei_labels to maintain a closed cytoplasmic region when labels are touching (if needed)
        eroded_nuclei_labels = cle.erode_labels(nuclei_labels, radius=erosion_radius)
        eroded_nuclei_labels = cle.pull(eroded_nuclei_labels)
        nuclei_labels = eroded_nuclei_labels

    # Dilate nuclei labels to simulate the surrounding cytoplasm
    cyto_nuclei_labels = cle.dilate_labels(nuclei_labels, radius=dilation_radius)
    cell = cle.pull(cyto_nuclei_labels)

    return cell

def simulate_cytoplasm_chunked_3d(nuclei_labels, dilation_radius=2, erosion_radius=0, chunk_size=(1, 1024, 1024)):
    """
    Simulates cytoplasm expansion around labeled nuclei in a 3D volume using chunked processing.
    Nuclei region is masked out generating a hollow sphere around it.

    Parameters:
    nuclei_labels (ndarray): 3D array of labeled nuclei.
    dilation_radius (int, optional): Radius for dilation of the nuclei. Default is 2.
    erosion_radius (int, optional): Radius for erosion of the nuclei. Default is 0.
    chunk_size (tuple, optional): Size of the chunks to process (z, y, x). Default is (1, 1024, 1024).

    Returns:
    ndarray: 3D array representing the simulated cytoplasm with nuclei regions removed. The values in the returned
             array indicate the cytoplasm regions after dilation, with zero values corresponding to the original 
             nuclei positions, ensuring no overlap.
    """
    cytoplasm = np.zeros_like(nuclei_labels)
    
    # Process the data in chunks to optimize memory usage and allow processing of large datasets
    for z in range(0, nuclei_labels.shape[0], chunk_size[0]):
        for y in range(0, nuclei_labels.shape[1], chunk_size[1]):
            for x in range(0, nuclei_labels.shape[2], chunk_size[2]):
                chunk = nuclei_labels[z:z+chunk_size[0], y:y+chunk_size[1], x:x+chunk_size[2]]
                
                # Apply erosion only if the radius is greater than or equal to 1 to avoid unnecessary processing
                if erosion_radius >= 1:
                    eroded_chunk = cle.erode_labels(chunk, radius=erosion_radius)
                    eroded_chunk = cle.pull(eroded_chunk)
                    chunk = eroded_chunk

                cyto_chunk = cle.dilate_labels(chunk, radius=dilation_radius)
                cyto_chunk = cle.pull(cyto_chunk)

                # Create a binary mask of the nuclei
                chunk_mask = chunk > 0
                # Set the corresponding values in the cyto_chunk array to zero
                cyto_chunk[chunk_mask] = 0

                cytoplasm[z:z+chunk_size[0], y:y+chunk_size[1], x:x+chunk_size[2]] = cyto_chunk
    
    return cytoplasm

def simulate_cell_chunked_3d(nuclei_labels, dilation_radius=2, erosion_radius=0, chunk_size=(1, 1024, 1024)):
    cell = np.zeros_like(nuclei_labels)
    
    for z in range(0, nuclei_labels.shape[0], chunk_size[0]):
        for y in range(0, nuclei_labels.shape[1], chunk_size[1]):
            for x in range(0, nuclei_labels.shape[2], chunk_size[2]):
                chunk = nuclei_labels[z:z+chunk_size[0], y:y+chunk_size[1], x:x+chunk_size[2]]
                
                if erosion_radius >= 1:
                    eroded_chunk = cle.erode_labels(chunk, radius=erosion_radius)
                    eroded_chunk = cle.pull(eroded_chunk)
                    chunk = eroded_chunk

                cell_chunk = cle.dilate_labels(chunk, radius=dilation_radius)
                cell_chunk = cle.pull(cell_chunk)

                cell[z:z+chunk_size[0], y:y+chunk_size[1], x:x+chunk_size[2]] = cell_chunk
    
    return cell

def display_segm_in_napari(directory_path, segmentation_type, model_name, index, slicing_factor_xy, slicing_factor_z, compression_factor, method, images):

    if compression_factor == None:
        compression_factor = 1

    if slicing_factor_xy == None:
        slicing_factor_xy = 1

    # Dinamically generate the results and nuclei_preds the user wants to explore
    results_path = Path("./results") / directory_path.name / segmentation_type / model_name
    nuclei_preds_path = directory_path / "nuclei_preds" / segmentation_type / model_name

    # Load the corresponding BP_populations_marker_+_summary_{method}.csv
    df = pd.read_csv(results_path / f"BP_populations_marker_+_summary_{method}.csv", index_col=0)

    # Extract the value for the filename column at input index
    filename = df.iloc[index]['filename']

    # List of subfolder names
    roi_names = [folder.name for folder in nuclei_preds_path.iterdir() if folder.is_dir()]

    # Read the database containing all populations
    per_label_df = pd.read_csv(results_path / f"BP_populations_marker_+_per_label_{method}.csv")

    # Scan for the corresponding image path
    for image in images:

        # Open that filepath, load the image and display labels for all cell_populations
        if filename in image:

            # Generate maximum intensity projection and extract filename
            img, filename = read_image(image, (slicing_factor_xy * compression_factor), (slicing_factor_z))

            # Show input image in Napari (2D or 3D-stack)
            viewer = napari.Viewer(ndisplay=2)
            if segmentation_type == "3D":
                viewer.add_image(img)
            elif segmentation_type == "2D":
                img_mip = maximum_intensity_projection(img)
                viewer.add_image(img_mip)

            for roi_name in roi_names:

                nuclei_labels = tifffile.imread(nuclei_preds_path / roi_name / f"{filename}.tiff")
                # ... ensures that all preceding dimensions (whether 2D or 3D) are retained without slicing.
                nuclei_labels = nuclei_labels[..., ::compression_factor, ::compression_factor]
                viewer.add_labels(nuclei_labels, name=f"nuclei_{roi_name}")

                # Filter based on ROI and filename
                per_roi_df = per_label_df[per_label_df["ROI"] == roi_name]
                per_filename_df = per_roi_df[per_roi_df["filename"] == filename]

                # Identify cell population columns (those with boolean values)
                cell_pop_cols = per_filename_df.select_dtypes(include=['bool']).columns

                for cell_pop in cell_pop_cols:

                    # Extract the labels corresponding to True values
                    true_labels = per_filename_df[per_filename_df[cell_pop]]['label'].tolist()

                    # Convert to a numpy array for better performance with np.isin()
                    true_labels_array = np.array(true_labels)

                    # Use the true_labels_array with np.isin()
                    mask = np.isin(nuclei_labels, true_labels_array)

                    # Use the mask to set values in 'nuclei_labels' that are not in 'label_values' to 0,
                    # creating a new array 'filtered_labels' with only the specified values retained
                    filtered_labels = np.where(mask, nuclei_labels, 0)

                    # Add the resulting filtered labels to Napari
                    viewer.add_labels(filtered_labels, name=f"{cell_pop}_in_{roi_name}")

def remap_labels(array):
    """Reorganize label indentifiers so they have consecutive ids ranging from 1 to the maximum label of objects"""

    # Get the unique labels (excluding 0)
    unique_labels = np.unique(array[array > 0])

    # Create a mapping from old label to new label
    new_labels = np.arange(1, len(unique_labels) + 1)

    # Create a dictionary for fast label mapping
    label_mapping = dict(zip(unique_labels, new_labels))

    # Use np.vectorize to remap the labels in the array, defaulting to 0 (background) for missing keys
    vectorized_remap = np.vectorize(lambda x: label_mapping.get(x, 0))
    remapped_array = vectorized_remap(array)
    
    return remapped_array