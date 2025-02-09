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
from skimage import measure
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

def segment_nuclei(nuclei_img, segmentation_type, model, n_tiles=None):

    if segmentation_type == "2D":
        # Ignore the z-dimension of the n_tiles tuple (x, y, z)
        n_tiles = n_tiles[-2:]
    
    normalized = normalize(nuclei_img)

    nuclei_labels, _ = model.predict_instances(normalized, n_tiles=n_tiles, show_tile_progress=True)

    return nuclei_labels

def extract_contour(roi: np.ndarray) -> np.ndarray:
    """
    Extracts the contour of a binary ROI image and returns a binary mask 
    where the contour pixels are set to 1.

    Parameters:
    -----------
    roi : np.ndarray
        A 2D NumPy array of dtype int8 representing the binary region of interest (ROI),
        where nonzero values indicate the foreground.

    Returns:
    --------
    np.ndarray
        A binary mask of the same shape as `roi`, where contour pixels are set to 1,
        and all other pixels are 0.

    Notes:
    ------
    - Uses `skimage.measure.find_contours` to detect continuous contour points.
    - Vectorized operations are used for efficiency.
    - The function assumes that `roi` is a binary image (values of 0 or 1).

    Example:
    --------
    >>> import numpy as np
    >>> from skimage.draw import disk
    >>> roi = np.zeros((100, 100), dtype=np.int8)
    >>> rr, cc = disk((50, 50), 20)
    >>> roi[rr, cc] = 1
    >>> contour_mask = extract_contour(roi)
    >>> print(contour_mask.sum())  # Nonzero values represent contour pixels
    """
    # Find contours, output is a list of (N, 2) arrays representing continuous (x, y) coordinates of contour points
    contours = measure.find_contours(roi, level=0.5)

    # Concatenate all contour points
    all_contours = np.vstack(contours)  # Shape: (N, 2)

    # Round to integer pixel indices
    all_contours = np.round(all_contours).astype(int)

    # Clip to ensure indices are within image bounds
    all_contours[:, 0] = np.clip(all_contours[:, 0], 0, roi.shape[0] - 1)
    all_contours[:, 1] = np.clip(all_contours[:, 1], 0, roi.shape[1] - 1)

    # Create an empty binary mask
    contour_mask = np.zeros_like(roi, dtype=np.uint8)

    # Set pixels at contour locations to 1 using NumPy advanced indexing
    contour_mask[all_contours[:, 0], all_contours[:, 1]] = 1
    
    return contour_mask

def remove_labels_touching_roi_edge(labels, roi):
    """
    Removes labels that are touching the edges of a binary region of interest (ROI) in a 2D or 3D array containing labels.

    Parameters:
    -----------
    labels : np.ndarray
        A 2D or 3D NumPy array of labeled structures (i.e. nuclei), where each unique integer value represents a different label
        and `0` represents the background.
        
    roi : np.ndarray
        A 2D NumPy array representing the binary region of interest (ROI), where nonzero values indicate the foreground.
        The function will either generate a border contour if `roi` is filled entirely with ones, or use the ROI to
        generate contours of the region.

    Returns:
    --------
    np.ndarray
        A 2D or 3D array with the labels that are touching the edges of the ROI removed (set to 0).

    Notes:
    ------
    - The function generates a contour for the ROI and uses it to identify labels that intersect with the ROI's contour.
    - For 3D images, the function will extend the 2D contour mask across the entire stack of slices.
    - The function assumes that `labels` is a labeled image and `roi` is a binary mask image (values of 0 or 1).
    - This function relabels the remaining labels in the output using `skimage.measure.label()` to ensure continuous labeling.

    Example:
    --------
    >>> labels = np.array([[1, 1, 0], [2, 1, 0], [2, 0, 0]])
    >>> roi = np.array([[0, 1, 1], [0, 1, 0], [0, 0, 0]])
    >>> filtered_labels = remove_labels_touching_roi_edge(labels, roi)
    >>> print(filtered_labels)
    """
    # Check if roi covers the entire image (all values stored in roi == 1):
    if np.all(roi == 1):
        #  Generate a contour that covers the border of the image
        contour_mask = np.zeros(roi.shape, dtype=np.int8)

        # Set the outer border to 1 
        contour_mask[0, :] = 1  # Top edge
        contour_mask[-1, :] = 1  # Bottom edge
        contour_mask[:, 0] = 1  # Left edge
        contour_mask[:, -1] = 1  # Right edge

    # Otherwise extract the contour of a user-defined roi:
    else:
        contour_mask = extract_contour(roi)

    # 3D segmentation logic, extend 2D mask across the entire stack volume
    if len(labels.shape) == 3:
        # Extract the number of z-slices to extend the mask into a 3D-volume
        slice_nr = labels.shape[0]

        # Extend the mask across the entire volume
        contour_mask = np.tile(contour_mask, (slice_nr, 1, 1))

    # Convert contour_mask to boolean inplace
    contour_mask = contour_mask.view(bool)  # ✅ Avoid extra copy

    # Identify labels that intersect with the ROI contour
    intersecting_labels = np.unique(labels[contour_mask])
    intersecting_labels = intersecting_labels[intersecting_labels != 0]  # Remove background label

    # Remove intersecting labels inplace
    np.putmask(labels, np.isin(labels, intersecting_labels), 0)  # ✅ Modify `labels` inplace
    del intersecting_labels  # ✅ Free up memory immediately

    # Relabel labels inplace (relabel modifies array in-place when return is assigned)
    labels = measure.label(labels, connectivity=1)

    return labels

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

def segment_marker_positive_labels (labels, marker_input, min_max_range, erosion_factor, segmentation_type):

    if segmentation_type == "2D":
        # Perform maximum intensity projection from the stack
        marker = np.max(marker_input, axis=0)
        # Create a 2D structuring element (square)
        structuring_element = np.ones((erosion_factor, erosion_factor), dtype=bool)

    elif segmentation_type == "3D":
        # Marker stack remains as a 3D-array with original intensities
        marker = marker_input
        # Create a 3D structuring element (cuboid)
        structuring_element = np.ones((erosion_factor, erosion_factor, erosion_factor), dtype=bool)

    # Convert label_masks to boolean mask
    label_masks_bool = labels.astype(bool)

    # Find labels that intersect with the marker signal defined range
    label_and_marker = label_masks_bool & (min_max_range[0] < marker) & (marker <= min_max_range[1])

    # Perform erosion with structuring element (2D or 3D)
    eroded_label_and_marker = binary_erosion(label_and_marker, structure=structuring_element)

    # Use NumPy's advanced indexing to identify labels that intersect with the eroded marker mask
    intersecting_labels = np.unique(labels[eroded_label_and_marker])
    intersecting_labels = intersecting_labels[intersecting_labels != 0]  # Remove background label

    # Create an empty array for the final labeled labels
    processed_region_labels = np.zeros_like(labels, dtype=int)

    # Recover the full extent of input labels that intersect with the marker mask
    for idx, label in enumerate(intersecting_labels):
        # Recover the entire region of the original label_mask that has this label
        processed_region_labels[labels == label] = label

    return label_and_marker, eroded_label_and_marker, marker, processed_region_labels

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