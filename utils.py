from cellpose import models
import numpy as np
from skimage import exposure, filters, measure
from scipy.ndimage import binary_erosion

# Define the Cellpose model that will be used
model = models.Cellpose(gpu=True, model_type="nuclei")

def segment_nuclei_2d (nuclei_stack, gaussian_sigma = 0, cellpose_nuclei_diameter = None):

    # Perform maximum intensity projections
    nuclei_mip = np.max(nuclei_stack, axis=0)

    # Might need to perform a Gaussian-blur before
    post_gaussian_img = filters.gaussian(
        nuclei_mip, sigma=gaussian_sigma
    )

    # Apply Contrast Stretching to improve Cellpose detection of overly bright nuclei
    p2, p98 = np.percentile(post_gaussian_img, (2, 98))
    img_rescale = exposure.rescale_intensity(
        post_gaussian_img, in_range=(p2, p98)
    )

    # Predict nuclei nuclei_masks using cellpose
    nuclei_labels, flows, styles, diams = model.eval(
        img_rescale,
        diameter=cellpose_nuclei_diameter,
        channels=[0, 0],
        net_avg=False,
    )

    return nuclei_mip, nuclei_labels

def segment_marker_positive_nuclei (nuclei_labels, marker_stack, marker_channel_threshold, erosion_factor):

    # Perform maximum intensity projections
    marker_mip = np.max(marker_stack, axis=0)

    # Convert nuclei_masks to boolean mask
    nuclei_masks_bool = nuclei_labels.astype(bool)

    # Find nuclei that intersect with the marker signal
    nuclei_and_marker = nuclei_masks_bool & (marker_mip > marker_channel_threshold)

    # Erode the result to remove small objects
    structuring_element = np.ones((erosion_factor, erosion_factor), dtype=bool)
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

    return marker_mip, processed_region_labels