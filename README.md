<h1>Nuclear Segmentation Utility for Brain and Organoid fluorescent samples (NEUROSEG)</h1>

[![License](https://img.shields.io/pypi/l/napari-accelerated-pixel-and-object-classification.svg?color=green)](https://github.com/adiezsanchez/brain_tissue_nuc_segm/blob/main/LICENSE)
[![Development Status](https://img.shields.io/pypi/status/napari-accelerated-pixel-and-object-classification.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)

This repository provides tools in the form of interactive **Jupyter Notebooks** to define cell populations based on the presence or absence of multiple fluorescent markers in multichannel 3D-stacks. The pipeline has been designed to be used with brain and organoid tissue sections but can also work for cell cultures. Analysis can be performed on the whole image by default or on multiple user-defined ROIs. 

## Overview

The pipeline begins with **nuclear segmentation**, performed using either **StarDist** or **Cellpose**, with support for custom-trained models. Based on the predicted **2D/3D nuclei labels**, the researcher can artificially generate (via a dilation operation):

- the **cytoplasmic area/volume**, or  
- the **entire cellular area/volume**

These compartments are used to check for the **presence of fluorescent markers** restricted to those regions.

## Methods for Defining Positive Cells

After segmentation, one of the following **three methods** can be used to determine whether a cell is positive for a marker. Each method has a SP (single-processing) and BP (batch-processing) mode. SP mode allows to explore different images in your dataset, apply the defining method and visualize the results in Napari in order to define your BP settings. BP mode applies the settings defined in SP mode to all images in a folder with no visual feedback in Napari.

### 1. Average Intensity Measurement with Min-Max Thresholding 

- Computes the **average intensity** of each marker within the cellular compartment.
- A cell is considered **positive** if the average intensity falls within the **user-defined min and max thresholds**.
- Choose **`000_SP_Avg_Intensity.ipynb`** & **`003_BP_Avg_Intensity.ipynb`**.

### 2. Pixel Intensity Thresholding with Morphological Operations 

- Generates a **binary mask** for each marker based on a **min-max intensity range**.
- Calculates **colocalization** between the mask and the cellular compartment.
- Applies **morphological erosion** to eliminate small, unspecific regions.
- Remaining colocalizing regions overlapping with the cellular compartment define **positive cells**.
- Choose **`000_SP_Colocalization.ipynb`** & **`003_BP_Colocalization.ipynb`**.

### 3. [APOC](https://github.com/haesleinhuepf/apoc/tree/main) Object Classifier 

- Allows training of an **object classifier** using **signal intensity features**.
- Requires manual annotation of a few sample objects in **Napari** using **`!_APOC_Obj_Class_training.ipynb`**
- Run **`000_SP_Object_Classifier.ipynb`** & **`003_BP_Object_Classifier.ipynb`**.

## Downstream Analysis

Once positive cells are identified:

- Define subpopulations based on **presence/absence of single or multiple markers** using **`004_BP_Data_analysis.ipynb`**
- Visualize segmentation results and subpopulations in **Napari** using  **`005_Napari_segm_viz.ipynb`**

## Input & File Formats

- Accepts **multichannel 3D stacks** (multiple z-slices) from **Nikon (.nd2)** and **Zeiss (.czi)** systems.
- Analysis can be performed on the **3D volume** or a **2D maximum intensity projection**.
- To support additional file formats, modify the `read_image()` function in  **`utils_stardist.py`** to return a **NumPy array** with shape `(ch, z, x, y)`.


<h2>Environment setup instructions</h2>

1. In order to run these Jupyter notebooks and .py scripts you will need to familiarize yourself with the use of Python virtual environments using Mamba. See instructions [here](https://biapol.github.io/blog/mara_lampert/getting_started_with_mambaforge_and_python/readme.html).

2. Then you will need to create a couple of virtual environment using the command below or from the .yml file in the envs folder (see step 3):

    For brain_nuc_stardist:

   <code>mamba create --name brain_nuc_stardist python=3.10 napari pyclesperanto-prototype apoc-backend plotly pyqt nbformat nd2 czifile cudatoolkit=11.2 cudnn=8.1.0 -c conda-forge</code>

   <code>mamba activate brain_nuc_stardist</code>

   <code>pip install "tensorflow<2.11"</code>

   <code>pip install stardist==0.8.5</code>

   <code>pip install gputools==0.2.15</code>

   <code>pip install edt</code>

   <code>pip install reikna==0.8.0</code>

   <code>pip install numpy==1.26.4</code>

   <code>pip install numba==0.59.1</code>


   For brain_nuc_cellpose:

   <code>mamba create -n brain_nuc_cellpose python=3.11 devbio-napari cellpose=3.0.11 pytorch==2.5.0 torchvision==0.20.0 pytorch-cuda=12.1 plotly pyqt python-kaleido nd2 -c conda-forge -c pytorch -c nvidia</code>

3. To recreate the venv from the environment.yml files stored in the envs folder (recommended) navigate into the envs folder using <code>cd</code> in your console and then execute:

   <code>mamba env create -f environment.yml</code>

4. Then launch VS Code to interact with the analysis pipelines.

<h2>Pretrained Stardist Models</h2>

This repository contains a few pretrained models applied by users at NTNU. You can also train your own after annotating a 2D or 3D subset of your dataset (then use the JN under <code>Stardist/Stardist_model_training</code>). For annotation instructions see the [Stardist repository](https://github.com/stardist/stardist). Pre-trained models with an sf_ suffix have been trained using downsampled input data in x and y, i.e. sf_None is trained using the full resolution images but sf_2 and sf_4 are downsampled by a factor if 2 and 4 respectively. This allows quicker nuclei label predictions in high resolution images. MEC 0.1 is the default model applied to the test_data you can obtain by [contacting me](mailto:alberto.d.sanchez@ntnu.no).

<h4>Hippocampus 1.0</h4>

Trained on images of mouse hippocampi from tissue sections of 5 and 30 µm. Images acquired on a Zeiss LSM880 system using a Plan-Apo 40X/1.4 Oil DIC M27 objective. Scaling per Pixel (x, y, z): 0.35µm x 0.35µm x 0.9µm. For detailed acquisition metadata see raw_data/test_data. Stardist3D model meant to be used with an image stack (multiple z-slices). Model rescued from this [repo](https://github.com/jvgrini/nuclei).

<h4>MEC 0.1</h4>

Trained on the same images as Hippocampus 1.0 in addition to images of mice entorhinal cortex. Images acquired on a Zeiss LSM880 system using a Plan-Apo 40X/1.4 Oil DIC M27 objective. Scaling per Pixel (x, y, z): 0.35µm x 0.35µm x 0.9µm. For detailed acquisition metadata see raw_data/test_data. Stardist3D model meant to be used with an image stack (multiple z-slices). Model rescued from this [repo](https://github.com/jvgrini/nuclei).

<h4>3D_org_nihanseb</h4>

Trained on images of human iPSC-derived brain organoid 18 µm sections. Images acquired on a Nikon Ti2 Crestoptics V3 spinning disk using a PLAN APO λD 40x OFN25 DIC N2 air objective. Scaling per Pixel (x, y, z): 0.166µm x 0.166µm x 1µm. For detailed acquisition metadata see <code>training_data/organoid_Nikon_nihanseb/MLD_1.8_block4_ARSA_MBP_batch_1_40x_ROI_1.nd2</code>. Stardist3D model meant to be used with an image stack (multiple z-slices).

<h4>3D_brain_Nikon_dagnysd_0.5</h4>

Trained on images of mouse hippocampi from 30 µm tissue sections. Images acquired on a Nikon Ti2 Crestoptics V3 spinning disk using a PLAN APO λD 40x OFN25 DIC N2 air objective. Scaling per Pixel (x, y, z): 0.166µm x 0.166µm x 0.5µm. For detailed acquisition metadata see <code>training_data/thick_brain_Nikon_dagnysd_0.5/A2_Brain4_C_TR1_ROI_1.nd2</code>. Stardist3D model meant to be used with an image stack (multiple z-slices). 