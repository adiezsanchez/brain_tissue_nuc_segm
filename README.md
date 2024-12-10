<h1>Nuclear Segmentation Utility for Brain and Organoid fluorescent samples (NEUROSEG)</h1>

[![License](https://img.shields.io/pypi/l/napari-accelerated-pixel-and-object-classification.svg?color=green)](https://github.com/adiezsanchez/brain_tissue_nuc_segm/blob/main/LICENSE)
[![Development Status](https://img.shields.io/pypi/status/napari-accelerated-pixel-and-object-classification.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)

Segmentation of nuclear signals in fluorescently stained mouse brain tissue sections or organoids imaged with Zeiss or Nikon systems. Nuclei segmentations performed with a fine-tuned Stardist 3D model and/or standard Cellpose 2D nuclei model.

<h2>Environment setup instructions</h2>

1. In order to run these Jupyter notebooks and .py scripts you will need to familiarize yourself with the use of Python virtual environments using Mamba. See instructions [here](https://biapol.github.io/blog/mara_lampert/getting_started_with_mambaforge_and_python/readme.html).

2. Then you will need to create a couple of virtual environment using the command below or from the .yml file in the envs folder (recommended, see step 3):

    For brain_nuc_stardist:

   <code>mamba create --name brain_nuc_stardist python=3.9 devbio-napari csbdeep stardist plotly pyqt nd2 cudatoolkit=11.2 cudnn=8.1.0 -c conda-forge</code>
   <code>mamba activate brain_nuc_stardist</code>
   <code>pip install "tensorflow<2.11"</code>

   For brain_nuc_cellpose:
   <code>mamba create -n brain_nuc_cellpose python=3.11 devbio-napari cellpose=3.0.11 pytorch==2.5.0 torchvision==0.20.0 pytorch-cuda=12.1 plotly pyqt python-kaleido nd2 -c conda-forge -c pytorch -c nvidia</code>

3. To recreate the venv from the environment.yml files stored in the envs folder (recommended) navigate into the envs folder using <code>cd</code> in your console and then execute:

   <code>mamba env create -f environment.yml</code>

4. Then launch VS Code to interact with the analysis pipelines.