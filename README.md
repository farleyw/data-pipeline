# ifcbUTOPIA Data Pipeline  

This repository contains a collection of notebooks to streamline the process of converting raw IFCB data to CNN-labeled and validated datasets.

## Setup

### Mac/Linux

_data-pipeline repository_  

1. Clone this repository to your local machine (instructions [here](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)).

_Create a new python environment_  

1. Use anaconda or miniconda to set up a new python environment.
2. In your prompt, navigate to the folder containing this repo's `environment.yaml` file.
3. Type `conda env create -f environment.yaml` and wait for the prompt to finish setting up the environment. It may take a while. 
4. The environment's name is "ifcb-utopia", so activate it with `conda activate ifcb-utopia`. 

### Windows

_data-pipeline repository_  

1. Clone this repository to your local machine (instructions [here](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)).

_Create a new python environment_  

1. Use anaconda or miniconda to set up a new python environment.
2. Open this repo's `environment.yaml` file in a text editor.
3. Comment out the tensorflow and utopia-pipeline-tools lines by typing # at the start of the line.
4. In your prompt, navigate to the folder containing this `environment.yaml` file.
5. Type `conda env create -f environment.yaml` and wait for the prompt to finish setting up the environment. It may take a while.
6. Activate the environment with `conda activate ifcb-utopia`.
6. Install tensorflow with `pip install tensorflow==2.13.1`.
7. Install utopia-pipeline-tools with `pip install utopia-pipeline-tools`.   

### Troubleshooting

- Make sure your pip is up to date (`pip install --upgrade pip`)

## Getting updates

- Sync any repository (notebook) updates to your local machine with `git pull`.   
- Get the latest version of utopia-pipeline-tools by activating the ifcb-utopia environment then typing `pip install --upgrade utopia-pipeline-tools`.  

## Running the notebooks

This pipeline uses [marimo](https://marimo.io/) notebooks. Open the notebook interface in the activated environment by typing `marimo edit`.  