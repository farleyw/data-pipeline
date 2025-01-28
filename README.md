# ifcbUTOPIA Data Pipeline  

This repository contains a collection of notebooks to streamline the process of converting raw IFCB data to CNN-labeled and validated datasets.

## Setup

### data-pipeline repository  

1. Clone this repository to your local machine (instructions [here](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)).  

### Create a new python environment

_Mac/Linux_ 

1. Use anaconda or miniconda to set up a new python environment.
2. In your prompt, navigate to the folder containing this repo's `environment.yaml` file.
3. Type `conda env create -f environment.yaml` and wait for the prompt to finish setting up the environment. It may take a while. 
4. The environment's name is "ifcb-utopia", so activate it with `conda activate ifcb-utopia`. 

_Windows_ 

1. Use anaconda or miniconda to set up a new python environment.
2. Open this repo's `environment.yaml` file in a text editor.
3. Comment out the tensorflow and utopia-pipeline-tools lines by typing # at the start of the line.
4. In your prompt, navigate to the folder containing this `environment.yaml` file.
5. Type `conda env create -f environment.yaml` and wait for the prompt to finish setting up the environment. It may take a while.
6. Activate the environment with `conda activate ifcb-utopia`.
6. Install tensorflow with `pip install tensorflow==2.13.1`.
7. Install utopia-pipeline-tools with `pip install utopia-pipeline-tools`.   

### Update your global values

1. Open `global_val_setup.py` and input your configuration information in the `config_info` dictionary and investigator information in `default_investigators`. 
2. If you want to use the Azure blob, make sure you include values for `"blob_storage_name"` and `"connection_string"` (e.g. `"blob_storage_name": "ifcbblob"`). 
3. To connect to an Azure SQL server and database, enter values for `"server"`, `"database"`, `"db_user"` (username), and `"db_password"`.
4. The CNN notebook can call a local or cloud keras model. Include values for `"subscription_id"`, `"resource_group"`, `"workspace_name"`, `"experiment_name"`, `"api_key"`, `"model_name"`, `"endpoint_name"`, and `"deployment_name"` if you want to use a cloud model. 
5. SeaBASS files require information about the researchers involved in the project. Add names, organizations, and emails to the `default_investigators` dictionary with the format: `"Firstname_Lastname": ["Organization", "email@org.com"]` for each individual.  

### Troubleshooting

- Make sure your pip is up to date (`pip install --upgrade pip`)
- If all else fails, you can brute-force create a new environment and individually install each package (this is not the best method, but it does allow for more modular troubleshooting). 
- Contact the utopia-pipeline-tools owner if its setup requirements are making your computer freak out. 

## Getting updates

- Sync any repository (notebook) updates to your local machine with `git pull`.   
- Get the latest version of utopia-pipeline-tools by activating the ifcb-utopia environment then typing `pip install --upgrade utopia-pipeline-tools`.  

## Running the notebooks

This pipeline uses [marimo](https://marimo.io/) notebooks. Open the notebook interface in the activated environment by typing `marimo edit`.  