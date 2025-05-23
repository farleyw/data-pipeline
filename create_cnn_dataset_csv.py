import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Creating the CNN Dataset .csv File From Images on the Azure Blob""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    __Steps for generating the .csv file of image filepaths, classification probabilities, and predicted labels__  

    1. Select whether your images are on the Azure blob or in a local folder. If stored locally, select the `ml` folder with the file browser and input the name of the dataset in the text box (no spaces). If the images are stored in the blob, select which container you want to classify from the drop-down menu.
    2. If you already have a csv file containing a 'filepath' column of all the IFCB image filepaths (e.g. `ml/{sample_ID}/{image_ID}.png`), select the checkbox and select the file in the corresponding file browser. 
    3. Otherwise, Click the "Retrieve filepaths" button to get the image filepaths from the blob storage or local folder. This will take a few minutes for smaller datasets (500,000 images) or about 70 minutes for larger datasets (10,000,000 images). The code automatically saves two csv files to your local machine - the list of image filepaths used during the classification process and the list of sample metadata csv filepaths, which can be used later to speed up the process of making SeaBASS files. 
    4. Click the "Predict labels" button to classify the images with the CNN. This will take about 4 hours per 1,000,000 images. I recommend running this portion overnight. If the dataset is particularly large, this may require a VM since a personal computer's working memory can be too small to handle massive datasets. For instance, after 10 hours of runtime on the taraeuropa23 dataset (almost 10,000,000 images), my computer gave me a memory error. 
    5. If no errors occur, a new .csv file has been saved to your local machine!
    """
    )
    return


@app.cell
def _():
    # imports
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import os

    from azure.storage.blob import ContainerClient

    import cv2

    import tensorflow as tf
    from tensorflow import keras
    from IPython.display import display
    import global_val_setup as gvs
    from utopia_pipeline_tools.azure_blob_tools import list_containers_in_blob, list_files_in_blob
    from utopia_pipeline_tools.ifcb_data_tools import retrieve_filepaths_from_local
    from utopia_pipeline_tools.cnn_tools import load_local_model, preprocess_input
    return (
        ContainerClient,
        cv2,
        display,
        gvs,
        list_containers_in_blob,
        list_files_in_blob,
        load_local_model,
        np,
        os,
        pd,
        preprocess_input,
        tf,
    )


@app.cell
def _(mo):
    mo.md(r"""## Setup""")
    return


@app.cell
def _(mo):
    # form to select whether your model is local or on the cloud
    folder_location_selectbox = mo.ui.dropdown(options=['local', 'cloud'])

    mo.md(f"""__Select `ml` folder location:__   
    {folder_location_selectbox}""")
    return (folder_location_selectbox,)


@app.cell
def _(display, folder_location_selectbox, mo):
    if folder_location_selectbox.value == 'local':
        filepath_form = mo.ui.file_browser(initial_path='Z:/projects/gnats/gnats-post2024/2025/', selection_mode='directory', multiple=False, label="Select folder...")
        display(mo.md(f"""__Select your root folder:__  
        This should be the `ml` folder that contains folders of images.  
        {filepath_form}"""))
    return (filepath_form,)


@app.cell
def _(display, folder_location_selectbox, mo):
    if folder_location_selectbox.value == 'local':
        dataset_name_form = mo.ui.text().form()
        display(mo.md(f"""__Enter the name of the dataset:__  
        This will be used to name the final `.csv` file.  
        {dataset_name_form}"""))
    return (dataset_name_form,)


@app.cell
def _(display, folder_location_selectbox, gvs, list_containers_in_blob, mo):
    if folder_location_selectbox.value == 'cloud':
        cstr = gvs.config_info['connection_string']
        # retrieve list of blob containers
        blob_containers = list_containers_in_blob(connection_string=cstr)

        # make a drop-down form to select the container
        container_form = mo.ui.dropdown(blob_containers)

        display(mo.md(f"""__Select your blob container:__  
        {container_form}"""))
    return container_form, cstr


@app.cell
def _(mo):
    csv_filepath_checkbox = mo.ui.checkbox(label='Using existing csv file of image filepaths', value=False)

    mo.md(f"{csv_filepath_checkbox}")
    return (csv_filepath_checkbox,)


@app.cell
def _(csv_filepath_checkbox, display, mo):
    if csv_filepath_checkbox.value:
        filepath_csv_form = mo.ui.file_browser(initial_path='/', selection_mode='file', 
                                               multiple=False, filetypes=['.csv'],
                                               label='Select image filepath csv:')
        display(mo.md(f"{filepath_csv_form}"))
    return (filepath_csv_form,)


@app.cell
def _(
    container_form,
    cstr,
    csv_filepath_checkbox,
    filepath_csv_form,
    filepath_form,
    folder_location_selectbox,
):
    # setting the run conditions for all buttons and code that requires information from the container and connection string forms
    try: 
        if folder_location_selectbox.value == 'local':
            if filepath_form.value is not None:
                if csv_filepath_checkbox.value is True:
                    if filepath_csv_form.path(0) is not None:
                        run_condition = True
                    else:
                        run_condition = False
                else:
                    run_condition = True
            else:
                run_condition = False
        elif folder_location_selectbox.value == 'cloud':
            if csv_filepath_checkbox.value is True:
                if container_form.value is not None and cstr is not None and filepath_csv_form.path(0) is not None: 
                # container and connection string must both be defined
                    run_condition = True
                else:
                    run_condition = False
            else:
                if container_form.value is not None and cstr is not None:
                    run_condition = True
        else:
            run_condition = False
    except:
        run_condition = False
    return (run_condition,)


@app.cell(hide_code=True)
def _(csv_filepath_checkbox, display, mo):
    if csv_filepath_checkbox.value is False:
        display(mo.md(r"""## Retrieve Filepaths from the Azure Blob"""))
    return


@app.cell
def _(csv_filepath_checkbox, display, mo, run_condition):
    if csv_filepath_checkbox.value is False:
        # button to run the cells that retrieve filepaths from the blob and organize them in a dataframe
        filepath_retrieval_button = mo.ui.run_button(label="Retrieve filepaths", 
                                                     disabled=bool(run_condition is False))
        display(filepath_retrieval_button)
    return (filepath_retrieval_button,)


@app.cell
def _(
    container_form,
    csv_filepath_checkbox,
    display,
    filepath_csv_form,
    filepath_form,
    filepath_retrieval_button,
    folder_location_selectbox,
    gvs,
    list_files_in_blob,
    os,
    pd,
):
    if csv_filepath_checkbox.value is False:
        if filepath_retrieval_button.value:
            # call the blob to list all the files in the container
            if folder_location_selectbox.value == 'cloud':
                all_df = list_files_in_blob(container=container_form.value, 
                                            connection_string=gvs.config_info['connection_string'],
                                            selection='all')
                png_df = pd.DataFrame({'filepath': [x for x in all_df['filepath'] if '.png' in x]})
                csv_df = pd.DataFrame({'filepath': [x for x in all_df['filepath'] if '.csv' in x]})


            #From Farley:
            elif folder_location_selectbox.value == 'local':
                folder_location = filepath_form.path(0)
                all_list = []

                for root, dirs, files in os.walk(folder_location):
                    for file in files:
                        if file.endswith(".png"):
                            file_path = os.path.join(root, file)

                        if os.path.getsize(file_path) > 2048:  # File size greater than 2KB
                            all_list.append(file_path)
            #Original
           # elif folder_location_selectbox.value == 'local':
            #    all_list = retrieve_filepaths_from_local(filepath_form.path(0))

                png_df = pd.DataFrame({'filepath': [x for x in all_list if '.png' in x]})
                csv_df = pd.DataFrame({'filepath': [x for x in all_list if '.csv' in x]})
    #Farley comment out next 2 lines
           # png_df.to_csv(f"{container_form.value}_image_filepaths.csv")
           # csv_df.to_csv(f"{container_form.value}_sample_metadata_filepaths.csv")

            display(png_df.head())
    elif filepath_csv_form.path(0) is not None:
        png_df = pd.read_csv(filepath_csv_form.path(0))
    return folder_location, png_df


@app.cell
def _(display, png_df, run_condition):
    if run_condition is True:
        display(png_df['filepath'].head())
    return


@app.cell
def _(mo):
    mo.md(r"""## Classify with the CNN""")
    return


@app.cell
def _(mo):
    # form to select whether your model is local or on the cloud
    model_location_selectbox = mo.ui.dropdown(options=['local', 'cloud'])

    mo.md(f"""__Select model location:__  
    {model_location_selectbox}""")
    return (model_location_selectbox,)


@app.cell
def _(mo):
    mo.md(
        r"""
    __Current local model file locations:__  
    assets/models/model-cnn-v1-b3.h5  
    assets/models/model-cnn-v1-b3.json  

    __Current cloud model connection information:__  
    subscription_id: 91804dbe-1fd2-4384-8b66-2b5e4ad1f2f2  
    resource_group: UTOPIA  
    workspace_name: pivot  
    experiment_name: adt-pivot  
    api_key: 4B2DTMyhNHgIk3lJtt8MSdBU5QodpCNf  
    model_name: basemodel  
    endpoint_name: basemodel-endpoint  
    deployment_name: pivot-basemodel
    """
    )
    return


@app.cell
def _(display, gvs, mo, model_location_selectbox):
    if model_location_selectbox.value == 'local':
        json_filepath = mo.ui.text(label='.json filepath').form()
        h5_filepath = mo.ui.text(label='.h5 filepath').form()

        display(mo.md(f"""__Enter the filepath locations of the .json and .h5 model files:__   
                {json_filepath}   
                {h5_filepath}
                """))

    elif model_location_selectbox.value == 'cloud':
        sub_id = gvs.config_info['subscription_id']
        resource_group = gvs.config_info['resource_group']
        workspace_name = gvs.config_info['workspace_name']
        experiment_name = gvs.config_info['experiment_name']
        api_key = gvs.config_info['api_key']
        model_name = gvs.config_info['model_name']
        endpoint_name = gvs.config_info['endpoint_name']
        deployment_name = gvs.config_info['deployment_name']

        display(mo.md(f"""__Review the cloud locations and configuration information of the CNN:__   
                {sub_id}  
                {resource_group}  
                {workspace_name}  
                {experiment_name}  
                {api_key}  
                {model_name}  
                {endpoint_name}  
                {deployment_name}  
                __NOTE: CLOUD MODEL NOT YET SET UP FOR CLASSIFICATION__
                """))

    else:
        print("ACTION REQUIRED: Please select the model location")
    return (
        api_key,
        deployment_name,
        endpoint_name,
        experiment_name,
        h5_filepath,
        json_filepath,
        model_name,
        resource_group,
        sub_id,
        workspace_name,
    )


@app.cell
def _(
    api_key,
    deployment_name,
    endpoint_name,
    experiment_name,
    h5_filepath,
    json_filepath,
    model_location_selectbox,
    model_name,
    resource_group,
    sub_id,
    workspace_name,
):
    if model_location_selectbox.value == 'local':
        if json_filepath.value == '' or h5_filepath == '':
            model_entry = False
        else:
            model_entry = True
    elif model_location_selectbox.value == 'cloud':
        if (sub_id.value == '' or resource_group.value == '' or workspace_name.value == '' or 
            experiment_name.value == '' or api_key.value == '' or model_name.value == '' or 
            endpoint_name.value == '' or deployment_name.value == ''):
            model_entry = False
        else:
            model_entry = True
    else:
        model_entry = False
    return (model_entry,)


@app.cell
def _(mo, model_entry, run_condition):
    # button to run the images through the CNN to predict labels
    prediction_button = mo.ui.run_button(label="Predict labels", 
                                         disabled=bool(run_condition is False or model_entry is False))
    prediction_button
    return (prediction_button,)


@app.cell
def _(
    ContainerClient,
    container_form,
    cstr,
    cv2,
    filepath_retrieval_button,
    folder_location_selectbox,
    h5_filepath,
    json_filepath,
    load_local_model,
    np,
    pd,
    png_df,
    prediction_button,
    preprocess_input,
    tf,
):
    if prediction_button.value:
        # connecting to the specific container
        if filepath_retrieval_button.value == 'cloud':
            container_client = ContainerClient.from_connection_string(conn_str=cstr,
                                                                      container_name=container_form.value)
        # load the model from the local files
        v1_b3_model = load_local_model(json_filepath.value, h5_filepath.value)

        # set up lists, counters, and splits
        data_index = np.arange(len(png_df.index))
        if len(png_df.index) < 500000:
            n_splits = int(len(png_df.index)/100)
        else:
            n_splits = int(len(png_df.index)/100000)
        data_split = np.array_split(data_index, n_splits)
        test_preds = []
        error_list = []
        counter = 0

        # looping through 10 subsets of the data
        for subset in data_split: 
            # set up lists
            image_list = []
            filepaths = []

            # looping through each image in the subset
            for i in subset: 
                # select the row in the dataset and get the filepath
                input_path = png_df.iloc[i]['filepath'] 

                if folder_location_selectbox.value == 'cloud':
                    # call the image from the blob
                    image_download = container_client.download_blob(blob=input_path)
                    image = cv2.imdecode(np.frombuffer(image_download.readall(), dtype=np.uint8),
                                         cv2.IMREAD_UNCHANGED)
                elif folder_location_selectbox.value == 'local':
                    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

                # preprocess the image
                try:
                    processed = np.array(preprocess_input(image))
                    image_list.append(processed)
                    filepaths.append(input_path)
                except:
                    error_list.append((counter, i))


            # predict the labels for all images in the subset

            pred_input = np.array(image_list)
                    #Farley- next two lines
            pred_input_grey = tf.image.rgb_to_grayscale(pred_input)
            probabilities = v1_b3_model.predict(pred_input_grey)
            #Original below 1 line:
            #probabilities = v1_b3_model.predict(pred_input)

            # make a dataframe of the probabilities, filepaths, and predicted labels
            pred_frame = pd.DataFrame(probabilities)
            pred_frame['filepath'] = filepaths
            pred_frame['pred_label'] = [np.argmax(x) for x in probabilities]

            # append the dataframe to the list
            test_preds.append(pred_frame)

            counter += 1 
            print('completed ', str(counter), ' of', n_splits, ' testing subsets')

            del image_list
            del filepaths
            del pred_input
            del probabilities
            del pred_frame


        # concatenate all of the subset dataframes
        test_eval = pd.concat(test_preds)
    return (test_eval,)


@app.cell
def _(
    container_form,
    dataset_name_form,
    display,
    folder_location,
    folder_location_selectbox,
    test_eval,
):
    # save the final dataframe as a csv to the local machine
    if folder_location_selectbox.value == 'cloud':
        test_eval.to_csv(f"{container_form.value}_prediction_results.csv")
        display(test_eval.head())
    elif folder_location_selectbox.value == 'local':
        if dataset_name_form.value is None:
            print("ACTION REQUIRED: Enter dataset name")
        else:
            test_eval.to_csv(f"{folder_location}\\{dataset_name_form.value}_prediction_results.csv")
            display(test_eval.head())
    return


@app.cell
def _(dataset_name_form, folder_location, os):
    print(f"{folder_location}{dataset_name_form.value}_prediction_results.csv")
    print(os.getcwd())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
