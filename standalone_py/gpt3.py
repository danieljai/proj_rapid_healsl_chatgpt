import os
import pandas as pd
import numpy as np
from openai import OpenAI
import datetime
import pytz
import logging

import textwrap
import json
import re

# Global Variables
TIMEZONE = pytz.timezone('America/Toronto')
TEMP_DIR = "tmp\gpt"
DATA_DIR = "tmp"
OUTPUT_DIR = "tmp\output"
INCLUDE_SAMPLING = True
script_dir = os.path.dirname(__file__)

# #############################################################################################
# STAGE 1 Parameters
# #############################################################################################

# File Settings
s1_out_csv_filename = "s1_output_full.csv"
s1_out_sample_csv_filename = "s1_output_sample.csv"

s1_in_dir = os.path.join(script_dir, DATA_DIR)
s1_out_full_csv = os.path.join(script_dir, TEMP_DIR, s1_out_csv_filename)
s1_out_smpl_csv = os.path.join(script_dir, TEMP_DIR, s1_out_sample_csv_filename)

# Sampling Settings
# How many samples and how many times to repeat each sample.
N_SAMPLES = 100
N_REPEAT_RESPONSES = 10

# #############################################################################################
# STAGE 2 Parameters
# #############################################################################################

# Demo mode artificially limits the number of rows from the import dataset for testing/demo purposes
DEMO_MODE = False
DEMO_RANDOM = False
DEMO_SIZE_LIMIT = 5

# Discard columns not recognized by the script before saving output file.
# Set to True to retain only essential columns in the output file.
# Set to False to retain all columns similar to the input file.
DROP_EXCESS_COLUMNS = False

# Number of API calls between saving the output file
SAVE_FREQ = 3

# Models Settings
# GPT4 = "gpt-4-0613"
# GPT3 = "gpt-3.5-turbo-0125"
models = {
    "gpt4": "gpt-4-0613",
    "gpt3": "gpt-3.5-turbo-0125"
}
SELECTED_MODEL = "gpt3"
MODEL_NAME = models[SELECTED_MODEL]
TEMPERATURE = 0
LOGPROBS = True
MAX_TOKENS = 30

SYS_PROMPT = """You are a physician with expertise in determining underlying causes of death in Sierra Leone by assigning the most probable ICD-10 code for each death using verbal autopsy narratives. Return only the ICD-10 code without description. E.g. A00 
If there are multiple ICD-10 codes, show one code per line."""

USR_PROMPT = """Determine the underlying cause of death and provide the most probable ICD-10 code for a verbal autopsy narrative of a AGE_VALUE_DEATH AGE_UNIT_DEATH old SEX_COD death in Sierra Leone: {open_narrative}"""

# File Settings
s2_in_full_csv = s1_out_full_csv
s2_in_smpl_csv = s1_out_smpl_csv

s2_out_full_processed_json = os.path.join(script_dir, TEMP_DIR, f"s2_output_{SELECTED_MODEL}_full.json")
s2_out_smpl_processed_json = os.path.join(script_dir, TEMP_DIR, f"s2_output_{SELECTED_MODEL}_sample.json")

# #############################################################################################
# STAGE 3 Parameters
# #############################################################################################

# File Settings
s3_in_full_json = s2_out_full_processed_json
s3_in_smpl_json = s2_out_smpl_processed_json

s3_out_full_csv = os.path.join(script_dir, OUTPUT_DIR, f"s3_output_{SELECTED_MODEL}_full.csv")
s3_out_smpl_csv = os.path.join(script_dir, OUTPUT_DIR, f"s3_output_{SELECTED_MODEL}_sample.csv")

# Data Analysis Settings
PAIRS = 5           # Generate up to 5 ICDs

# Output Settings
DROP_EXCESS_COLUMNS = False         # Set True to remove 'other_columns' from output dataframe
DROP_RAW = False                    # Set True to remove the 'raw' column, the original raw response, from the export file

# #############################################################################################
# STAGE 4a Parameters
# #############################################################################################

s4a_in_full_json = s3_out_full_csv
VERSION = "v2"                     # uses as suffix in the output filename
SPLIT_COLNAME = "round"             # column name to split by

s4a_out_filename_template = "healsl_ROUND_rapid_MODELNAME_VERSION.csv"

# #############################################################################################
# STAGE 4b Parameters
# #############################################################################################

# Path to file after STAGE 3; extracting ICD10 codes
# PROCESS_FILE = "s3_output_gpt3_sample.csv"

s4b_in_smpl_csv_filepath = s3_out_smpl_csv
# smpl_analysis_in_csv_path = os.path.join(script_dir, OUTPUT_DIR, PROCESS_FILE)

# Path to export file after sample analysis
s4b_out_csv_filename = "healsl_rd1to2_rapid_gpt3_sample100_v1.csv"
s4b_out_csv_filepath = os.path.join(script_dir, OUTPUT_DIR, s4b_out_csv_filename)
# analyzed_out_csv_filename = PROCESS_FILE.replace(".csv", "_aggregated.csv")

# File required to convert ICD10 codes to CGHR10 titles
s4b_icd10_cghr10_filename = "icd10_cghr10_v1.csv"
icd10_cghr10_map_file = os.path.join(script_dir, DATA_DIR, s4b_icd10_cghr10_filename)


# COLOUR class for colour coding text in the console
class COLOUR:
    yellow = '\033[93m'
    red = '\033[91m'
    green = '\033[92m'
    blue = '\033[94m'
    purple = '\033[95m'
    cyan = '\033[96m'
    white = '\033[97m'
    black = '\033[90m'
    bold = '\033[1m'
    underline = '\033[4m'
    end = '\033[0m'

def stage_1_prepare_data(
    import_data_dir: str, 
    export_full_path: str, 
    export_sample_path: str, 
    include_sampling: bool = True,
    sample_size:int = 100, 
    sample_rep:int = 10
) -> None:
    """
    Runs Stage 1: Data Preparation. This function loads the Open Mortality data
    files from the specified directory, extracts relevant columns, and merges
    them into a single dataframe to create the full data dataframe. Secondly, it
    extracts samples from the full data dataframe and creates the sample data
    dataframe. The full and sample data dataframes are then exported to CSV
    files.

    Args:
        import_dir (str): The directory path where the  data files are located.
        export_full_path (str): The file path to export the full data dataframe.
        export_sample_path (str): The file path to export the sample data dataframe. 
        include_sampling (bool, optional): Flag to indicate whether to include sampling. Defaults to True.
        sample_size (int, optional): The number of samples to extract for the sample dataframe. Defaults to 100. 
        sample_rep (int, optional): The number of repetitions for each sample. Defaults to 10.

    Returns:
        None
    """

    def reorder_df(df):
        """
        Reorders the columns of a DataFrame by moving the 'uid' column to the front.
        
        Args:
            df (pandas.DataFrame): The DataFrame to be reordered.
            
        Returns:
            pandas.DataFrame: The reordered DataFrame.
        """
        return df[['uid'] + [col for col in df.columns if col != 'uid']]

    def get_samples(df, n=sample_size, repetition=sample_rep):
        """
        Extracts samples from a dataframe based on specified parameters. Samples are extracted proportionally from each group.
        A unique key 'uid' is created for each sample to represent the sampling repetition.

        Args:
            df (pandas.DataFrame): The input dataframe.
            n (int, optional): The number of samples to extract. Defaults to the global variable 'sample_size'.
            repetition (int, optional): The number of repetitions for each sample. Defaults to the global variable 'sample_rep'.

        Returns:
            pandas.DataFrame: The final sample dataframe.

        """
        app_logger.info(f"Extracting samples from dataframe. n={n}, repetition={repetition}")
        temp_df = df.copy()

        # age_group + round is essentially the group, this column will assist collecting samples from different groups
        temp_df['group'] = temp_df['age_group'] + "_" + temp_df['round']

        # Get the sampling fraction
        sampling_frac = ((temp_df.value_counts('group') / len(temp_df)) * n).round(0).astype(int).to_dict()

        # Initialize the dictionary to store the sample ids
        sample_ids = {}

        # Get sample based on fraction for each group
        for sample in sampling_frac:
            sample_ids[sample] = temp_df[temp_df['group'] == sample].sample(sampling_frac[sample], random_state=1).rowid.tolist()    
            app_logger.info(f"{sample}: {sampling_frac[sample]} records")

        # Sort dict from largest group to smallest
        sorted_sample_ids = dict(sorted(sample_ids.items(), key=lambda item: len(item[1]), reverse=True))

        # Get the actual samples count
        sample_values_count = len([item for subitem in sorted_sample_ids.values() for item in subitem])

        # If sample count is more than N_SAMPLES required, remove the excess samples
        # starting from the group with the most samples and more than 10 samples
        if sample_values_count > n:
            excess = sample_values_count - n
            app_logger.info(f"Sampled more than {n} needed. Removing excess samples.")

            for _ in range(excess):
                for key in sorted_sample_ids:

                    if len(sorted_sample_ids[key]) > 10:
                        sorted_sample_ids[key].pop()
                        break
        else:
            app_logger.info(f"Total samples: {sample_values_count}. No need to remove any samples.")

        # Flatten the sample dictionary to a list of rowids
        sample_ids_list = [item for sublist in sorted_sample_ids.values() for item in sublist]
        app_logger.debug(f"Sampled {len(sample_ids_list)} rows.")

        # Compile a dataframe according to sampled rowids
        random_rowids = temp_df[temp_df['rowid'].isin(sample_ids_list)]
        app_logger.debug(f"Compile dataframe according to sampled rowids")

        # Construct a unique id rowid_repetition and append to the dataframe
        list_of_repeated_dfs = [random_rowids.assign(uid=random_rowids['rowid'].astype(str) + "_" + str(r)) for r in range(repetition)]
        final_sample_df = pd.concat(list_of_repeated_dfs)
        app_logger.debug(f"Create uid for sampling repetition")

        # done with 'group', so drop it
        final_sample_df = final_sample_df.drop(columns=['group'])
        app_logger.debug(f"Drop 'group' column")

        # Reorder uid as first column for presentation
        final_sample_df = reorder_df(final_sample_df)
        app_logger.debug(f"Reorder dataframe columns")

        # Fill sex_code with empty string
        final_sample_df['sex_cod'] = final_sample_df['sex_cod'].fillna("")
        app_logger.debug(f"Fill empty string in NaN values")

        app_logger.info(f"Sample Dataframe complete. Returning dataframe.")
        app_logger.debug(f"Final sample dataframe shape: {final_sample_df.shape}")
        return final_sample_df
    
    def get_full_df(df):
        """
        Returns a the full data dataframe with the 'uid' column added and reordered. Also ensures that the
        sex_cod column has no NaN values.

        Parameters:
        - df: pandas.DataFrame
            The input dataframe.

        Returns:
        - pandas.DataFrame
            The full data dataframe with the above transformations applied.
        """
        app_logger.info(f"Creating full data dataframe.")
        app_logger.debug(f"Duplicate dataframe")
        temp_df = df.copy()
        
        app_logger.debug(f"Create uid for full dataframe")
        temp_df = temp_df.assign(uid=temp_df['rowid'])
        
        app_logger.debug(f"Reorder dataframe columns")
        temp_df = temp_df[['uid'] + [col for col in temp_df.columns if col != 'uid']]
        
        app_logger.debug(f"Fill empty string in NaN values")
        temp_df['sex_cod'] = temp_df['sex_cod'].fillna("")
        
        app_logger.info(f"Full Dataframe complete. Returning dataframe.")
        app_logger.debug(f"Final full dataframe shape: {temp_df.shape}")
        
        return temp_df

    # Main Program - main body of Stage 1    
    app_logger.debug(f"Checking if export files exist...")
    full_data_exists = check_file_exists(export_full_path)
    sample_data_exists = check_file_exists(export_sample_path)
    
    app_logger.info(f"Full prepared data exists: {COLOUR.green if full_data_exists else COLOUR.red}{full_data_exists}{COLOUR.end}")
    app_logger.info(f"Sample prepared data exists: {COLOUR.green if sample_data_exists else COLOUR.red}{sample_data_exists}{COLOUR.end}")
    
    # full and sample data already exists, skip Stage 1
    if full_data_exists and sample_data_exists:
        app_logger.info(f"Export files found. {COLOUR.green}Skipping Stage 1.{COLOUR.end}")
        return
    
    # If export dir does not exist, create it
    export_dir = os.path.dirname(os.path.abspath(export_full_path))

    if not os.path.exists(export_dir):
        app_logger.info(f"Export dir not found. Creating directory: {COLOUR.yellow}{export_dir}{COLOUR.end}")
        os.makedirs(export_dir)
        
    # Check if the DATA dir exists; for import data
    data_dir_exists = os.path.isdir(import_data_dir)
    app_logger.error(f"Import directory exists: {COLOUR.green if data_dir_exists else COLOUR.red}{data_dir_exists}{COLOUR.end}")
    
    if not data_dir_exists:
        raise ValueError(f"Error: Import dir {import_data_dir} not found. Make sure you create the directory and place all the relevant data files there.")

    # Iteratively read various data files, gather relevant columns, merge them into a single dataframe
    merged_all_df = pd.DataFrame()

    rounds = ['rd1', 'rd2']
    age_groups = ['adult', 'child', 'neo']

    for r in rounds:
        for a in age_groups:

            # Guess the file path           
            questionnaire_data_path = os.path.join(import_data_dir, f"healsl_{r}_{a}_v1.csv")
            age_data_path = os.path.join(import_data_dir, f"healsl_{r}_{a}_age_v1.csv")
            narrative_data_path = os.path.join(import_data_dir, f"healsl_{r}_{a}_narrative_v1.csv")
            
            try:
                questionnaire_df =  pd.read_csv(questionnaire_data_path, low_memory=False)
                age_df =            pd.read_csv(age_data_path, low_memory=False)
                narrative_df =      pd.read_csv(narrative_data_path, low_memory=False)
            except FileNotFoundError as e:
                raise ValueError(f"File not found: {e}")
                
            narrative_df = narrative_df.rename(columns={'summary': 'open_narrative'})
            
            # Extract relevant columns from each dataframe and merge them
            try:
                narrative_only = narrative_df[['rowid','open_narrative']]
                sex_only = questionnaire_df[['rowid','sex_cod']]
                age_only = age_df[['rowid','age_value_death','age_unit_death']]
            except KeyError as e:
                raise ValueError(f"KeyError: {e}")
            
            merged_df = narrative_only.merge(sex_only, on='rowid').merge(age_only, on='rowid')

            # Fill in missing values with empty string
            merged_df['sex_cod'] = merged_df['sex_cod'].fillna('')
            
            # Add extra columns age_group and round for clarify
            merged_df['age_group'] = a
            merged_df['round'] = r
            

            assert not merged_df.isnull().values.any(), "Execution halted: NaN values found in merged_df"

            app_logger.info(f"Reading: {COLOUR.yellow}{r}{COLOUR.end}, {COLOUR.yellow}{a}{COLOUR.end} - {COLOUR.yellow}{str(merged_df.shape[0])}{COLOUR.end} records")
            
            merged_all_df = pd.concat([merged_all_df, merged_df])

    app_logger.debug(f"Merging dataframes complete. Total rows: {merged_all_df.shape[0]}")    
        
    # Get samples and export
    if include_sampling:
        final_sample_df = get_samples(merged_all_df, n=N_SAMPLES, repetition=N_REPEAT_RESPONSES)
        app_logger.info(f"Exporting sample data, shape {final_sample_df.shape} to {COLOUR.yellow}{export_sample_path}{COLOUR.end}")
        final_sample_df.to_csv(export_sample_path, index=False)
    else:
        app_logger.info(f"{COLOUR.red}Skipping sample data export.{COLOUR.end}")

    # Get full data and export
    final_full_df = get_full_df(merged_all_df)    
    app_logger.info(f"Exporting full data, shape {final_full_df.shape} to {COLOUR.yellow}{export_full_path}{COLOUR.end}")
    final_full_df.to_csv(export_full_path, index=False)


    app_logger.info(f"{COLOUR.green}Stage 1 completed{COLOUR.end}")
    pass

def stage_2_generate_gpt_responses(
    include_sampling: bool = True,
    ) -> None:
    """
    Runs Stage 2: Generates GPT responses for the verbal autopsy narratives and
    periodically saves the responses to a file.
    
    Args:
        include_sampling (bool, optional): Flag to indicate whether to include sampling. Defaults to True.

    Returns:
        None
    """
    
    def load_va_data(filename) -> pd.DataFrame:
        """
        Load verbal autopsy preprocessed in Stage 1 from a CSV file and return as a DataFrame.

        Args:
            filename (str): The path to the CSV file containing the dataset.

        Returns:
            pd.DataFrame: The loaded dataset as a pandas DataFrame.

        Raises:
            ValueError: If the specified file is not found.
        """
        try:
            # Load dataset
            app_logger.debug(f"Loading dataset from {filename}")
            temp_df = pd.read_csv(filename)
            
            app_logger.debug(f"Dataset loaded. Returning DataFrame. Shape: {temp_df.shape}")    
            return temp_df
        except FileNotFoundError as e:
            raise ValueError(f"Error: {e}")
    
    def check_dataframe_columns(df):
        """
        Check if all the required columns are present in the verbal autopsy DataFrame.

        Args:
            df (pandas.DataFrame): The dataframe to be checked.

        Returns:
            list: A list of extra column names found in the dataframe.
        """
        dataset_passed = True
        required_colnames = ['uid', 'rowid', 'age_value_death', 'age_unit_death',
                            'open_narrative', 'sex_cod']
        app_logger.debug(f"Required columns: {required_colnames}")

        # Check all required columns are in the df
        for colnames in required_colnames:
            if colnames not in df.columns:
                logging.error(f"Missing column: \"{colnames}\"")
                dataset_passed = False

        if not dataset_passed:
            raise ValueError(f"Error: Some columns missing. Expected {required_colnames}")

        # Get columns names that are not required
        extra_colnames = [colname for colname in df.columns if colname not in required_colnames]

        if len(extra_colnames) > 0:
            app_logger.info(f"{len(extra_colnames)} extra column(s) found")
            extra_colnames = ['uid'] + extra_colnames

        return extra_colnames
    
    def trim_dataframe(df, demo_size_limit=10, demo_random=False) -> pd.DataFrame:
        """
        For demo purposes, return a subset of the input DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame to be trimmed.
            demo_size_limit (int): The maximum number of records to keep in the trimmed DataFrame. Defaults to 10.
            demo_random (bool): If True, randomly select records from the DataFrame. If False, select the first records. Defaults to False.

        Returns:
            pd.DataFrame: The trimmed DataFrame.

        """
        # When DEMO_MODE is set to True, process only a subset of the data        
        app_logger.info(f"Trimming DataFrame. Reducing rows {df.shape[0]} -> {demo_size_limit}.") 
        
        limit_records = int(demo_size_limit)        
        temp_df = df.sort_values(by='uid')
        
        if demo_random:
            temp_df = temp_df.sample(limit_records)
        else:
            temp_df = temp_df.head(limit_records)
        
        app_logger.info(f"Trimming complete. New DataFrame shape: {temp_df.shape}")    
        return temp_df

    def recursive_dict(obj):
        """
        Recursively converts a nested object into a dictionary. Used to convert ChatCompletion objects to 
        dictionaries for quick serialization.

        Args:
            obj: The object to be converted.

        Returns:
            A dictionary representation of the object.
        """
        if isinstance(obj, dict):
            return {k: recursive_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_dict(v) for v in obj]
        elif hasattr(obj, '__dict__'):
            return recursive_dict(obj.__dict__)
        else:
            return obj
    
    def load_export_data(filename=s2_out_full_processed_json) -> dict:
        """
        Load previously processed data if available. The file is initially
        generated when processing any verbal autopsy record via the API. If no
        file is detected, an empty dictionary is returned. This allows the
        script to resume from its last point in case of interruptions and
        minimizes unnecessary API calls.

        Args:
            filename (str): The name of the file to load the data from. Defaults
            to the global variable `s2_out_full_processed_json`.

        Returns:
            dict: The loaded data as a dictionary.

        """
        
        app_logger.info(f"Locating previous safepoint...")
        app_logger.debug(f"Checking for file: {filename}")
        
        if os.path.exists(filename):            
            with open(filename, 'r') as file:
                data = json.load(file)
                app_logger.info(f"Safepoint found. {COLOUR.yellow}{len(data)}{COLOUR.end} records loaded.")
            return data        
        
        app_logger.info(f"Safepoint {COLOUR.red}not found{COLOUR.end}. Starting from scratch.")
        return {}

    def save_export_data(data, filename=s2_out_full_processed_json):
        """
        Save the entire processed dictionary to a JSON file. This is meant to be
        used to save the data periodically.

        Args:
            data: The data to be saved.
            filename: The name of the file to save the data to. Defaults to the
            global variable 's2_out_full_processed_json'.

        Returns:
            None
        """
        # Save data to a file   
        with open(filename, 'w') as file:
            json.dump(data, file)
        
    def get_completion(
        messages: list[dict[str, str]],
        model: str = "gpt-3.5-turbo-0125",
        max_tokens=30,
        temperature=0,        
        tools=None,
        logprobs=None,
        top_logprobs=None,
    ) -> str:
        """
        Generates a completion using the OpenAI Chat API.

        Args:
            messages (list[dict[str, str]]): A list of messages in the conversation.
            model (str, optional): The model to use for generating the completion. Defaults to "gpt-3.5-turbo-0125".
            max_tokens (int, optional): The maximum number of tokens in the generated completion. Defaults to 30.
            temperature (float, optional): Controls the randomness of the generated completion. Defaults to 0.
            tools (str, optional): Additional tools to use for generating the completion. Defaults to None.
            logprobs (int, optional): Include log probabilities in the response. Defaults to None.
            top_logprobs (int, optional): Include top log probabilities in the response. Defaults to None.

        Returns:
            str: The generated completion.

        """
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
        }
        if tools:
            params["tools"] = tools

        completion = client.chat.completions.create(**params)
        return completion
    
    def get_api_response(
        input_df: pd.DataFrame, 
        passthrough_colnames: list, 
        output_response_path: str
    ) -> None:
        """
        Iteratively process each row in a DataFrame using the OpenAI API and
        periodically save the responses to a file.

        Args:
            input_df (pandas.DataFrame): The input DataFrame containing VA records.
            passthrough_colnames (list): List of column names to be passed through to the output.
            output_response_path (str): The file path to save the output response data.

        Returns:
            None
        """
        width = 70
        
        app_logger.info(f"Processing {input_df.shape[0]} VA records using model {MODEL_NAME}")        
                
        data_storage = load_export_data(filename=output_response_path)
        
        # return
        
        # data_storage = load_data()
        skipped_rows = []
        repeated_skips = False
        
        print()
        print(" BEGIN MODEL RESPONSE GENERATION ".center(width, '*'))
        print()
        print(f"Model: {COLOUR.yellow}{MODEL_NAME}{COLOUR.end}")
        print(f"Temperature: {COLOUR.yellow}{TEMPERATURE}{COLOUR.end}")
        print(f"Logprobs: {COLOUR.yellow}{LOGPROBS}{COLOUR.end}")
        print(f"Data Shape: {COLOUR.yellow}{input_df.shape}{COLOUR.end}")
        
        print(f"Generating responses using VA records...")
        print()
        
        for index, row in input_df.iterrows():            
            uid = row['uid']    
            
            # Check if rowid already processed. Testing both because json changes int keys to str    
            if (uid) in data_storage or str(uid) in data_storage:
                repeated_skips = True
                skipped_rows.append(uid)
                continue

            rowid = row['rowid']
            narrative = row['open_narrative']
            sex_cod = row['sex_cod']
            age_value_death = row['age_value_death']
            age_unit_death = row['age_unit_death']
            other_columns = row[passthrough_colnames].to_dict()
            
            prompt = USR_PROMPT
            prompt = prompt.replace('AGE_VALUE_DEATH', str(age_value_death))
            prompt = prompt.replace('AGE_UNIT_DEATH', age_unit_death.lower())
            prompt = prompt.replace('SEX_COD', str(sex_cod).lower())
            prompt = prompt.format(open_narrative=narrative)
                        
            completion = get_completion(
                [
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": prompt}
                ] ,
                model=MODEL_NAME,
                logprobs=LOGPROBS,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            
 
            current_time = datetime.datetime.now(tz=TIMEZONE).isoformat()
            
            data_storage[str(uid)] = {
                'uid': uid,               # 'uid' is the unique identifier for the dataset
                'rowid': rowid,
                'param_model': MODEL_NAME,
                'param_temperature': TEMPERATURE,
                'param_logprobs': True,
                'param_max_tokens': MAX_TOKENS,
                'param_system_prompt': SYS_PROMPT,
                'param_user_prompt': prompt,
                'timestamp': current_time,
                'output': recursive_dict(completion),
            }

            if not DROP_EXCESS_COLUMNS:
                # Append columns not required by the script but exists on the original dataset
                data_storage[str(uid)].update(other_columns)

            # Save data periodically (you can adjust the frequency based on your needs)    
            if index % SAVE_FREQ == 0 and index > 0:
                if repeated_skips:
                    print("\n", flush=True)
                repeated_skips = False
                
                save_export_data(data=data_storage, filename=output_response_path)

                print(f"Saving index: {str(index).ljust(8)} Processing: {str(uid).ljust(12)} Rows skipped: {len(skipped_rows)}", sep=' ', end='\r', flush=True)
                # break
                

        try:
            save_export_data(data=data_storage, filename=output_response_path)
            print(f"Saving index: {str(index).ljust(8)} Processing: {str(uid).ljust(12)} Rows skipped: {len(skipped_rows)}", sep=' ', end='\r', flush=True)
            print("", flush=True)
            print("Data saved successfully.")
        except Exception as e:
            print(f"Error saving last data: {e}")

        print()
        print(" GENERATION COMPLETED ".center(width, '*'))
        print()
        
        
        if len(skipped_rows) > 0:
            print(f"{len(skipped_rows)} rows skipped. Check skip file for details.")
            
            # Write skipped rows to a file
            # s1_out_full_csv = os.path.join(script_dir, TEMP_DIR, s1_out_csv_filename)
            
            skipped_report = os.path.join(script_dir, f"{TEMP_DIR}/log_stage_2_skipped_{get_curr_et_datetime_str()}.txt")

            with open(skipped_report, "w") as file:
                file.write(f"The follow rows are skipped because they were already processed.\n")
                for item in skipped_rows:        
                    file.write(f"{str(item)}\n")        

    # Main Program - main body of Stage 2
        
    # Check if export directory exists
    app_logger.info("Checking if export directory exists...")
    create_dir(s2_out_full_processed_json)
    
    # Initialize OpenAI client
    try:
        app_logger.info("Initializing OpenAI client.")

        open_api_key = os.environ.get('OPEN_API_KEY')
        client = OpenAI(api_key=open_api_key)
        
    except Exception as e:
        raise ValueError(f"Error: {e}")

    # Begin processing FULL dataset
    app_logger.info(f"{COLOUR.green}Begin FULL dataset Responses Generation{COLOUR.end}")
    
    # Load FULL dataset from specified CSV file
    df_full = load_va_data(s2_in_full_csv)
    
    # Validate to ensure all required columns are present, extract extra column names
    full_extra_colnames = check_dataframe_columns(df_full)
    
    # If DEMO_MODE is True, 'df_full' will become a subset of its original size    
    if DEMO_MODE:
        app_logger.info(f"{COLOUR.yellow}DEMO MODE: {DEMO_MODE}{COLOUR.end}")
        df_full = trim_dataframe(
            df=df_full,
            demo_size_limit=DEMO_SIZE_LIMIT,
            demo_random=DEMO_RANDOM            
            )
    
    # Push each DataFrame row to API and save response to file
    get_api_response(
        input_df=df_full,
        passthrough_colnames=full_extra_colnames,
        output_response_path=s2_out_full_processed_json
    )
    
    app_logger.info(f"{COLOUR.green}FULL dataset Responses Generation completed{COLOUR.end}")
    
    
    # Repeat same steps for SAMPLE dataset
    if include_sampling:
        app_logger.info(f"{COLOUR.green}Begin SAMPLE dataset Responses Generation{COLOUR.end}")
        df_sample = load_va_data(s2_in_smpl_csv)
        sample_extra_colnames = check_dataframe_columns(df_sample)
        
        if DEMO_MODE:
            app_logger.info(f"{COLOUR.yellow}DEMO MODE: {DEMO_MODE}{COLOUR.end}")
            df_sample = trim_dataframe(
                df=df_sample,
                demo_size_limit=DEMO_SIZE_LIMIT,
                demo_random=DEMO_RANDOM
            )
            
        get_api_response(
            input_df=df_sample,
            passthrough_colnames=sample_extra_colnames,
            output_response_path=s2_out_smpl_processed_json
        )
        
        app_logger.info(f"{COLOUR.green}SAMPLE dataset Responses Generation completed{COLOUR.end}")
    else:
        app_logger.info(f"{COLOUR.red}Skipping sample data responses generation.{COLOUR.end}")
    
    app_logger.info(f"{COLOUR.green}Stage 2 Complete{COLOUR.end}")

    pass

def stage_3_extract_info(response_data_file: str, return_data_file: str):
    """
    Runs Stage 3: Extracts information from the response data file.

    This function loads the response data from the specified file, extracts
    ICD-10 codes and their associated probabilities using token analysis,
    converts the extracted codes into individual columns, and generates export
    filenames based on the input filename.

    Args:
        response_data_file (str): The path to the response data file.
        return_data_file (str): The path to the return data file.

    Returns:
        None
    """


    def load_response_data(filename):
        """
        Load response data from a JSON file, processed from Stage 2.

        Args:
            filename (str): The path to the JSON file.

        Returns:
            list: The loaded data as a list of records.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                data = json.load(file)
                app_logger.info(f"{filename} loaded. {COLOUR.cyan}{len(data)}{COLOUR.end} records found.")
            return data
        else:
            raise FileNotFoundError(f"{filename} not found.")

    def extract_icd_probabilities(logprobs, debug=False):
        """
        Extracts ICD-10 codes and their associated probabilities from a list of
        tokens and log probabilities.

        This function iterates over the list of tokens and log probabilities,
        concatenating tokens together and checking if they match the pattern of
        an ICD-10 code. If a match is found, it calculates the mean linear
        probability of the ICD-10 code and packages the ICD-10 code, mean linear
        probability, and associated tokens and log probabilities into a
        dictionary. It then appends this dictionary to a list of parsed ICD-10
        codes.

        Args:
            logprobs (list): A list of lists, where each inner list contains a token and its associated log probability.
            debug (bool, optional): If set to True, the function prints debug information. Defaults to False.

        Returns:
            list: A list of dictionaries, where each dictionary contains an ICD-10 code, its mean linear probability, 
                and a dictionary of associated tokens and log probabilities.
        """
        parsed_icds = []
        tmp_df = pd.DataFrame(logprobs)
        if debug > 0:
            app_logger.debug(repr(''.join(tmp_df.iloc[:,0])))
        tmp_df_limit = len(tmp_df)
        for pos in range(tmp_df_limit):
            # Concatenate 2, 4, or 5 tokens to form ICD-10 codes
            temp_concat_ANN = ''.join(tmp_df.iloc[pos:pos+2, 0]).strip()
            temp_concat_ANN_NNN = ''.join(tmp_df.iloc[pos:pos+4, 0]).strip()
            temp_concat_ANN_NNN_A = ''.join(tmp_df.iloc[pos:pos+5, 0]).strip()
            temp_concat_ANA_NNN = ''.join(tmp_df.iloc[pos:pos+5, 0]).strip()
            
            # Reference: https://www.webpt.com/blog/understanding-icd-10-code-structure
            
            # Pattern for ICD-10 codes, A = Alphabet, N = Number
            # 'ANN' (e.g., 'A10')
            # 'ANN.NNN' (e.g., 'A10.001')
            # 'ANN.NNNA' (e.g., 'A10.001A') 
            # Note: last alphabet valid only if there are 6 characters before it
            pattern_ANN = r"^[A-Z]\d[0-9A-Z]$"
            pattern_ANN_NNN = r"^[A-Z]\d[0-9A-Z]\.\d{1,3}$"        
            pattern_ANN_NNN_A = r"^[A-Z]\d[0-9A-Z]\.\d{3}[A-Z]$"        
            
            # Check if the concatenated tokens match the ICD-10 code patterns
            match_ANN = re.match(pattern_ANN, temp_concat_ANN)
            match_ANN_NNN = re.match(pattern_ANN_NNN, temp_concat_ANN_NNN)
            match_ANN_NNN_A = re.match(pattern_ANN_NNN_A, temp_concat_ANN_NNN_A)
            match_ANA_NNN = re.match(pattern_ANN_NNN, temp_concat_ANA_NNN)
            
            # [debug] Each line will show which of the 3 patterns matched for the 3 token
            if debug == 2:
                app_logger.debug(
                    str(pos).ljust(4), 
                    repr(temp_concat_ANN).ljust(10), 
                    ('yes' if match_ANN else 'no').ljust(15), 
                    repr(temp_concat_ANN_NNN).ljust(10), 
                    ('yes' if match_ANN_NNN else 'no').ljust(15), 
                    repr(temp_concat_ANN_NNN_A).ljust(10), 
                    ('yes' if match_ANN_NNN_A else 'no').ljust(15),
                    repr(temp_concat_ANA_NNN).ljust(10), 
                    ('yes' if match_ANA_NNN else 'no').ljust(5)
                    )
            
            # Check match from longest to shortest
            # If a match is found, calculate the mean linear probability 
            # and package the ICD-10 code and associated data
            if match_ANN_NNN_A:
                winning_df = pd.DataFrame(logprobs[pos:pos+5])
                winning_icd = temp_concat_ANN_NNN_A
            elif match_ANA_NNN:
                winning_df = pd.DataFrame(logprobs[pos:pos+5])
                winning_icd = temp_concat_ANA_NNN
            elif match_ANN_NNN:
                winning_df = pd.DataFrame(logprobs[pos:pos+4])
                winning_icd = temp_concat_ANN_NNN            
            elif match_ANN:
                winning_df = pd.DataFrame(logprobs[pos:pos+2])
                winning_icd = temp_concat_ANN            
            else:
                continue

            # detect any rows that are just whitespace (e.g. \n), and drop those rows
            whitespc_index = winning_df[winning_df.loc[:, 0].str.isspace()].index.tolist()
            winning_df = winning_df.drop(whitespc_index)
            
            # [debug] Display the winning ICD-10 code and its associated data
            if debug == 2:
                print(f"**** {winning_icd} - VALID ICD ****")
                app_logger.debug(winning_df)
            
            # Convert log probabilities to linear probabilities and calculate the mean
            winning_mean = np.exp(winning_df.iloc[:, 1]).mean()
            
            # Package the ICD-10 code and associated data
            winning_package = {
                'icd': winning_icd,
                'icd_linprob_mean': winning_mean,
                'logprobs': winning_df.rename(columns={0: 'token', 1:'logprob'}).to_dict(orient='list')
            }

            # check if this ICD-10 is already in the list
            if winning_package in parsed_icds:
                if debug > 0:
                    app_logger.debug("Duplicate ICD-10 code found. Skipping...")
                continue
            
            # Append the package to the list of parsed ICD-10 codes
            parsed_icds.append(winning_package)
        
        # [debug] Display the parsed ICD-10 codes
        if debug > 0:
            app_logger.debug(parsed_icds) 
        
        # Check if parsed_icds is empty
        if not parsed_icds:
            # If it is, raise an error and show the logprobs in question
            app_logger.warning(f"ICD-10 code not found in this logprobs: {logprobs}")
                    
        # Drop the last element if there are more than 5 ICD10 extracted.
        if len(parsed_icds) > 5:
            parsed_icds = parsed_icds[:-1]

        return parsed_icds

    def output_icds_to_cols(value, pairs=PAIRS, sort_probs=False):
        """
        Converts a list of ICD-10 codes and their associated probabilities into a one-dimensional pandas Series.

        This function takes a list of tuples, where each tuple contains an ICD-10 code and its associated 
        probability. It converts this list into a DataFrame, sorts the DataFrame by descending probability, 
        drops the 'logprobs' column, reshapes the DataFrame into a one-dimensional Series, and pads the Series 
        to fill a specified number of columns.

        Args:
            value (list): A list of tuples, where each tuple contains an ICD-10 code and its associated probability.
            pairs (int, optional): The number of columns to pad the Series to. Defaults to PAIRS.

        Returns:
            pandas.Series: A one-dimensional Series containing the ICD-10 codes and their associated probabilities.
        """
        if value == []:
            return pd.Series([np.nan] * pairs * 2).astype(object)

        tmp = pd.DataFrame(value) # convert list of tuples to dataframe
        
        if sort_probs:
            tmp = tmp.sort_values(by="icd_linprob_mean", ascending=False) # sort by descending probability
            
        tmp = tmp.drop(columns=['icd_linprob_mean', 'logprobs'])
        tmp = tmp.stack().reset_index(drop=True) # convert to 1 row
        tmp = tmp.reindex(range(pairs*2), axis=1) # pad to fill PAIRS*2 columns
        

        return tmp

    def generate_export_filename(file_path) -> str:
        '''
        Generates an export filename for the parsed CSV file based on the input
        filename.

        Parameters:
            file_path (str): The full path of the input file.

        Returns:
            str: The name of the parsed CSV file.
        '''
        directory = os.path.dirname(file_path)
        file = os.path.basename(file_path)
        
        temp = file.split(".json")[0]
        temp = temp[2:]
        
        return f"{directory}/03{temp}_parsed_first_ICD.csv"
        
    # ##############################################################
    # Main Program - main body of Stage 3

    app_logger.info(f"{COLOUR.green}Beginning Stage 3: Information Extraction{COLOUR.end}")

    # Load response data from Stage 2 and convert to DataFrame
    app_logger.info(f"Loading response data from {response_data_file}.")
    data_storage = load_response_data(response_data_file)
    df = pd.DataFrame(data_storage).T

    # global export_first_ICD_CSV_file
    if return_data_file is None:
        return_data_file = generate_export_filename(response_data_file, type="first")
        app_logger.info(f"No export file specified. Defaulting to filename {return_data_file}.")

    # Get unrecognized colnames
    required_colnames = ['uid', 'rowid', 'param_model', 'param_max_tokens', 'param_temperature',
                        'param_logprobs', 'param_system_prompt', 'param_user_prompt',
                        'timestamp', 'output']

    # Get columns names that are not required
    extra_colnames = [colname for colname in df.columns if colname not in required_colnames]

    # Extract 4 new columns from 'output'
    app_logger.info("Extracting columns from 'output'...")
    df = df.assign(
        output_msg = df.output.apply(lambda x: x['choices'][0]['message']['content']),
        output_logprobs = df.output.apply(lambda x: [(token['token'], float(token['logprob'])) for token in x['choices'][0]['logprobs']['content']]),
        output_usage_completion_tokens = df.output.apply(lambda x: x['usage']['completion_tokens']),
        output_usage_prompt_tokens = df.output.apply(lambda x: x['usage']['prompt_tokens'])
        
    )

    # Extract ICD-10 codes and their associated probabilities to a new column
    app_logger.info("Extracting ICD-10 codes and probabilities...")
    df = df.assign(output_probs=df['output_logprobs'].apply(extract_icd_probabilities))

    # Count the number of ICD-10 codes in each response
    df['icd10_count'] = df['output_probs'].apply(len)

    # Generate column names for the exploded ICDs in cause{n}_icd10 and cause{n}_icd10_prob format
    icd_column_names_mapping = {i: f"cause{i + 1}_icd10" for i in range(PAIRS)}

    app_logger.info("Parsing ICD-10 codes to columns...")
    parsed_first_icd10_df = df.output_probs.apply(
        lambda x: output_icds_to_cols(x, sort_probs=False)
    ).rename(columns=icd_column_names_mapping)

    parsed_first_icd10_df = df.merge(
        parsed_first_icd10_df, 
        left_index=True, 
        right_index=True
    )

    # Define the mapping variable
    app_logger.info("Renaming columns...")
    column_mapping = {
        'model': 'output_model',
        'system_prompt': 'output_system_prompt',
        'user_prompt': 'output_user_prompt',
        'user_prompt': 'output_user_prompt',
        'timestamp': 'output_created',
    }

    # Rename the columns based on mapping 
    parsed_first_icd10_df = parsed_first_icd10_df.rename(columns=column_mapping)

    # Extract and reorganize only columns that are required
    app_logger.info("Organizing columns...")
    export_columns = []
    export_columns += ['rowid']
    export_columns += list(icd_column_names_mapping.values())
    export_columns += [
                        'param_model',
                        'param_system_prompt' , 
                        'param_user_prompt',
                        'param_max_tokens', 
                        'param_temperature',
                        'output_created',
                        'output_usage_completion_tokens', 
                        'output_usage_prompt_tokens', 
                        'output_msg',
                        'icd10_count',
                        'output_probs',
                    ]

    # Append extra columns if DROP_EXCESS_COLUMNS is False
    if not DROP_EXCESS_COLUMNS:
        export_columns += extra_colnames
    
    # Append 'output' column if DROP_RAW is False
    if not DROP_RAW:
        export_columns += ['output']


    # Show only relevant columns in the final dataframe
    export_parsed_first_icd10_df = parsed_first_icd10_df[export_columns]

    app_logger.info(f"Parsing finished. DataFrame shape: {COLOUR.cyan}{export_parsed_first_icd10_df.shape}{COLOUR.end}")
    app_logger.info(f"Export path: {COLOUR.yellow}{return_data_file}{COLOUR.end}")
    
    # Export the parsed DataFrame to a CSV file
    create_dir(return_data_file)
    export_parsed_first_icd10_df.to_csv(return_data_file, index=False)

    app_logger.info(f"{COLOUR.green}Information Extraction completed{COLOUR.end}")

    pass

def stage_4a_split_by_rounds(import_full_path: str, output_template: str = s4a_out_filename_template) -> None:
    app_logger.info(f"{COLOUR.green}[TOOLS] Begin splitting data by round{COLOUR.end}")

    if "gpt3" in import_full_path:
        model_name = "gpt3"
    elif "gpt4" in import_full_path:
        model_name = "gpt4"
    
    filename = output_template.replace("VERSION", VERSION)
    filename = filename.replace("MODELNAME", model_name) 
        

    df = pd.read_csv(import_full_path)
    
    split_unique_val_list = df[SPLIT_COLNAME].unique()
    
    app_logger.info(f"Input file: {import_full_path}")
    app_logger.info(f"Directory: {OUTPUT_DIR}")
    app_logger.info(f"Model: {model_name}")
    app_logger.info(f"Version: {VERSION}")
    app_logger.info(f"Split by {SPLIT_COLNAME}: {split_unique_val_list}")
    
    
    for curr_split in split_unique_val_list:
        split_df = df[df[SPLIT_COLNAME] == curr_split]
        
        app_logger.info(f"Preparing round: {curr_split} Shape: {COLOUR.cyan}{split_df.shape}{COLOUR.end}")
        
        
        
        split_filename = filename.replace("ROUND", curr_split)
        split_filename = os.path.join(script_dir, OUTPUT_DIR, split_filename)
        app_logger.info(f"Export to: {COLOUR.yellow}{split_filename}{COLOUR.end}")
        split_df.to_csv(split_filename, index=False)

def stage_4b_sample_analysis(input_data:str, output_data:str):
    

    def same_cause_icd10(input_data) -> pd.DataFrame:
        """
        Perform analysis by aggregating ICD10 codes and counting their frequencies.

        Args:
            input_data (str): The filepath of the input data file.

        Returns:
            pandas.DataFrame: The resulting dataframe containing the analysis results.
        """
        
        app_logger.info(f"{COLOUR.green}[TOOLS] Analysis by Aggregating {COLOUR.bold}ICD10 codes{COLOUR.end}")
        
        # Read the input data
        app_logger.info(f"Reading input data...")
        df = pd.read_csv(input_data)
        
        # Remove any ICDs with decimals
        app_logger.info(f"Same cause ICD10 analysis")
        app_logger.info(f"Removing decimals from ICD10 codes...")
        df[['cause1_icd10', 'cause2_icd10', 'cause3_icd10', 'cause4_icd10', 'cause5_icd10']] = df[['cause1_icd10', 'cause2_icd10', 'cause3_icd10', 'cause4_icd10', 'cause5_icd10']].map(lambda x: x.split('.')[0] if pd.notnull(x) else x)

        # Group similar rowids and count the frequency of ICD10 codes
        app_logger.info(f"Combining similar rowids and counting ICD10 code frequency...")
        grouped_df = df.groupby('rowid')
        same_cause_count_df = pd.DataFrame(grouped_df['cause1_icd10'].value_counts())
        
        # Create a blank df with 10 columns, 1x...10x.
        blank_df = pd.DataFrame(index=same_cause_count_df.reset_index().rowid.unique(), columns=[x+1 for x in range(10)])
        
        # Create a df and count the frequency of ICD10 codes
        # e.g. if there are two codes repeated twice and one code six times, 2x will be 2 and 6x will be 1
        dummy_df = pd.get_dummies(same_cause_count_df['count']).astype(int).groupby('rowid').sum()
        
        # Combine the blank and dummy df
        same_cause_count_df = blank_df.combine_first(dummy_df)
        
        # Rename the columns
        same_cause_count_df = same_cause_count_df.rename(columns=lambda x: f'same_cause1_icd10_{x}x')
        same_cause_icd10_colnames = same_cause_count_df.columns

        # non-binarized shows the values as is.
        # binarized reduces the repeated counts of a rowid record to 1
        binarized_sum = same_cause_count_df.sum()
        nbinarized_sum = same_cause_count_df[same_cause_icd10_colnames].apply(lambda x: x.astype(bool)).sum()
        print("Analysis of Repeated ICD10 Codes per record")    
        print("-------------------------------------------------------")
        print(pd.DataFrame({'binarized': binarized_sum, 'non-binarized': nbinarized_sum}))
        print()
        print("Binarized reduces repeated times-count of the same row to 1.")
        print("E.g. A row with (3) 2x plus (1) 4x will be counted as (1) 2x plus (1) 4x")
        print()
        print(f"Majority repeated (6x and above) similarity (0.0-1.0): {COLOUR.bold}{binarized_sum.iloc[-5:].sum()/len(df.rowid.unique())}{COLOUR.end}")
        
        # Generate dataframe
        
        aggregated_cause1_icd10_rows = []

        for name, group in grouped_df:
            aggregated_cause1_icd10_rows.append([name, group['cause1_icd10'].value_counts().to_dict()])
            
        combined_icd10_df = pd.DataFrame(aggregated_cause1_icd10_rows, columns=['rowid', 'cause1_icd10']).set_index('rowid')

        final_icd_df = same_cause_count_df.merge(combined_icd10_df, left_index=True, right_index=True)
        final_icd_df = final_icd_df.merge(df[['rowid', 'age_group', 'round']].drop_duplicates(subset='rowid').set_index('rowid'), left_index=True, right_index=True)
        # print(final_icd_df)
        
        return final_icd_df

    def same_cause_cghr10(input_data):
        
        app_logger.info(f"{COLOUR.green}[TOOLS] Analysis by Aggregating {COLOUR.bold}CGHR10 titles{COLOUR.end}")
        
        
        try:
            app_logger.info(f"Reading ICD10 to CGHR10 mapping file...")
            icd10_to_cghr_mapping = pd.read_csv(icd10_cghr10_map_file)
        except FileNotFoundError:        
            app_logger.error(f"{COLOUR.red}File not found: {icd10_cghr10_map_file}{COLOUR.end}")
            app_logger.error(f"This file is required to convert ICD10 codes to CGHR10 titles. Skipping CGHR10 analysis...")
            return None
        
        app_logger.info(f"Reading input data...")
        df = pd.read_csv(input_data)
        
        app_logger.info(f"Removing decimals from ICD10 codes...")
        df[['cause1_icd10', 'cause2_icd10', 'cause3_icd10', 'cause4_icd10', 'cause5_icd10']] = df[['cause1_icd10', 'cause2_icd10', 'cause3_icd10', 'cause4_icd10', 'cause5_icd10']].map(lambda x: x.split('.')[0] if pd.notnull(x) else x)

        # Map file has mapping information for all age groups, with some ICD10 shared across multiple 
        # age groups with different CGHR10 titles. To simplify this, we split each age group into separate
        # dataframes and store in a dictionary with age group as key.
        app_logger.info(f"Building CGHR10 mapping helper dictionary...")
        cghr_map_helper = {}
        for group in icd10_to_cghr_mapping.cghr10_age.unique():
            cghr_map_helper[group] = icd10_to_cghr_mapping[icd10_to_cghr_mapping.cghr10_age == group].set_index('icd10_code')

        # Convert cause1_icd10 to to CGHR10 title, based on the age group of the record.
        # In situations where mapping info does not exist, 'NA' is used as the CGHR10 title.
        cghr_df = df.assign(
            cause1_cghr10 = df.apply(lambda row: 
                'NA' if row.cause1_icd10 not in cghr_map_helper[row.age_group].index
                else cghr_map_helper[row.age_group].loc[row.cause1_icd10]['cghr10_title']
                , axis=1)
        )
        
        # count how many after grouping by rowid
        unmapped_code_count = len(cghr_df[cghr_df.cause1_cghr10 == 'NA'].groupby('rowid').size().value_counts())
        if unmapped_code_count > 0:
            app_logger.warning(f"{COLOUR.yellow}Some ICD10 codes could not be mapped to CGHR. For those records, the CGHR10 code is set to 'NA'.{COLOUR.end}")
            app_logger.warning(f"Number of NA in cause1_cghr10: {unmapped_code_count}")
        else:
            app_logger.info(f"All ICD10 codes were successfully mapped to CGHR10 titles.")
        
        # Similar process to same_cause_icd10(), see that function for details
        grouped_cghr_df = cghr_df.groupby('rowid')
        same_cause_count_cghr_df = pd.DataFrame(grouped_cghr_df['cause1_cghr10'].value_counts())
        blank_df = pd.DataFrame(index=same_cause_count_cghr_df.reset_index().rowid.unique(), columns=[x for x in range(1,11)])
        dummy_df = pd.get_dummies(same_cause_count_cghr_df['count']).astype(int).groupby('rowid').sum()
        
        same_cause_count_cghr_df = blank_df.combine_first(dummy_df)
        same_cause_count_cghr_df = same_cause_count_cghr_df.rename(columns=lambda x: f'same_cause1_cghr10_{x}x')
        same_cause_cghr10_colnames = same_cause_count_cghr_df.columns

        binarized_cghr_sum = same_cause_count_cghr_df.sum()
        nbinarized_cghr_sum = same_cause_count_cghr_df[same_cause_cghr10_colnames].apply(lambda x: x.astype(bool)).sum()
        
        print("Binarized and non-binarized sum (binarized reduces repeated counts of a rowid record to 1)")
        print("-------------------------------------------------------")
        print(pd.DataFrame({'binarized': binarized_cghr_sum, 'non-binarized': nbinarized_cghr_sum}))
        print()
        print("Binarized reduces repeated times-count of the same row to 1.")
        print("E.g. A row with (3) 2x plus (1) 4x will be counted as (1) 2x plus (1) 4x")
        print()
        print(f"Majority repeated CGHR10 similarity (6x and above) similarity (0.0-1.0): {COLOUR.bold}{binarized_cghr_sum.iloc[-5:].sum()/len(df.rowid.unique())}{COLOUR.end}")

        # Generate dataframe
        aggregated_cause1_cghr10_rows = []
        
        for name, group in grouped_cghr_df:
            aggregated_cause1_cghr10_rows.append([name, group['cause1_cghr10'].value_counts().to_dict()])

        combined_cghr10_df = pd.DataFrame(aggregated_cause1_cghr10_rows, columns=['rowid', 'cause1_cghr10']).set_index('rowid')

        final_cghr_df = same_cause_count_cghr_df.merge(combined_cghr10_df, left_index=True, right_index=True)
        
        # print(final_cghr_df)
        return final_cghr_df
        

    # main program
    # Aggregate ICD10 codes
    icd10_df = same_cause_icd10(input_data=input_data)
    
    # Aggregate CGHR10 titles
    cghr10_df = same_cause_cghr10(input_data=input_data)
    
    # When CGHR10 Analysis fails, export ICD10 analysis only
    if cghr10_df is None:
        app_logger.warning(f"CGHR10 Skipped. Exporting ICD10 analysis only...")
        icd10_df.to_csv(s4b_out_csv_filepath.replace("_aggregated.csv", "_icd10_similarity.csv"), index=False)
        return
    
    # Combine ICD10 and CGHR10 analysis, then export
    app_logger.info(f"Combining ICD10 and CGHR10 analysis into one DataFrame...")
    final_agg_df = icd10_df.merge(cghr10_df, left_index=True, right_index=True)
    final_agg_colnames = [c for c in final_agg_df.columns if c not in ['age_group', 'round']] + ['age_group', 'round']
    final_agg_df = final_agg_df[final_agg_colnames]
    
    final_agg_df = final_agg_df.reset_index().rename(columns={'index': 'rowid', 'cause1_icd10' : 'cause_icd10', 'cause1_cghr10': 'cause_cghr10'})
    
    # print(final_agg_df)
    app_logger.info(f"Exporting similarities data to: {COLOUR.yellow}{s4b_out_csv_filepath}{COLOUR.end}")
    final_agg_df.to_csv(s4b_out_csv_filepath, index=False)
    
    app_logger.info(f"{COLOUR.green}[TOOLS] Analysis complete{COLOUR.end}")
    pass

def logging_process():
    """
    This function initializes a logger named 'app_logger' and sets its level to DEBUG.
    It also creates a handler with a formatter and adds it to the logger.
    """
    global app_logger
    
    # Create a logger and set its level
    app_logger = logging.getLogger('app_logger')
    app_logger.setLevel(logging.DEBUG)

    # Create a handler with a formatter and add it to the logger
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(fmt='%(asctime)s:%(msecs)03d[%(funcName)s]: %(message)s', datefmt='%H:%M:%S')
    )

    app_logger.addHandler(handler)    

def check_file_exists(file):
    """
    Check if a file exists.

    Args:
        file (str): The path to the file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    app_logger.info(f"Checking if '{file}' exists... {'yes' if os.path.isfile(file) else 'no'}")
    return os.path.isfile(file)

def create_dir(file_path):
    """    
    Check and creates the directory of the given file path if it does not exist.

    
    Args:
        file_path (str): The path of the file.
    """
    dir_path = os.path.dirname(file_path)
    
    if not os.path.isdir(dir_path):
        app_logger.info(f"Directory '{dir_path}' does not exist. Creating it...")
        os.makedirs(dir_path)

def get_curr_et_datetime_str() -> str:
    """
    Returns the current Eastern Time (ET) datetime as a formatted string.
    
    Returns:
        str: The current ET datetime in the format "%y%m%d_%H%M%S".
    """
    return datetime.datetime.now(tz=TIMEZONE).strftime("%y%m%d_%H%M%S")

def precheck(
    import_data_dir: str, 
) -> bool:
    precheck_passed = True
        
    rounds = ['rd1', 'rd2']
    age_groups = ['adult', 'child', 'neo']

    print(f"Base Dir: {script_dir}")
    
    print(f"Required Files by Stage 1:")
    for r in rounds:
        for a in age_groups:
            # Guess the file path           
            questionnaire_data_path = os.path.join(import_data_dir, f"healsl_{r}_{a}_v1.csv")
            age_data_path = os.path.join(import_data_dir, f"healsl_{r}_{a}_age_v1.csv")
            narrative_data_path = os.path.join(import_data_dir, f"healsl_{r}_{a}_narrative_v1.csv")
            
            trimmed_questionnaire_data_path = os.path.relpath(questionnaire_data_path, script_dir)
            trimmed_age_data_path = os.path.relpath(age_data_path, script_dir)
            trimmed_narrative_data_path = os.path.relpath(narrative_data_path, script_dir)
            
            if os.path.isfile(questionnaire_data_path):
                print(f"[{COLOUR.green}\u2713{COLOUR.end}] {trimmed_questionnaire_data_path}")
            else:
                print(f"[{COLOUR.red}\u2717{COLOUR.end}] {trimmed_questionnaire_data_path}")
                
                precheck_passed = False
            
            if os.path.isfile(age_data_path):
                print(f"[{COLOUR.green}\u2713{COLOUR.end}] {trimmed_age_data_path}")
            else:
                print(f"[{COLOUR.red}\u2717{COLOUR.end}] {trimmed_age_data_path}")
                precheck_passed = False
            
            if os.path.isfile(narrative_data_path):
                print(f"[{COLOUR.green}\u2713{COLOUR.end}] {trimmed_narrative_data_path}")
            else:
                print(f"[{COLOUR.red}\u2717{COLOUR.end}] {trimmed_narrative_data_path}")
                precheck_passed = False            
    
    print(f"Required files by Stage 4b:")
    
    trimmed_icd10_cghr10_map_file = os.path.relpath(icd10_cghr10_map_file, script_dir)
    
    if INCLUDE_SAMPLING:
        if os.path.isfile(icd10_cghr10_map_file):
            print(f"[{COLOUR.green}\u2713{COLOUR.end}] {trimmed_icd10_cghr10_map_file}")
        else:
            print(f"[{COLOUR.red}\u2717{COLOUR.end}] {trimmed_icd10_cghr10_map_file}")
            print(f"This file is required to convert ICD10 codes to CGHR10 titles. Sample Analysis cannot be performed without it.")
            precheck_passed = False
        
    return precheck_passed

def main():
    """
    Process all stages; from acquiring neccessary features, process through API, to extracting results    
    """
    logging_process()
    
    # STAGE 0 - Prechecks
    app_logger.info(f"{COLOUR.green}Running prechecks...{COLOUR.end}")
    precheck_status = precheck(import_data_dir=s1_in_dir)
    
    if precheck_status:
        print(f"{COLOUR.green}Prechecks passed. Starting main process...{COLOUR.end}")
    else:
        print(f"{COLOUR.red}Prechecks failed, some files are missing.{COLOUR.end}")
        print(f"Please obtain all files from Open Mortality at openmortality.org")
        raise FileExistsError("Missing required files.")
    
    
    # STAGE 1 - Preprocess Data
    # preprocess data into a format that can be used by GPT
    app_logger.info(f"{COLOUR.green}Running stage 1: Preprocessing Data...{COLOUR.end}")
    stage_1_prepare_data(
        import_data_dir=s1_in_dir,
        export_full_path=s1_out_full_csv,
        export_sample_path=s1_out_smpl_csv,
        include_sampling=INCLUDE_SAMPLING,
        sample_size=N_SAMPLES,
        sample_rep=N_REPEAT_RESPONSES        
        )


    # STAGE 2 - Generate GPT Responses
    # send VA records and get GPT responses    
    app_logger.info(f"{COLOUR.green}Running stage 2: Generating GPT responses...{COLOUR.end}")
    stage_2_generate_gpt_responses()

    # STAGE 3 - Extract Information
    # full data
    app_logger.info("Running stage 3 on FULL data...")
    stage_3_extract_info(response_data_file=s3_in_full_json, return_data_file=s3_out_full_csv)
    
    # sample data. run only if sample data is available
    if INCLUDE_SAMPLING:        
        app_logger.info("Running stage 3 on SAMPLE data...")
        stage_3_extract_info(response_data_file=s3_in_smpl_json, return_data_file=s3_out_smpl_csv)
        
    # STAGE 4a - Split Full Data by rounds
    app_logger.info(f"{COLOUR.green}Running stage 4a: Split FULL data results by rounds...{COLOUR.end}")
    stage_4a_split_by_rounds(
        import_full_path=s4a_in_full_json,
        output_template=s4a_out_filename_template
    )

    # STAGE 4b - Analyze Sample Data
    if INCLUDE_SAMPLING:
        app_logger.info(f"{COLOUR.green}Running stage 4b: Sampling Analysis...{COLOUR.end}")
        stage_4b_sample_analysis(
            input_data=s4b_in_smpl_csv_filepath,
            output_data=s4b_out_csv_filepath
        )

if __name__ == "__main__":
    main()