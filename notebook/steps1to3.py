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

# #############################################################################################
# STEP 1 Parameters
# Set PROPORTIONAL_SAMPLING to True to sample the dataset proportionally, and False retain the entire dataset
# N_REPEAT_RESPONSES and N_SAMPLES are only used when PROPORTIONAL_SAMPLING is True
PROPORTION_SAMPLING = False
N_REPEAT_RESPONSES = 10
N_SAMPLES = 100  # Replace X with the desired number of values to select

STEP1_IMPORT_REL_DIR = "data_202402/"
STEP1_EXPORT_FULL_FILE = "step1_full_export.csv"
STEP1_EXPORT_SAMPLE_FILE = "step1_sample_export.csv"

# #############################################################################################
# STEP 2 Parameters
DEMO_MODE = True        # If set to True, the script will only process a subset of the data
DEMO_RANDOM = False     # If set to True, the script will process a random subset of the data
DEMO_SIZE_LIMIT = 5    # Size of the demo


# Discard columns not recognized by the script before saving output file.
# Keep this to False if you want the output file to retain columns needed for post-processing purposes.
DROP_EXCESS_COLUMNS = False

# File Settings
INPUT_FULL_CSV_FILE = STEP1_EXPORT_FULL_FILE
INPUT_SAMPLE_CSV_FILE = STEP1_EXPORT_SAMPLE_FILE

OUTPUT_FULL_PROCESSED_JSON_FILE = "./_temp/step2_processed_full.json"
OUTPUT_SAMPLE_PROCESSED_JSON_FILE = "./_temp/step2_sample_full.json"

WORDWRAP_WIDTH = 100

# How often the temp storage is saved to disk
SAVE_FREQ = 2

# Models Settings
GPT4 = "gpt-4-0613"
GPT3 = "gpt-3.5-turbo-0125"
MODEL_NAME = GPT3
TEMPERATURE = 0
LOGPROBS = True

SYS_PROMPT = """You are a physician with expertise in determining underlying causes of death in Sierra Leone by assigning the most probable ICD-10 code for each death using verbal autopsy narratives. Return only the ICD-10 code without description. E.g. A00 
If there are multiple ICD-10 codes, show one code per line."""

USR_PROMPT = """Determine the underlying cause of death and provide the most probable ICD-10 code for a verbal autopsy narrative of a AGE_VALUE_DEATH AGE_UNIT_DEATH old SEX_COD death in Sierra Leone: {open_narrative}"""

# #############################################################################################



def step1():
    app_logger.info("Beginning Step 1: Preprocessing data")

    if not os.path.isdir(STEP1_IMPORT_REL_DIR):
        raise ValueError("Invalid path_prefix: Directory does not exist")

    merged_all_df = pd.DataFrame()

    rounds = ['rd1', 'rd2']
    age_groups = ['adult', 'child', 'neo']

    for r in rounds:
        for a in age_groups:
            
            try:
                questionnaire_df =  pd.read_csv(f"{STEP1_IMPORT_REL_DIR}healsl_{r}_{a}_v1.csv", low_memory=False)
                age_df =            pd.read_csv(f"{STEP1_IMPORT_REL_DIR}healsl_{r}_{a}_age_v1.csv", low_memory=False)
                narrative_df =      pd.read_csv(f"{STEP1_IMPORT_REL_DIR}healsl_{r}_{a}_narrative_v1.csv", low_memory=False)
            except FileNotFoundError as e:
                raise ValueError(f"File not found: {e}")
                
            narrative_df = narrative_df.rename(columns={'summary': 'open_narrative'})
            
            # Merge the dataframes
            try:
                narrative_only = narrative_df[['rowid','open_narrative']]
                sex_only = questionnaire_df[['rowid','sex_cod']]
                age_only = age_df[['rowid','age_value_death','age_unit_death']]
            except KeyError as e:
                raise ValueError(f"KeyError: {e}")
            
            merged_df = narrative_only.merge(sex_only, on='rowid').merge(age_only, on='rowid')

            # Fill in missing values with empty string
            merged_df['sex_cod'] = merged_df['sex_cod'].fillna('')
            
            merged_df['age_group'] = a
            merged_df['round'] = r
            
            # merged_df['group'] = f"{a}_{r}"

            assert not merged_df.isnull().values.any(), "Execution halted: NaN values found in merged_df"

            app_logger.info(f"Reading: {r}, {a} - {str(merged_df.shape[0])} records")
            # print(f"Sample of merged_df {merged_df.shape}:")
            # display(merged_df.sample(5))
            
            merged_all_df = pd.concat([merged_all_df, merged_df])

    app_logger.info("Merging dataframes complete. Total rows: {merged_all_df.shape[0]}")    

    def reorder_df(df):
        return df[['uid'] + [col for col in df.columns if col != 'uid']]

    def get_samples(df, n=100, repetition=10):
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
    
    final_sample_df = get_samples(merged_all_df, n=N_SAMPLES, repetition=N_REPEAT_RESPONSES)

    final_full_df = get_full_df(merged_all_df)

    app_logger.info(f"Exporting sample to {STEP1_EXPORT_SAMPLE_FILE}")
    final_sample_df.to_csv(STEP1_EXPORT_SAMPLE_FILE, index=False)

    app_logger.info(f"Exporting full to {STEP1_EXPORT_FULL_FILE}")
    final_full_df.to_csv(STEP1_EXPORT_FULL_FILE, index=False)
    pass

def step2():
    
    # Function to load CSV dataset and return a DataFrame    
    def load_import_data(filename) -> pd.DataFrame:
        try:
            # Load dataset
            app_logger.debug(f"Loading dataset from {filename}")
            temp_df = pd.read_csv(filename)
            
            app_logger.debug(f"Dataset loaded. Returning DataFrame. Shape: {temp_df.shape}")    
            return temp_df
        except FileNotFoundError as e:
            raise ValueError(f"Error: {e}")
    
    # Function to check if all required columns are in the DataFrame, and return any extra colnames
    def validate_columns(df):
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
    
    # Function to trim the dataset to a limited size for demo purposes
    def trim_for_demo(df, demo_size_limit=DEMO_SIZE_LIMIT, demo_random=DEMO_RANDOM) -> pd.DataFrame:
        # When DEMO_MODE is set to True, process only a subset of the data        
        app_logger.info(f"Running trim_for_demo(). Reducing records {df.shape[0]} record -> {demo_size_limit} records.") 
        
        limit_records = int(demo_size_limit)        
        temp_df = df.sort_values(by='uid')
        
        if demo_random:
            temp_df = temp_df.sample(limit_records)
        else:
            temp_df = temp_df.head(limit_records)
        
        app_logger.info(f"Trimming complete. Returning DataFrame shape: {temp_df.shape}")    
        return temp_df
    
    # Function to get current time in string YYMMDD_HHMMSS format
    def get_current_str_time() -> str:
        return datetime.datetime.now(tz=TIMEZONE).strftime("%y%m%d_%H%M%S")

    # Used to convert ChatCompletion object to a dictionary, recursively
    def recursive_dict(obj):
        if isinstance(obj, dict):
            return {k: recursive_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_dict(v) for v in obj]
        elif hasattr(obj, '__dict__'):
            return recursive_dict(obj.__dict__)
        else:
            return obj
    
    # Function to load storage data to handle OpenAI responses
    def load_export_data(filename=OUTPUT_FULL_PROCESSED_JSON_FILE) -> dict:        
        if os.path.exists(filename):
            app_logger.info(f"{filename} found. Retrieving previous storage data...")
            with open(filename, 'r') as file:
                data = json.load(file)
                app_logger.info(f"{len(data)} records loaded.")
            return data        
        
        app_logger.info(f"{filename} not found. Initializing new storage data...")
        return {}

    # Function to save OpenAI response to storage file
    def save_export_data(data, filename=OUTPUT_FULL_PROCESSED_JSON_FILE):                
        
        # Save data to a file   
        with open(filename, 'w') as file:
            json.dump(data, file)
        
    # Function to send message to OpenAI API
    def get_completion(
        messages: list[dict[str, str]],
        model: str = "gpt-3.5-turbo-0125",
        max_tokens=30,
        temperature=0,        
        tools=None,
        logprobs=None,
        top_logprobs=None,
    ) -> str:

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
    
    def get_openai_response(input_df, passthrough_colnames, output_response_path):
        width = 70
        
        app_logger.info(f"Processing {input_df.shape[0]} VA records using model {MODEL_NAME}")        
        
        app_logger.info(f"Loading export data...")
        data_storage = load_export_data(filename=output_response_path)
        
        # return
        
        # data_storage = load_data()
        skipped_rows = []
        repeated_skips = False
        
        print()
        print(" BEGIN MODEL RESPONSE GENERATION ".center(width, '*'))
        print()
        
        print(f"Generating responses for every VA record...")
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
            )
            
 
            current_time = datetime.datetime.now(tz=TIMEZONE).isoformat()
            
            data_storage[str(uid)] = {
                'uid': uid,               # 'uid' is the unique identifier for the dataset
                'rowid': rowid,
                'param_model': MODEL_NAME,
                'param_temperature': 0,
                'param_logprobs': True,
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
            with open(f"./_temp/step2_skipped_{get_current_str_time()}.txt", "w") as file:
                file.write(f"The follow rows are skipped because they were already processed.\n")
                for item in skipped_rows:        
                    file.write(f"{str(item)}\n")        

    # Main Program
    app_logger.info("Beginning Step 2: Generating OpenAI responses")
    
    app_logger.info("Checking if export directory exists...")
    create_dir(OUTPUT_FULL_PROCESSED_JSON_FILE)
    
    try:
        app_logger.info("Initializing OpenAI client.")

        open_api_key = os.environ.get('OPEN_API_KEY')
        client = OpenAI(api_key=open_api_key)
        
    except Exception as e:
        raise ValueError(f"Error: {e}")

    # Load FULL dataset from specified CSV file
    df_full = load_import_data(INPUT_FULL_CSV_FILE)
    
    # Validate to ensure all required columns are present, extract extra column names
    full_extra_colnames = validate_columns(df_full)
    
    # If DEMO_MODE is set to True, process only a subset of the data    
    if DEMO_MODE:
        app_logger.info(f"DEMO MODE: {DEMO_MODE}")
        df_full = trim_for_demo(df_full)
    
    # Process the FULL dataset with OpenAI and save response to file
    get_openai_response(
        input_df=df_full,
        passthrough_colnames=full_extra_colnames,
        output_response_path=OUTPUT_FULL_PROCESSED_JSON_FILE
    )
    
    # Repeat same steps for SAMPLE dataset
    df_sample = load_import_data(INPUT_SAMPLE_CSV_FILE)
    sample_extra_colnames = validate_columns(df_sample)
    
    if DEMO_MODE:
        app_logger.info(f"DEMO MODE: {DEMO_MODE}")
        df_sample = trim_for_demo(df_sample)
        
    get_openai_response(
        input_df=df_sample,
        passthrough_colnames=sample_extra_colnames,
        output_response_path=OUTPUT_SAMPLE_PROCESSED_JSON_FILE
    )
    
    
    pass

def step3():
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

def main():
    # load logging preference
    # logging.getLogger().setLevel(logging.WARNING)
    logging_process()
    
    app_logger.info(f"Searching for '{STEP1_EXPORT_FULL_FILE}' and '{STEP1_EXPORT_SAMPLE_FILE}'.")    
    if not check_file_exists(STEP1_EXPORT_FULL_FILE) or not check_file_exists(STEP1_EXPORT_SAMPLE_FILE):
        app_logger.info("All or some export files missing. Proceeding with Step 1.")
        step1()
    else:
        app_logger.info("Export files found. Skipping Step 1.")

    step2()
    step3()

if __name__ == "__main__":
    main()