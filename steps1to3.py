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
# STEP 3 Parameters

# IMPORT_O2_JSON_DATA = OUTPUT_FULL_PROCESSED_JSON_FILE
# IMPORT_O2_JSON_DATA = "./notebook/_working_data_240315/02_(all)_gpt3_0309.json"
IMPORT_02_JSON_DATA = "./notebook/_working_data_240315/02_(sampled)_gpt3_0308.json"
export_first_ICD_CSV_file = "./temp/step3_processed_full.csv"

# Data Analysis Settings
PAIRS = 5           # Generate up to 5 ICDs

# Output Settings
DROP_EXCESS_COLUMNS = False         # Set True to remove 'other_columns' from output dataframe
DROP_RAW = False                    # Set True to remove the 'raw' column, the original raw response, from the export file




def step_1_prepare_data():
    app_logger.info("Beginning Step 1: Data Preparation")

    app_logger.info(f"Searching for '{STEP1_EXPORT_FULL_FILE}' and '{STEP1_EXPORT_SAMPLE_FILE}'.")

    if check_file_exists(STEP1_EXPORT_FULL_FILE) and check_file_exists(STEP1_EXPORT_SAMPLE_FILE):
        app_logger.info("Export files found. Skipping Step 1.")
        return
    else:
        app_logger.info("All or some export files missing. Proceeding with Step 1.")

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

    app_logger.info("Data Preparation completed")
    pass

def step_2_generate_gpt_responses():
    
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
            with open(f"./_temp/step2_skipped_{get_curr_et_datetime_str()}.txt", "w") as file:
                file.write(f"The follow rows are skipped because they were already processed.\n")
                for item in skipped_rows:        
                    file.write(f"{str(item)}\n")        

    # Main Program
    app_logger.info("Beginning Step 2: GPT Responses Generation")
    
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
    
    app_logger.info("GPT Responses Generation completed")

    pass

def step_3_extract_info(response_data_file):

    def load_response_data(filename):
        if os.path.exists(filename):
            # print(f"{filename} found. Loading data...")
            with open(filename, 'r') as file:
                data = json.load(file)
                app_logger.info(f"{filename} loaded. {len(data)} records found.")
            return data
        else:
            app_logger.error(f"Step 3 failed.")
            raise FileNotFoundError(f"Error: {filename} not found.")

    # F(x): Extract ICD probabilities from tokens
    def extract_icd_probabilities(logprobs, debug=False):
        """
        Extracts ICD-10 codes and their associated probabilities from a list of tokens and log probabilities.

        This function iterates over the list of tokens and log probabilities, concatenating tokens together 
        and checking if they match the pattern of an ICD-10 code. If a match is found, it calculates the mean 
        linear probability of the ICD-10 code and packages the ICD-10 code, mean linear probability, and 
        associated tokens and log probabilities into a dictionary. It then appends this dictionary to a list 
        of parsed ICD-10 codes.

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

    # F(x): Convert list of ICD10 into individual columns
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
            
        tmp = tmp.drop(columns=['logprobs'])
        tmp = tmp.stack().reset_index(drop=True) # convert to 1 row
        tmp = tmp.reindex(range(pairs*2), axis=1) # pad to fill PAIRS*2 columns
        

        return tmp

    # F(x): Get export filenames based on input filename
    def generate_export_filename(file_path, type="first") -> tuple[str, str]:
        '''
        Takes a file path as input and returns a tuple containing the names of the parsed JSON and CSV files.

        Parameters:
            file_path (str): The full path of the input file.

        Returns:
            tuple[str(json), str(csv)]: A tuple containing the names of the parsed JSON and CSV files.
        '''
        directory = os.path.dirname(file_path)
        file = os.path.basename(file_path)
        
        temp = file.split(".json")[0]
        temp = temp[2:]
        
        if type == "first":
            return f"{directory}/03{temp}_parsed_first_ICD.csv"
        
        return f"{directory}/03{temp}_parsed_sorted_ICD.csv"
        
    # ##############################################################
    # Main Program

    app_logger.info("Beginning Step 3: Information Extraction")

        
    data_storage = load_response_data(response_data_file)
    df = pd.DataFrame(data_storage).T

    global export_first_ICD_CSV_file
    if export_first_ICD_CSV_file is None:
        export_first_ICD_CSV_file = generate_export_filename(response_data_file, type="first")
        app_logger.info(f"Exporting filename undefined. Exporting to default filename {export_first_ICD_CSV_file}.")

    # Get unrecognized colnames
    required_colnames = ['uid', 'rowid', 'param_model', 'param_temperature',
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
    icd_column_names_mapping = {i: f"cause{i // 2 + 1}_icd10" for i in range(PAIRS)}

    # cause1...5 are filled in the order they appear in the logprobs
    # parsed_first_icd10_df = df.merge(df.output_probs.apply(lambda x: output_icds_to_cols(x, sort_probs=False)).rename(columns=icd_column_names_mapping), left_index=True, right_index=True)
    # cause1...5 are filled in the order they appear in the logprobs

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

    # Rename the columns using the mapping
    parsed_first_icd10_df = parsed_first_icd10_df.rename(columns=column_mapping)

    app_logger.info("Organizing columns...")
    export_columns = []
    export_columns += ['rowid']
    export_columns += list(icd_column_names_mapping.values())
    export_columns += [
                        'output_created',
                        'param_model',
                        'param_system_prompt' , 
                        'param_user_prompt', 
                        'output_usage_completion_tokens', 
                        'output_usage_prompt_tokens', 
                        'output_msg',
                        'icd10_count',
                        'output_probs',
                    ]

    if not DROP_EXCESS_COLUMNS:
        export_columns += extra_colnames
        
    if not DROP_RAW:
        export_columns += ['output']


    # Show only relevant columns in the final dataframe
    export_parsed_first_icd10_df = parsed_first_icd10_df[export_columns]

    app_logger.info(f"Finished parsing ICD-10 codes. Final dataframe shape: {export_parsed_first_icd10_df.shape}")
    app_logger.info(f"Exporting to: {export_first_ICD_CSV_file}")

    create_dir(export_first_ICD_CSV_file)

    export_parsed_first_icd10_df.to_csv(export_first_ICD_CSV_file, index=False)

    app_logger.info("Information Extraction completed")

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

# Function to get current time in string YYMMDD_HHMMSS format
    def get_curr_et_datetime_str() -> str:
        return datetime.datetime.now(tz=TIMEZONE).strftime("%y%m%d_%H%M%S")
    
def main():
    # load logging preference
    # logging.getLogger().setLevel(logging.WARNING)
    logging_process()
    
    step_1_prepare_data()
    step_2_generate_gpt_responses()
    step_3_extract_info(response_data_file=IMPORT_02_JSON_DATA)

if __name__ == "__main__":
    main()