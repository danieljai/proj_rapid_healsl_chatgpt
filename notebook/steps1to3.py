import os
import pandas as pd
import numpy as np
from openai import OpenAI
import datetime
import pytz
import logging

# Global Variables
TIMEZONE = pytz.timezone('America/Toronto')


# STEP 1 Parameters
# Set PROPORTIONAL_SAMPLING to True to sample the dataset proportionally, and False retain the entire dataset
# N_REPEAT_RESPONSES and N_SAMPLES are only used when PROPORTIONAL_SAMPLING is True
PROPORTION_SAMPLING = False
N_REPEAT_RESPONSES = 10
N_SAMPLES = 100  # Replace X with the desired number of values to select

STEP1_IMPORT_REL_DIR = "data_202402/"
STEP1_EXPORT_FULL_FILE = "step1_full_export.csv"
STEP1_EXPORT_SAMPLE_FILE = "step1_sample_export.csv"


def step1():
    logging.info("Preprocessing data.")

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

            logging.info(f"Reading: {r}, {a} - {str(merged_df.shape[0])} records.")
            # print(f"Sample of merged_df {merged_df.shape}:")
            # display(merged_df.sample(5))
            
            merged_all_df = pd.concat([merged_all_df, merged_df])

    logging.info("Merging dataframes complete.")
    logging.info(f"Total number of rows: {merged_all_df.shape[0]}")

    def reorder_df(df):
        return df[['uid'] + [col for col in df.columns if col != 'uid']]

    def get_samples(df, n=100, repetition=10):
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
            logging.info(f"{sample}: {sampling_frac[sample]} records")

        # Sort dict from largest group to smallest
        sorted_sample_ids = dict(sorted(sample_ids.items(), key=lambda item: len(item[1]), reverse=True))

        # Get the actual samples count
        sample_values_count = len([item for subitem in sorted_sample_ids.values() for item in subitem])

        # If sample count is more than N_SAMPLES required, remove the excess samples
        # starting from the group with the most samples and more than 10 samples
        if sample_values_count > n:
            excess = sample_values_count - n
            logging.info(f"Sampled more than {n} needed. Removing excess samples.")

            for _ in range(excess):
                for key in sorted_sample_ids:

                    if len(sorted_sample_ids[key]) > 10:
                        sorted_sample_ids[key].pop()
                        break
        else:
            logging.info(f"There are {sample_values_count} samples. No need to remove any samples.")

        # Flatten the sample dictionary to a list of rowids
        sample_ids_list = [item for sublist in sorted_sample_ids.values() for item in sublist]
        logging.debug(f"Sampled {len(sample_ids_list)} rows.")

        # Compile a dataframe according to sampled rowids
        random_rowids = temp_df[temp_df['rowid'].isin(sample_ids_list)]
        logging.debug(f"Compile dataframe according to sampled rowids")

        # Construct a unique id rowid_repetition and append to the dataframe
        list_of_repeated_dfs = [random_rowids.assign(uid=random_rowids['rowid'].astype(str) + "_" + str(r)) for r in range(repetition)]
        final_sample_df = pd.concat(list_of_repeated_dfs)
        logging.debug(f"Create uid for sampling repetition")
        
        # done with 'group', so drop it
        final_sample_df = final_sample_df.drop(columns=['group'])
        logging.debug(f"Drop 'group' column")

        # Reorder uid as first column for presentation
        final_sample_df = reorder_df(final_sample_df)
        logging.debug(f"Reorder dataframe columns")

        # Fill sex_code with empty string
        final_sample_df['sex_cod'] = final_sample_df['sex_cod'].fillna("")
        logging.debug(f"Fill empty string in NaN values")

        logging.info(f"Sample Dataframe complete. Returning dataframe.")
        logging.debug(f"Final sample dataframe shape: {final_sample_df.shape}")
        return final_sample_df
    
    def get_full_df(df):
        logging.debug(f"Creating full dataframe")

        logging.debug(f"Duplicate dataframe")
        temp_df = df.copy()
        logging.debug(f"Create uid for full dataframe")
        temp_df = temp_df.assign(uid=temp_df['rowid'])
        logging.debug(f"Reorder dataframe columns")
        temp_df = temp_df[['uid'] + [col for col in temp_df.columns if col != 'uid']]
        logging.debug(f"Fill empty string in NaN values")
        temp_df['sex_cod'] = temp_df['sex_cod'].fillna("")

        logging.info(f"Full Dataframe complete. Returning dataframe.")
        logging.debug(f"Final full dataframe shape: {temp_df.shape}")
        return temp_df
    
    final_sample_df = get_samples(merged_all_df, n=N_SAMPLES, repetition=N_REPEAT_RESPONSES)

    final_full_df = get_full_df(merged_all_df)

    logging.info(f"Exporting sample to {STEP1_EXPORT_SAMPLE_FILE}")
    final_sample_df.to_csv(STEP1_EXPORT_SAMPLE_FILE, index=False)

    logging.info(f"Exporting full to {STEP1_EXPORT_FULL_FILE}")
    final_full_df.to_csv(STEP1_EXPORT_FULL_FILE, index=False)
    pass

def step2():
    pass

def step3():
    pass

def logging_process():
    # Create a logger and set its level
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create a handler with a formatter and add it to the logger
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(fmt='%(asctime)s:%(msecs)03d[%(funcName)s]: %(message)s', datefmt='%H:%M:%S')
    )

    logger.addHandler(handler)

def check_file_exists(file):
    return os.path.isfile(file)

def main():
    # load logging preference
    logging_process()

    logging.info("Starting process.")
    logging.info(f"Searching for '{STEP1_EXPORT_FULL_FILE}' and '{STEP1_EXPORT_SAMPLE_FILE}'.")
    logging.info(f"If either file is missing, Step 1 will be run, overwriting the existing files.")
    if not check_file_exists(STEP1_EXPORT_FULL_FILE) or not check_file_exists(STEP1_EXPORT_SAMPLE_FILE):
        logging.info("Step 1 files not found. Running Step 1.")
        step1()
    else:
        logging.info("Step 1 files found. Skipping Step 1.")

    step2()
    step3()

if __name__ == "__main__":
    main()