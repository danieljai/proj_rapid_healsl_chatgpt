import os
import pandas as pd
import numpy as np
import json
import re
import logging

OUTPUT_DIR = "output"
PROCESS_FILE = "s3_output_gpt3_full.csv"
VERSION = "v2b"
SPLIT_COLNAME = "round"

out_file_template = "healsl_ROUND_rapid_MODELNAME_VERSION.csv"

script_dir = os.path.dirname(__file__)

class COLOUR:
    yellow = '\033[93m'
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

def main(input_file, output_template):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"{COLOUR.green}[TOOLS] Begin splitting data by round{COLOUR.end}")

    if "gpt3" in input_file:
        model_name = "gpt3"
    elif "gpt4" in input_file:
        model_name = "gpt4"
    
    filename = out_file_template.replace("VERSION", VERSION)
    filename = filename.replace("MODELNAME", model_name) 
        

    df = pd.read_csv(input_file)
    
    split_unique_val_list = df[SPLIT_COLNAME].unique()
    
    logging.info(f"Input file: {input_file}")
    logging.info(f"Directory: {OUTPUT_DIR}")
    logging.info(f"Model: {model_name}")
    logging.info(f"Version: {VERSION}")
    logging.info(f"Split by {SPLIT_COLNAME}: {split_unique_val_list}")
    
    
    for curr_split in split_unique_val_list:
        split_df = df[df[SPLIT_COLNAME] == curr_split]
        
        logging.info(f"Preparing round: {curr_split} Shape: {COLOUR.cyan}{split_df.shape}{COLOUR.end}")
        
        
        
        split_filename = filename.replace("ROUND", curr_split)
        split_filename = os.path.join(script_dir, OUTPUT_DIR, split_filename)
        logging.info(f"Export to: {COLOUR.yellow}{split_filename}{COLOUR.end}")
        split_df.to_csv(split_filename, index=False)
        
        

    # rd1_df = df[df['round'] == "rd1"]
    # rd2_df = df[df['round'] == "rd2"]

    # logging.info(f"round 1 shape: {rd1_df.shape}")
    # logging.info(f"round 2 shape: {rd2_df.shape}")

    # rd1_filename = out_file_template.replace("ROUND", "rd1")
    # rd1_filename = dir_path + "/" + rd1_filename
    # logging.info(f"Exporting round 1 data to: {COLOUR.yellow}{rd1_filename}{COLOUR.end}")
    # rd1_df.to_csv(rd1_filename, index=False)

    # rd2_filename = out_file_template.replace("ROUND", "rd2")
    # rd2_filename = dir_path + "/" + rd2_filename
    # logging.info(f"Exporting round 2 data to: {COLOUR.yellow}{rd2_filename}{COLOUR.end}")
    # rd2_df.to_csv(rd2_filename, index=False)

    logging.info(f"{COLOUR.green}[TOOLS] Data splitting complete{COLOUR.end}")
    
if __name__ == "__main__":
    # input_file = "./output/s3_output_gpt3_full.csv"
    input_file = os.path.join(script_dir, OUTPUT_DIR, PROCESS_FILE)
    main(
        input_file=input_file,
        output_template=out_file_template
    )