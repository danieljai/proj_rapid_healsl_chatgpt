import pandas as pd
import logging
import os

script_dir = os.path.dirname(__file__)

# Output directory where the input and output files are stored
OUTPUT_DIR = "output"

# Path to file after Step 3; extracting ICD10 codes
PROCESS_FILE = "s3_output_gpt3_sample.csv"


smpl_analysis_in_csv_path = os.path.join(script_dir, OUTPUT_DIR, PROCESS_FILE)

# Path to export file after sample analysis
analyzed_out_csv_filename = "healsl_rd1to2_rapid_gpt3_sample100_v1.csv"
# analyzed_out_csv_filename = PROCESS_FILE.replace(".csv", "_aggregated.csv")
analyzed_out_csv_path = os.path.join(script_dir, OUTPUT_DIR, analyzed_out_csv_filename)

# File required to convert ICD10 codes to CGHR10 titles
icd10_cghr10_map_file = './icd10_cghr10_v1.csv'


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


def same_cause_icd10(input_data):
    
    app_logger.info(f"{COLOUR.green}[TOOLS] Analysis by Aggregating {COLOUR.bold}ICD10 codes{COLOUR.end}")
    
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
    
    # Create a blank df with 10 columns, 1x to 10x
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
    
    
def main():
    logging_process()
    
    # Analyze ICD10 codes    
    icd10_df = same_cause_icd10(input_data=smpl_analysis_in_csv_path)
    
    # Analyze CGHR10 titles
    cghr10_df = same_cause_cghr10(input_data=smpl_analysis_in_csv_path)
    
    # When CGHR10 Analysis fails, export ICD10 analysis only
    if cghr10_df is None:
        app_logger.warning(f"CGHR10 Skipped. Exporting ICD10 analysis only...")
        icd10_df.to_csv(analyzed_out_csv_path.replace("_aggregated.csv", "_icd10_similarity.csv"), index=False)
        return
    
    # Combine ICD10 and CGHR10 analysis, then export
    app_logger.info(f"Combining ICD10 and CGHR10 analysis into one DataFrame...")
    final_agg_df = icd10_df.merge(cghr10_df, left_index=True, right_index=True)
    final_agg_colnames = [c for c in final_agg_df.columns if c not in ['age_group', 'round']] + ['age_group', 'round']
    final_agg_df = final_agg_df[final_agg_colnames]
    
    final_agg_df = final_agg_df.reset_index().rename(columns={'index': 'rowid', 'cause1_icd10' : 'cause_icd10', 'cause1_cghr10': 'cause_cghr10'})
    
    # print(final_agg_df)
    app_logger.info(f"Exporting similarities data to: {COLOUR.yellow}{analyzed_out_csv_path}{COLOUR.end}")
    final_agg_df.to_csv(analyzed_out_csv_path, index=False)
    
    app_logger.info(f"{COLOUR.green}[TOOLS] Analysis complete{COLOUR.end}")
    pass


if __name__ == "__main__":
    main()