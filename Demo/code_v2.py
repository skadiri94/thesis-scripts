import os
import glob
import yaml
import pandas as pd


def merge_yaml_to_excel(root_yaml_directory, output_excel_file, recursive=True):
    """
    Merges data from multiple YAML files found in a root directory and its
    subdirectories into a single Excel file.

    Args:
        root_yaml_directory (str): The path to the root directory containing
                                   YAML files and/or subdirectories with YAML files.
        output_excel_file (str): The name of the Excel file to create.
        recursive (bool): If True, search for YAML files in subdirectories as well.
                          This is typically what you want for multiple directories.
    """
    all_data = []

    # Define the pattern for finding .yml files
    if recursive:
        # This pattern finds *.yml in root_yaml_directory and all its subdirectories
        pattern = os.path.join(root_yaml_directory, '**', '*.yml')
        # pattern = os.path.join(root_yaml_directory, '**', '*.yaml')
    else:
        # This pattern finds *.yml only directly in root_yaml_directory
        pattern = os.path.join(root_yaml_directory, '*.yml')
        # pattern = os.path.join(root_yaml_directory, '*.yaml')

    # Find all YAML files
    # The recursive=True argument for glob.glob is essential here for '**' to work
    yaml_files = glob.glob(pattern, recursive=True)

    if not yaml_files:
        print(
            f"No YAML files found in '{root_yaml_directory}' or its subdirectories.")
        return

    print(f"Found {len(yaml_files)} YAML files. Processing...")

    # Define the expected order of columns based on your YAML structure
    # Add or remove keys as necessary. This helps maintain a consistent column order.
    expected_keys = ['id', 'name', 'kind', 'version', 'severity', 'description', 'status', 'type', 'date', 'author', 'queryFrequency', 'queryPeriod', 'triggerOperator', 'triggerThreshold', 'tactics', 'relevantTechniques', 'query',
                     'search', 'requiredDataConnectors_ids', 'requiredDataConnectors_dataTypes', 'entityMappings_json', 'metadata_source_kind', 'metadata_author_name', 'metadata_support_tier', 'metadata_categories_domains', 'data_source', 'mitre_attack_id']

    for filepath in yaml_files:
        print(f"Processing: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

                if data:
                    # Store the relative path to the YAML file for better identification
                    data['source_file'] = os.path.relpath(
                        filepath, root_yaml_directory)

                    # Add directory_name: the immediate parent directory of the file
                    data['directory_name'] = os.path.basename(
                        os.path.dirname(filepath))

                    # If 'name' is not present in YAML, derive it from the base filename
                    if 'name' not in data:
                        data['name'] = os.path.splitext(
                            os.path.basename(filepath))[0]

                    # Handle list-type fields like 'data_source'
                    # Convert list to a comma-separated string for easier Excel viewing
                    if 'data_source' in data and isinstance(data['data_source'], list):
                        data['data_source'] = ', '.join(
                            map(str, data['data_source']))

                    mitre_attack_id = None
                    if 'tags' in data and isinstance(data['tags'], dict):
                        mitre_attack_id = data['tags'].get(
                            'mitre_attack_id', None)
                        # Convert list to comma-separated string if present
                        if isinstance(mitre_attack_id, list):
                            mitre_attack_id = ', '.join(
                                map(str, mitre_attack_id))
                    data['mitre_attack_id'] = mitre_attack_id

                    all_data.append(data)
                else:
                    print(
                        f"Warning: File '{filepath}' is empty or not valid YAML.")

        except yaml.YAMLError as e:
            print(f"Error parsing YAML file '{filepath}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred with file '{filepath}': {e}")

    if not all_data:
        print("No data collected from YAML files.")
        return

    # Create a Pandas DataFrame
    df = pd.DataFrame(all_data)

    # Reorder columns: start with expected_keys, then add others, with source_file last.
    final_columns = [col for col in expected_keys if col in df.columns]
    other_columns = [
        col for col in df.columns if col not in final_columns and col != 'source_file' and col != 'directory_name']

    # Ensure 'source_file' and 'directory_name' are included, preferably at the end
    for special_col in ['directory_name', 'source_file']:
        if special_col in df.columns and special_col not in final_columns:
            final_columns.append(special_col)

    final_columns = final_columns + other_columns

    df = df[final_columns]

    # Save to Excel
    try:
        df.to_excel(output_excel_file, index=False, engine='openpyxl')
        print(f"\nSuccessfully merged YAML data into '{output_excel_file}'")
    except Exception as e:
        print(f"Error saving to Excel file: {e}")


# --- Configuration ---
# ROOT_YAML_DIR should be the parent directory that contains all the
# subdirectories with your YAML files (e.g., 'SPLUNK_DETECTIONS').
# The script will search recursively within this directory.
ROOT_YAML_DIR = 'splunk_detections'
OUTPUT_EXCEL = 'splunk_detections.xlsx'

if __name__ == "__main__":
    merge_yaml_to_excel(ROOT_YAML_DIR, OUTPUT_EXCEL)
