import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import re
import time
import os
import json
import ast  

# --- Configuration ---
INPUT_EXCEL_FILE_PATH = "trials_filtered.xlsx"
OUTPUT_EXCEL_FILE_PATH = "trials_filtered_with_coordinates.xlsx"
TEMP_OUTPUT_FILE_PATH = "trials_filtered_with_coordinates.tmp.xlsx" 

GEOCODE_CACHE_FILE_PATH = "geocode_cache.json"
TEMP_GEOCODE_CACHE_FILE_PATH = "geocode_cache.tmp.json" 
LOCATIONS_COLUMN_NAME = "Locations"
NEW_COORDINATES_COLUMN_NAME = "location_coordinates"
DEFAULT_COUNTRY_CODE = "US"
NOMINATIM_USER_AGENT = "ai_health_connect/1.0" 

SAVE_INTERVAL = 20  
API_REQUEST_DELAY = 1.05 
API_TIMEOUT = 20 


geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT)

def load_persistent_geocode_cache(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                cache = json.load(f)
                print(f"  Successfully loaded geocode cache from '{filepath}' with {len(cache)} entries.")
                return cache
        except json.JSONDecodeError:
            print(f"  Warning: Error decoding JSON from '{filepath}'. Starting with an empty cache.")
        except Exception as e:
            print(f"  Warning: Could not load geocode cache from '{filepath}': {e}. Starting with an empty cache.")
    else:
        print(f"  Geocode cache file '{filepath}' not found. Starting with an empty cache.")
    return {}

def save_persistent_geocode_cache(cache_data, target_path, temp_path):
    """Saves the geocode cache to a JSON file atomically."""
    try:
        with open(temp_path, 'w') as f:
            json.dump(cache_data, f, indent=4)
        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename(temp_path, target_path)
        # print(f"    Geocode cache saved to '{target_path}'") # Can be verbose
    except Exception as e:
        print(f"    Error saving geocode cache to '{target_path}': {e}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e_rem:
                print(f"    Additionally, error removing temp cache file '{temp_path}': {e_rem}")

def get_coords_with_persistent_cache(zip_code, country_code, persistent_cache):
    cache_key = f"{zip_code}_{country_code}".lower()
    
    if cache_key in persistent_cache:
       
        return persistent_cache[cache_key] 
    try:
        time.sleep(API_REQUEST_DELAY)
        location_query = f"{zip_code}, {country_code}"
        location_data = geolocator.geocode(location_query, timeout=API_TIMEOUT)

        if location_data:
            coords = (location_data.latitude, location_data.longitude)
            persistent_cache[cache_key] = coords # Cache successful result
            save_persistent_geocode_cache(persistent_cache, GEOCODE_CACHE_FILE_PATH, TEMP_GEOCODE_CACHE_FILE_PATH)
            return coords
        else:
           
            return None 
    except (GeocoderTimedOut, GeocoderUnavailable) as e:
        print(f"      -> Geocode API Warning (Service Issue for {zip_code}, {country_code}): {e}")
        return None # Don't cache temporary service issues persistently by default
    except Exception as e:
        print(f"      -> Geocode API Error (Exception for {zip_code}, {country_code}): {e}")
        return None

# --- Helper Functions ---
def extract_zip_codes(location_string):
    if not isinstance(location_string, str):
        return []
    zip_pattern = r'\b\d{5}(?:-\d{4})?\b'
    return re.findall(zip_pattern, location_string)

def safe_save_dataframe(df, target_path, temp_path, columns_to_save=None):
    try:
        df_to_save = df
        if columns_to_save:
            valid_columns_to_save = [col for col in columns_to_save if col in df.columns]
            if len(valid_columns_to_save) != len(columns_to_save):
                missing = set(columns_to_save) - set(valid_columns_to_save)
                print(f"    Warning: Columns {missing} not found in DataFrame, will not be in saved Excel.")
            df_to_save = df[valid_columns_to_save]
        
        df_to_save.to_excel(temp_path, index=False, engine='openpyxl')
        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename(temp_path, target_path)
    except Exception as e:
        print(f"    Error saving DataFrame to '{target_path}': {e}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e_rem:
                print(f"    Additionally, error removing temp DataFrame file '{temp_path}': {e_rem}")

def parse_excel_coord_string(val):
    if isinstance(val, list): 
        return val
    if pd.isna(val) or not isinstance(val, str) or not val.strip():
        return pd.NA 
    try:
        parsed_list = ast.literal_eval(val)
        if isinstance(parsed_list, list):
            return parsed_list
        return pd.NA
    except (ValueError, SyntaxError, TypeError):
        return pd.NA


def preprocess_trials_data_final():
    print("--- Starting Clinical Trial Data Pre-processing (Final Version) ---")
    
    persistent_geocode_cache = load_persistent_geocode_cache(GEOCODE_CACHE_FILE_PATH)

    try:
        df_main = pd.read_excel(INPUT_EXCEL_FILE_PATH)
        print(f"Successfully loaded base data from '{INPUT_EXCEL_FILE_PATH}' ({len(df_main)} rows).")
    except FileNotFoundError:
        print(f"FATAL: Base input Excel file '{INPUT_EXCEL_FILE_PATH}' not found.")
        return
    except Exception as e:
        print(f"FATAL: Error reading base Excel file: {e}")
        return

    if LOCATIONS_COLUMN_NAME not in df_main.columns:
        print(f"FATAL: Required 'Locations' column ('{LOCATIONS_COLUMN_NAME}') not found in the base Excel file.")
        print(f"Available columns: {df_main.columns.tolist()}")
        return


    if os.path.exists(OUTPUT_EXCEL_FILE_PATH):
        print(f"Found existing output file '{OUTPUT_EXCEL_FILE_PATH}'. Attempting to load progress...")
        try:
            df_processed_previously = pd.read_excel(OUTPUT_EXCEL_FILE_PATH)
            if NEW_COORDINATES_COLUMN_NAME in df_processed_previously.columns and \
               len(df_processed_previously) == len(df_main): 
                print(f"  Successfully loaded previous progress ({len(df_processed_previously)} rows).")
                df_main[NEW_COORDINATES_COLUMN_NAME] = df_processed_previously[NEW_COORDINATES_COLUMN_NAME].apply(parse_excel_coord_string)
            else:
                reason = "row count mismatch" if len(df_processed_previously) != len(df_main) else f"missing '{NEW_COORDINATES_COLUMN_NAME}' column"
                print(f"  Warning: Previous output file has {reason}. Coordinates column will be re-initialized/re-processed as needed.")
                df_main[NEW_COORDINATES_COLUMN_NAME] = pd.NA
        except Exception as e:
            print(f"  Error loading or parsing previous output file '{OUTPUT_EXCEL_FILE_PATH}': {e}. Coordinates column will be re-initialized/re-processed as needed.")
            df_main[NEW_COORDINATES_COLUMN_NAME] = pd.NA
    else:
        print(f"No existing output file found at '{OUTPUT_EXCEL_FILE_PATH}'. Initializing coordinates column.")
        df_main[NEW_COORDINATES_COLUMN_NAME] = pd.NA 
    
    df_main[NEW_COORDINATES_COLUMN_NAME] = df_main[NEW_COORDINATES_COLUMN_NAME].astype(object)

    total_rows = len(df_main)
    rows_processed_this_session = 0
    unprocessed_rows_indices = df_main[df_main[NEW_COORDINATES_COLUMN_NAME].isna()].index
    unprocessed_rows_count_start = len(unprocessed_rows_indices)
    already_processed_count = total_rows - unprocessed_rows_count_start
    
    print(f"\nInitial Status: Total rows: {total_rows}.")
    print(f"  Rows with existing coordinate data (or empty [] from previous processing): {already_processed_count}")
    print(f"  Rows marked for coordinate processing this session (value is NA): {unprocessed_rows_count_start}")

    if unprocessed_rows_count_start == 0:
        print("All rows appear to have coordinate data. No processing needed for coordinates column.")
    else:
        print(f"\nStarting geocoding for {unprocessed_rows_count_start} rows needing coordinates...")

    for index in unprocessed_rows_indices: 
        excel_row_number = index + 2 
        print(f"\nProcessing Excel Row: {excel_row_number} (Index: {index}, Item {rows_processed_this_session + 1}/{unprocessed_rows_count_start} for this session)...")
        
        location_text = df_main.at[index, LOCATIONS_COLUMN_NAME]
        current_row_coords_list = []

        if pd.isna(location_text) or not isinstance(location_text, str) or not location_text.strip():
            print(f"  No location text for Excel Row {excel_row_number}.")
        else:
            extracted_zips = extract_zip_codes(location_text)
            if not extracted_zips:
                print(f"  No zip codes found in location string for Excel Row {excel_row_number}: '{location_text[:70].strip()}...'")
            else:
                unique_zips_for_this_row = sorted(list(set(extracted_zips)))
                print(f"  Found zips: {unique_zips_for_this_row}. Acquiring coordinates...")
                for zip_code in unique_zips_for_this_row:
                    coords = get_coords_with_persistent_cache(zip_code, DEFAULT_COUNTRY_CODE, persistent_geocode_cache)
                    if coords:
                        current_row_coords_list.append(coords)
                
                if current_row_coords_list:
                    print(f"  -> Stored coordinates for Excel Row {excel_row_number}: {len(current_row_coords_list)} locations.")
                else:
                    print(f"  -> No valid coordinates obtained for Excel Row {excel_row_number} from zips: {unique_zips_for_this_row}.")
        
        df_main.at[index, NEW_COORDINATES_COLUMN_NAME] = current_row_coords_list
        
        rows_processed_this_session += 1
        if rows_processed_this_session > 0 and rows_processed_this_session % SAVE_INTERVAL == 0 and unprocessed_rows_count_start > 0 :
            print(f"\n--- Saving intermediate progress to Excel (processed {rows_processed_this_session} rows this session) ---")
          
            safe_save_dataframe(df_main, OUTPUT_EXCEL_FILE_PATH, TEMP_OUTPUT_FILE_PATH)

            # break
    
    # --- Finalization ---
    print(f"\n--- Processing loop complete. Total rows processed for coordinates this session: {rows_processed_this_session} ---")

    final_columns_to_save = []
    original_named_cols = [col for col in df_main.columns if not col.startswith('Unnamed:') and col != NEW_COORDINATES_COLUMN_NAME]

    if LOCATIONS_COLUMN_NAME in original_named_cols:
        loc_idx = original_named_cols.index(LOCATIONS_COLUMN_NAME)
        final_columns_to_save.extend(original_named_cols[:loc_idx + 1])
        final_columns_to_save.append(NEW_COORDINATES_COLUMN_NAME) # Add new column
        final_columns_to_save.extend(original_named_cols[loc_idx + 1:])
    else:
        print(f"  Warning: '{LOCATIONS_COLUMN_NAME}' not found in original named columns. Appending '{NEW_COORDINATES_COLUMN_NAME}' at the end of named columns.")
        final_columns_to_save.extend(original_named_cols)
        final_columns_to_save.append(NEW_COORDINATES_COLUMN_NAME)


    if NEW_COORDINATES_COLUMN_NAME not in df_main.columns:
        print(f"  Error: The column '{NEW_COORDINATES_COLUMN_NAME}' was expected but not found in the DataFrame for final save. This is a bug.")
        final_columns_to_save = [col for col in original_named_cols if col in df_main.columns]


    print(f"\nPerforming final save of DataFrame to '{OUTPUT_EXCEL_FILE_PATH}' with ordered and filtered columns.")
    safe_save_dataframe(df_main, OUTPUT_EXCEL_FILE_PATH, TEMP_OUTPUT_FILE_PATH, columns_to_save=final_columns_to_save)
    
    print(f"\nFinal save of geocode cache to '{GEOCODE_CACHE_FILE_PATH}'.")
    save_persistent_geocode_cache(persistent_geocode_cache, GEOCODE_CACHE_FILE_PATH, TEMP_GEOCODE_CACHE_FILE_PATH)

    # Final check
    final_na_count = df_main[NEW_COORDINATES_COLUMN_NAME].isna().sum()
    if unprocessed_rows_count_start > 0 and final_na_count > 0 :
         print(f"  Warning: {final_na_count} rows are still marked as NA in their coordinates column after processing. This may indicate issues for these rows or an interruption.")
    elif final_na_count == 0 :
        print(f"  All rows now have coordinate data (or an empty list '[]' if no valid coordinates were found/processed).")

    print(f"\n--- Pre-processing Complete ---")

if __name__ == "__main__":
    preprocess_trials_data_final()