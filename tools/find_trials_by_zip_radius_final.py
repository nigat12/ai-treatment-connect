import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import ast  # For safely evaluating string representation of list/tuples
import time
import json
import os

# --- Configuration ---
PREPROCESSED_EXCEL_FILE_PATH = "trials_filtered_with_coordinates.xlsx"
COORDINATES_COLUMN_NAME = "location_coordinates"  # Column with pre-geocoded trial coordinates
# This is for displaying trial info, adjust if you have a different ID column or want other info
# If you don't have a specific ID, it can be removed or adapted to use other columns for display.
DISPLAY_TRIAL_ID_COLUMN = "NCT Number" # Example: use NCT Number for display
DISPLAY_TRIAL_TITLE_COLUMN = "Study Title" # Example: use Study Title

DEFAULT_COUNTRY_CODE = "US"
# IMPORTANT: Change to your app name/email. Nominatim requires a unique user agent.
NOMINATIM_USER_AGENT = "ai_health_connect/1.0" # PLEASE UPDATE THIS

GEOCODE_CACHE_FILE_PATH = "geocode_cache.json" # Shared with pre-processing script
TEMP_GEOCODE_CACHE_FILE_PATH = "geocode_cache.tmp.json"

API_REQUEST_DELAY = 1.05 # Seconds to wait between API calls
API_TIMEOUT = 15 # Seconds to wait for API response

# --- Geolocator Initialization (for user's zip) ---
user_geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT)

# --- Persistent Geocode Cache Functions (Shared with pre-processor, simplified for user zip) ---
def load_persistent_geocode_cache(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception: # Broad exception for loading
            pass # If error, will proceed as if cache is empty
    return {}

def save_persistent_geocode_cache(cache_data, target_path, temp_path):
    try:
        with open(temp_path, 'w') as f:
            json.dump(cache_data, f, indent=4)
        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename(temp_path, target_path)
    except Exception: # Broad exception for saving
        pass # Non-critical if cache save fails here

def get_user_zip_coords_with_cache(zip_code, country_code, persistent_cache):
    """ Geocodes user's zip, using and updating the persistent cache. """
    cache_key = f"{zip_code}_{country_code}".lower()
    
    if cache_key in persistent_cache and persistent_cache[cache_key] is not None:
        print(f"  Cache hit for your zip '{zip_code}'. Using cached coordinates.")
        return persistent_cache[cache_key]

    print(f"  Geocoding your zip '{zip_code}' ({country_code}) via API...")
    try:
        time.sleep(API_REQUEST_DELAY)
        location_query = f"{zip_code}, {country_code}"
        location_data = user_geolocator.geocode(location_query, timeout=API_TIMEOUT)
        if location_data:
            coords = (location_data.latitude, location_data.longitude)
            persistent_cache[cache_key] = coords # Update cache
            save_persistent_geocode_cache(persistent_cache, GEOCODE_CACHE_FILE_PATH, TEMP_GEOCODE_CACHE_FILE_PATH)
            return coords
        else:
            print(f"    -> Warning: Could not geocode your zip code: {zip_code}")
            # Optionally cache this failure if desired: persistent_cache[cache_key] = None
            return None
    except (GeocoderTimedOut, GeocoderUnavailable) as e:
        print(f"    -> Warning: Geocoding service issue for your zip code {zip_code}: {e}")
        return None
    except Exception as e:
        print(f"    -> Warning: An error occurred while geocoding your zip {zip_code}: {e}")
        return None

# --- Helper Functions ---
def parse_trial_coordinates_from_excel(coord_string_val):
    """
    Safely parses a string representation of a list of coordinate tuples from Excel.
    e.g., "[(40.7128, -74.0060), (34.0522, -118.2437)]" or "[]"
    """
    if isinstance(coord_string_val, list): # Already a list
        return coord_string_val
    if pd.isna(coord_string_val) or not isinstance(coord_string_val, str) or not coord_string_val.strip():
        return [] # Return empty list for empty/NaN/invalid string
    try:
        parsed_list = ast.literal_eval(coord_string_val)
        if isinstance(parsed_list, list):
            # Basic validation: ensure items are tuples of two numbers
            valid_coords = []
            for item in parsed_list:
                if isinstance(item, (tuple, list)) and len(item) == 2 and \
                   all(isinstance(num, (int, float)) for num in item):
                    valid_coords.append(tuple(item)) # Ensure it's a tuple
                # else:
                    # print(f"    Skipping invalid item in coordinate list: {item}") # For debugging
            return valid_coords
        return [] # Parsed, but not a list of valid coordinate structures
    except (ValueError, SyntaxError, TypeError):
        # print(f"  Warning: Could not parse coordinate string: '{str(coord_string_val)[:50]}...'. Returning empty list.")
        return []


# --- Main Search Function ---
def main_search():
    print("--- Clinical Trial Locator (using pre-processed data) ---")
    
    # Load persistent geocode cache (for user's zip)
    persistent_geocode_cache = load_persistent_geocode_cache(GEOCODE_CACHE_FILE_PATH)

    # 1. Load the pre-processed Excel file
    try:
        df_trials = pd.read_excel(PREPROCESSED_EXCEL_FILE_PATH)
        print(f"Successfully loaded pre-processed trial data from '{PREPROCESSED_EXCEL_FILE_PATH}' ({len(df_trials)} trials).")
    except FileNotFoundError:
        print(f"FATAL: Pre-processed Excel file '{PREPROCESSED_EXCEL_FILE_PATH}' not found.")
        print("Please run the pre-processing script (e.g., preprocess_trial_data_final.py) first.")
        return
    except Exception as e:
        print(f"FATAL: Error reading pre-processed Excel file: {e}")
        return

    if COORDINATES_COLUMN_NAME not in df_trials.columns:
        print(f"FATAL: Required coordinates column '{COORDINATES_COLUMN_NAME}' not found in the pre-processed Excel file.")
        print("Ensure the pre-processing script ran correctly and generated this column.")
        print(f"Available columns: {df_trials.columns.tolist()}")
        return

    # 2. Get user input
    user_zip = ""
    while not user_zip: # Basic validation: ensure zip is not empty
        user_zip = input("Enter your zip code: ").strip()
        if not user_zip:
            print("Zip code cannot be empty. Please try again.")
            
    radius_miles = -1.0
    while radius_miles <= 0:
        try:
            radius_miles_str = input("Enter search radius in miles: ")
            radius_miles = float(radius_miles_str)
            if radius_miles <= 0:
                print("Radius must be a positive number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number for the radius.")

    # 3. Geocode user's zip code (using cache)
    print("\nGeocoding your location...")
    user_coords = get_user_zip_coords_with_cache(user_zip, DEFAULT_COUNTRY_CODE, persistent_geocode_cache)
    if not user_coords:
        print(f"Could not determine coordinates for your zip code '{user_zip}'. Exiting.")
        return
    print(f"Your geocoded location ({user_zip}): Latitude {user_coords[0]:.4f}, Longitude {user_coords[1]:.4f}")

    # 4. Find nearby trials using pre-geocoded trial data
    nearby_trials_info = []
    print(f"\nSearching for trials with locations within {radius_miles} miles of {user_zip}...")

    for index, trial_row in df_trials.iterrows():
        # Get pre-geocoded coordinates string for the trial and parse it
        raw_trial_coords_data = trial_row[COORDINATES_COLUMN_NAME]
        trial_site_coordinates_list = parse_trial_coordinates_from_excel(raw_trial_coords_data)

        if not trial_site_coordinates_list: # No valid geocoded locations for this trial
            continue

        min_distance_for_this_trial = float('inf')
        closest_site_coords_for_this_trial = None

        for trial_site_coords_tuple in trial_site_coordinates_list:
            try:
                # Ensure trial_site_coords_tuple is indeed a tuple (or list) of two numbers
                if not (isinstance(trial_site_coords_tuple, (tuple, list)) and len(trial_site_coords_tuple) == 2 and
                        all(isinstance(c, (int, float)) for c in trial_site_coords_tuple)):
                    # print(f"  Skipping invalid coordinate format in trial data: {trial_site_coords_tuple}") # Debug
                    continue
                
                distance = geodesic(user_coords, trial_site_coords_tuple).miles
                
                if distance < min_distance_for_this_trial:
                    min_distance_for_this_trial = distance
                    closest_site_coords_for_this_trial = trial_site_coords_tuple

            except Exception as e:
                # This might happen if coordinates are malformed despite parsing, though less likely with validation
                print(f"  Warning: Error calculating distance for trial at index {index} between {user_coords} and {trial_site_coords_tuple}: {e}")
                continue 
        
        # Check if the closest site for this trial is within the radius
        if min_distance_for_this_trial <= radius_miles:
            trial_info_to_store = {
                "index": index, # Store original index for reference
                "matched_trial_site_coords": closest_site_coords_for_this_trial,
                "distance_miles": min_distance_for_this_trial,
            }
            # Add other displayable info from the trial_row
            if DISPLAY_TRIAL_ID_COLUMN in trial_row and pd.notna(trial_row[DISPLAY_TRIAL_ID_COLUMN]):
                trial_info_to_store['id'] = trial_row[DISPLAY_TRIAL_ID_COLUMN]
            if DISPLAY_TRIAL_TITLE_COLUMN in trial_row and pd.notna(trial_row[DISPLAY_TRIAL_TITLE_COLUMN]):
                trial_info_to_store['title'] = trial_row[DISPLAY_TRIAL_TITLE_COLUMN]
                
            trial_info_to_store['Locations'] = trial_row["Locations"]
            nearby_trials_info.append(trial_info_to_store)
            # For debugging, print immediately:
            # trial_display_id = trial_info_to_store.get('id', f"Trial at Excel Row {index+2}")
            # print(f"  Found Match: {trial_display_id} - Closest site is {min_distance_for_this_trial:.2f} miles away.")


    # 5. Display results
    print("\n--- Search Results ---")
    if nearby_trials_info:
        # Sort by distance (optional, but nice for users)
        nearby_trials_info.sort(key=lambda x: x['distance_miles'])
        
        print(f"Found {len(nearby_trials_info)} clinical trial(s) with at least one location within {radius_miles} miles of {user_zip}:")
        for i, info in enumerate(nearby_trials_info):
            trial_display_id = info.get('id', f"Trial at Excel Row {info['index']+2}") # Fallback to row if ID missing
            trial_title = info.get('title', "N/A")
            

            print(f"\nResult {i+1}:")
            print(f"  Trial Identifier: {trial_display_id}")
            if trial_title != "N/A":
                print(f"  Title: {trial_title}")
            print(f"  Closest Matched Location Coordinates for this Trial: Lat {info['matched_trial_site_coords'][0]:.4f}, Lon {info['matched_trial_site_coords'][1]:.4f}")
            print(f"  Distance to this location: {info['distance_miles']:.2f} miles")
            print(f" Original locations text: {info['Locations']}")
            # You can add more details from df_trials.iloc[info['index']] if needed
            # e.g., print(f"  Status: {df_trials.at[info['index'], 'Study Status']}")
    else:
        print(f"No clinical trials found with locations within {radius_miles} miles of {user_zip}.")
    
    print(f"\nShared geocode cache currently has {len(persistent_geocode_cache)} entries.")


if __name__ == "__main__":
    if NOMINATIM_USER_AGENT == "your_unique_app_name_or_email_locator_final/1.0":
        print("\n!!! CRITICAL WARNING !!!")
        print("Please update the 'NOMINATIM_USER_AGENT' variable in the script with your unique application name or email.")
        print("Using the default agent may lead to your requests being blocked by Nominatim.")
        print("Exiting script. Please update and re-run.\n")
    else:
        main_search()
        print("\n--- Search Complete ---")