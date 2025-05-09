import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import re
import time

# --- Configuration ---
EXCEL_FILE_PATH = "trials.xlsx"  # Your Excel file name
LOCATIONS_COLUMN_NAME = "Locations" # Column name in Excel with trial locations
TRIAL_ID_COLUMN_NAME = "trial_id" # Or whatever uniquely identifies a trial
DEFAULT_COUNTRY_CODE = "US" # Assume US zip codes primarily
NOMINATIM_USER_AGENT = "clinical_trial_locator/1.0" # IMPORTANT: Change to your app/email

# --- Geocoding Cache and Geolocator Initialization ---
geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT)
geocode_cache = {} # To store {zip_code: (lat, lon)}

def get_coords_with_cache(zip_code, country_code=DEFAULT_COUNTRY_CODE):
    """
    Geocodes a zip code to get latitude and longitude, using a cache.
    """
    cache_key = f"{zip_code}_{country_code}"
    if cache_key in geocode_cache:
        return geocode_cache[cache_key]

    print(f"Geocoding {zip_code}...")
    try:
        # Adding country for more specific results
        # Nominatim can be slow, be patient or use a more robust service for many lookups
        time.sleep(1) # IMPORTANT: Respect Nominatim's usage policy (1 req/sec)
        location_query = f"{zip_code}, {country_code}"
        location_data = geolocator.geocode(location_query, timeout=10)

        if location_data:
            coords = (location_data.latitude, location_data.longitude)
            geocode_cache[cache_key] = coords
            return coords
        else:
            print(f"Warning: Could not geocode zip code: {zip_code}")
            geocode_cache[cache_key] = None # Cache failure to avoid retrying
            return None
    except GeocoderTimedOut:
        print(f"Warning: Geocoding service timed out for zip code: {zip_code}")
        geocode_cache[cache_key] = None
        return None
    except GeocoderUnavailable:
        print(f"Warning: Geocoding service unavailable for zip code: {zip_code}")
        geocode_cache[cache_key] = None
        return None
    except Exception as e:
        print(f"Warning: An error occurred while geocoding {zip_code}: {e}")
        geocode_cache[cache_key] = None
        return None

def extract_zip_codes(location_string):
    """
    Extracts US-style 5-digit or 9-digit (XXXXX or XXXXX-XXXX) zip codes from a string.
    Adjust regex if your zip code formats are different.
    """
    if not isinstance(location_string, str):
        return []
    # Regex for 5-digit or 5-digit+4 zip codes
    # \b ensures word boundaries (so it doesn't match part of a larger number)
    zip_pattern = r'\b\d{5}(?:-\d{4})?\b'
    return re.findall(zip_pattern, location_string)

def main():
    # 1. Load the Excel file
    try:
        df_trials = pd.read_excel(EXCEL_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Excel file '{EXCEL_FILE_PATH}' not found.")
        return
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    if LOCATIONS_COLUMN_NAME not in df_trials.columns:
        print(f"Error: Locations column '{LOCATIONS_COLUMN_NAME}' not found in the Excel file.")
        print(f"Available columns: {df_trials.columns.tolist()}")
        return
    if TRIAL_ID_COLUMN_NAME not in df_trials.columns:
        print(f"Warning: Trial ID column '{TRIAL_ID_COLUMN_NAME}' not found. Using index for identification.")


    # 2. Get user input
    user_zip = input("Enter your zip code: ").strip()
    while True:
        try:
            radius_miles = float(input("Enter search radius in miles: "))
            if radius_miles <= 0:
                print("Radius must be a positive number.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a number for the radius.")

    # 3. Geocode user's zip code
    user_coords = get_coords_with_cache(user_zip, DEFAULT_COUNTRY_CODE)
    if not user_coords:
        print(f"Could not determine coordinates for your zip code '{user_zip}'. Exiting.")
        return
    print(f"Your location ({user_zip}): {user_coords}")

    # 4. Find nearby trials
    nearby_trials_info = []

    print(f"\nSearching for trials within {radius_miles} miles of {user_zip}...")

    for index, trial_row in df_trials.iterrows():
        trial_id = trial_row.get(TRIAL_ID_COLUMN_NAME, f"Row {index+2}") # Use index if ID column is missing
        location_text = trial_row[LOCATIONS_COLUMN_NAME]

        if pd.isna(location_text): # Handle empty location cells
            continue

        trial_zip_codes = extract_zip_codes(str(location_text)) # Ensure it's a string
        if not trial_zip_codes:
            # print(f"No zip codes found for trial {trial_id} in location string: '{location_text[:50]}...'")
            continue

        found_nearby_location_for_this_trial = False
        for trial_zip in trial_zip_codes:
            trial_coords = get_coords_with_cache(trial_zip, DEFAULT_COUNTRY_CODE)
            if trial_coords:
                distance = geodesic(user_coords, trial_coords).miles
                if distance <= radius_miles:
                    # Found a location for this trial within the radius
                    trial_info = {
                        "trial_data": trial_row.to_dict(), # Store all data for the trial
                        "matched_zip": trial_zip,
                        "matched_zip_coords": trial_coords,
                        "distance_miles": distance
                    }
                    nearby_trials_info.append(trial_info)
                    found_nearby_location_for_this_trial = True
                    print(f"  Match: Trial '{trial_id}' (location zip {trial_zip}) is {distance:.2f} miles away.")
                    break # Move to the next trial once one matching location is found
            # else:
                # print(f"Could not get coordinates for trial zip {trial_zip}")
        
        # if found_nearby_location_for_this_trial:
        #     print(f" -> Trial {trial_id} is nearby.")


    # 5. Display results
    print("\n--- Search Results ---")
    if nearby_trials_info:
        print(f"Found {len(nearby_trials_info)} clinical trial(s) within {radius_miles} miles of {user_zip}:")
        for i, info in enumerate(nearby_trials_info):
            print(f"\nTrial {i+1}:")
            # Print relevant details from info['trial_data']
            # For example, if you have a 'Title' or 'Description' column:
            print(f"  ID: {info['trial_data'].get(TRIAL_ID_COLUMN_NAME, 'N/A')}")
            print(f"  Title: {info['trial_data'].get('Title', 'N/A')}") # Assuming you have a 'Title' column
            print(f"  Matched Location Zip: {info['matched_zip']}")
            print(f"  Distance: {info['distance_miles']:.2f} miles")
            # You can print more columns from info['trial_data'] as needed
            # print(f"  Full Trial Data: {info['trial_data']}")
    else:
        print(f"No clinical trials found within {radius_miles} miles of {user_zip}.")

    print(f"\nGeocoding cache size: {len(geocode_cache)} entries.")

if __name__ == "__main__":
    main()