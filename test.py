# ct_search_auto_test_v4_metadata_fixed.py
import requests
import argparse
import json
import logging
from urllib.parse import urlencode, quote_plus
import time

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
CTGOV_API_V2_BASE_URL = "https://clinicaltrials.gov/api/v2/" # Base for different endpoints

# --- Metadata Fetching (FIXED) ---
def get_api_metadata(endpoint="studies/metadata"):
    """Fetches metadata to find correct field names and values."""
    url = f"{CTGOV_API_V2_BASE_URL}{endpoint}"
    print(f"\n--- Attempting to fetch metadata from: {url} ---")
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        metadata = response.json()
        print(f"--- Metadata Received from {endpoint} (Type: {type(metadata)}) ---")

        # Handle dictionary response (likely from /studies/metadata)
        if isinstance(metadata, dict):
            print("\n--- Relevant Sections from Dictionary Metadata (if found) ---")
            # Look for status module
            status_module = metadata.get('protocolSection', {}).get('statusModule', {})
            if status_module and isinstance(status_module, dict):
                 print("\nStatus Module Properties:")
                 for prop in status_module.get('properties', []):
                     if prop.get('query'): # Only show queryable fields
                         print(f"  Name: {prop.get('name')}, Type: {prop.get('type')}, Queryable: {prop.get('query')}")
                         if 'enum' in prop:
                             print(f"    Enum Values: {prop.get('enum')}")

            # Look for design module (for Phase, StudyType)
            design_module = metadata.get('protocolSection', {}).get('designModule', {})
            if design_module and isinstance(design_module, dict):
                print("\nDesign Module Properties:")
                for prop in design_module.get('properties', []):
                     # Check nested properties too (like PhaseList.Phase)
                     if prop.get('query'):
                         print(f"  Name: {prop.get('name')}, Type: {prop.get('type')}, Queryable: {prop.get('query')}")
                         if 'enum' in prop:
                              print(f"    Enum Values: {prop.get('enum')}")
                     # Check for nested queryable fields like Phase
                     if prop.get('name') == 'PhaseList' and isinstance(prop.get('properties'), list):
                          phase_prop = next((p for p in prop['properties'] if p.get('name') == 'Phase'), None)
                          if phase_prop and phase_prop.get('query'):
                              print(f"  Name: Phase (within PhaseList), Type: {phase_prop.get('items', {}).get('type')}, Queryable: {phase_prop.get('query')}")
                              if 'enum' in phase_prop.get('items', {}):
                                  print(f"    Enum Values: {phase_prop['items']['enum']}")


        # Handle list response (likely from /studies/search-areas)
        elif isinstance(metadata, list):
            print("\n--- Queryable Fields from List Metadata (Search Areas) ---")
            found_status = False
            found_type = False
            for item in metadata:
                if isinstance(item, dict):
                    name = item.get('name')
                    queryable = item.get('query')
                    # Check for potential Status and Type fields
                    if queryable:
                        if 'status' in name.lower(): # Look for 'status' in the name
                            print(f"\nPotential Status Field:")
                            print(json.dumps(item, indent=2))
                            found_status = True
                        if 'type' in name.lower() and 'study' in name.lower(): # Look for 'study' and 'type'
                             print(f"\nPotential Study Type Field:")
                             print(json.dumps(item, indent=2))
                             found_type = True
                        if name == 'Phase': # Phase worked, let's confirm
                             print(f"\nConfirmed Phase Field:")
                             print(json.dumps(item, indent=2))

            if not found_status: print("\nWARNING: Could not definitively identify a queryable Status field in search areas.")
            if not found_type: print("\nWARNING: Could not definitively identify a queryable Study Type field in search areas.")

        else:
            print("Received metadata in unexpected format.")

        print("\n--- End Metadata View ---")
        return metadata

    except requests.exceptions.RequestException as e:
        print(f"Error fetching metadata: {e}")
    except json.JSONDecodeError:
        print("Error decoding metadata JSON.")
    except Exception as e:
        print(f"An unexpected error occurred fetching metadata: {e}")
        logging.error("Metadata fetching error", exc_info=True) # Log full traceback for this error
    return None

# --- Filters to Test (Start with simple ones based on previous results) ---
# We KNOW Phase works like this. Status and Type are the problems.
TEST_FILTER_PHASE = '(Phase:"Phase 2" OR Phase:"Phase 3")'
# We need to FIND the correct Status and Type filters from the metadata output
TEST_FILTER_STATUS_TBD = "FIELD_NAME_FROM_METADATA:VALUE_FROM_METADATA" # Placeholder
TEST_FILTER_TYPE_TBD = "FIELD_NAME_FROM_METADATA:VALUE_FROM_METADATA"  # Placeholder


DEFAULT_FIELDS = ["NCTId","BriefTitle","OverallStatus","Phase","StudyType","Condition"]

# --- Helper --- (clean_text function remains the same)
def clean_text(text, max_len=100):
    if not text: return "N/A"; text = str(text).replace('\n', ' ').replace('\r', '').strip()
    if len(text) > max_len: return text[:max_len-3] + "..."; return text if text else "N/A"

# --- Main Search Function --- (search_trials function remains the same)
def search_trials(search_terms, filter_expression, limit, fields):
    results_found = False;
    if not search_terms and not filter_expression: logging.error("Cannot search without terms or filter."); print("Error: No terms/filter."); return False
    user_query_part = " ".join(search_terms)
    if user_query_part and filter_expression: full_query_term = f"({user_query_part}) AND ({filter_expression})"
    elif user_query_part: full_query_term = user_query_part
    elif filter_expression: full_query_term = filter_expression
    else: logging.error("Logic error."); return False
    logging.info(f"Constructed Full Query for query.term: '{full_query_term}'"); params = {'query.term': full_query_term, 'pageSize': limit, 'format': 'json', 'fields': ",".join(fields)}
    try:
        request = requests.Request('GET', f"{CTGOV_API_V2_BASE_URL}studies", params=params).prepare(); logging.info(f"Requesting URL: {request.url}"); response = requests.Session().send(request, timeout=30)
        if response.status_code == 400:
             logging.error(f"API Error (400 Bad Request) for Query: '{full_query_term}'");
             try: error_data = response.json(); error_message = error_data.get('message', str(error_data)); logging.error(f"API Error Details: {error_message}"); print(f"Result: API Error (400): Bad Request. Details: {error_message}")
             except json.JSONDecodeError: logging.error("Could not decode error body."); print("Result: API Error (400): Bad Request.")
             return False
        elif response.status_code == 404: logging.warning(f"No studies found (404) for Query: '{full_query_term}'"); print("Result: --- No studies found (404) ---"); return False
        response.raise_for_status(); data = response.json(); studies = data.get('studies', []); total_count = data.get('totalCount', len(studies))
        if not studies: logging.warning(f"API returned 200 OK but no studies found for Query: '{full_query_term}'"); print("Result: --- No studies found (200 OK but empty list) ---"); return False
        print(f"Result: SUCCESS - Found {len(studies)} studies (Total: {total_count})"); results_found = True
        for i, study in enumerate(studies):
            protocol = study.get('protocolSection', {}); ident_module = protocol.get('identificationModule', {}); status_module = protocol.get('statusModule', {}); design_module = protocol.get('designModule', {}); cond_module = protocol.get('conditionsModule', {})
            nct_id = ident_module.get('nctId', 'N/A'); title = study.get('briefTitle', 'N/A'); status = status_module.get('overallStatus', 'N/A'); phases_list = design_module.get('phases', ['N/A']); phases = ', '.join(phases_list); study_type = design_module.get('studyType', 'N/A'); conditions_list = cond_module.get('conditionList', {}).get('condition', ['N/A']); conditions = ', '.join(conditions_list)
            print(f"  {i+1}. {nct_id} - {clean_text(title)} (Status: {status}, Phase: {phases}, Type: {study_type}, Condition: {clean_text(conditions, 50)})")
        print("-" * 40); return results_found
    except requests.exceptions.Timeout: logging.error("API request timed out."); print("Result: Error - Request timed out.")
    except requests.exceptions.HTTPError as e: logging.error(f"API request failed with HTTP error: {e.response.status_code}"); print(f"Result: Error - HTTP {e.response.status_code} - {e.response.reason}")
    except requests.exceptions.RequestException as e: logging.error(f"API request failed: {e}", exc_info=False); print(f"Result: Error - Could not connect to API.")
    except json.JSONDecodeError as e: logging.error(f"Failed to decode API JSON response: {e}", exc_info=True); print("Result: Error - Invalid response received.")
    except Exception as e: logging.error(f"An unexpected error occurred: {e}", exc_info=True); print(f"Result: An unexpected error occurred: {e}")
    return False

# --- Script Execution ---
if __name__ == "__main__":
    # 1. Fetch and display metadata first - USER ACTION REQUIRED HERE
    print(">>> ACTION NEEDED: Please examine the metadata output below <<<")
    get_api_metadata("studies/metadata")
    get_api_metadata("studies/search-areas")
    print(">>> ACTION NEEDED: Identify the correct QUERYABLE field names and ENUM values for:")
    print("    1. Study Recruitment Status (e.g., is it 'OverallStatus', 'Status', etc.? What is the exact value for 'Recruiting'?)")
    print("    2. Study Type (e.g., is it 'StudyType', 'Type', etc.? What is the exact value for 'Interventional'?)")
    print("    3. Phase (We know 'Phase' and '\"Phase 2\"' work, confirm enum values)")
    print(">>> ACTION NEEDED: Update the 'TEST_FILTER_STATUS_TBD' and 'TEST_FILTER_TYPE_TBD' variables below <<<")
    print("-------------------------------------------------------------")

    # 2. Define test scenarios using PLACEHOLDERS initially
    #    *** USER MUST UPDATE THE _TBD filters below based on metadata ***
    TEST_FILTER_STATUS_TBD = "OverallStatus:RECRUITING" # Replace with correct field:value from metadata
    TEST_FILTER_TYPE_TBD = "StudyType:INTERVENTIONAL"    # Replace with correct field:value from metadata
    TEST_FILTER_PHASE = '(Phase:"Phase 2" OR Phase:"Phase 3")' # This one is likely correct

    # Build the combined filter using the (potentially updated) TBD variables
    COMBINED_FILTER_TBD = f"{TEST_FILTER_STATUS_TBD} AND {TEST_FILTER_PHASE} AND {TEST_FILTER_TYPE_TBD}"

    test_scenarios_for_filters = [
        {"id": "Test_Status_Filter", "terms": [], "filter": TEST_FILTER_STATUS_TBD},
        {"id": "Test_Type_Filter", "terms": [], "filter": TEST_FILTER_TYPE_TBD},
        {"id": "Test_Phase_Filter", "terms": [], "filter": TEST_FILTER_PHASE}, # Re-verify
        {"id": "Test_Combined_Filter", "terms": [], "filter": COMBINED_FILTER_TBD},
        {"id": "Test_Combo_Keywords_Combined", "terms": ["lung", "cancer"], "filter": COMBINED_FILTER_TBD},
    ]

    # 3. Run the tests AFTER potentially updating the TBD filters
    print("\nStarting ClinicalTrials.gov V2 API Filter Test...")
    print("(Using filters defined as TEST_FILTER_..._TBD variables)")
    print("===================================================")
    test_limit = 3

    for scenario in test_scenarios_for_filters:
        print(f"\n>>> Testing Scenario: {scenario['id']}")
        print(f"    Terms: {scenario['terms']}")
        print(f"    Filter Expression: '{scenario['filter']}'")

        search_trials(
            search_terms=scenario['terms'],
            filter_expression=scenario['filter'],
            limit=test_limit,
            fields=DEFAULT_FIELDS
        )
        time.sleep(0.5)

    print("\n===================================================")
    print("Filter Test Complete.")
    print("If filters failed ('No studies found' or 'Error 400'), adjust the _TBD variables in the script based on the metadata and re-run.")