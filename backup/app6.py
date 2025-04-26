# -*- coding: utf-8 -*-
import streamlit as st
import requests
import os
from groq import Groq, RateLimitError, APIError
from dotenv import load_dotenv
import json
import re
import urllib.parse
import logging
from datetime import datetime
import time # For simulating processing

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# ** SECURITY WARNING: Saving PII to a local text file is INSECURE and NOT SUITABLE
# ** for production environments handling real patient data. Use a secure database
# ** and appropriate encryption/access controls in a real application.
USER_DATA_FILE = "user_contact_data_demo.txt"
FDA_API_BASE_URL = "https://api.fda.gov/drug/label.json"
CTGOV_API_BASE_URL = "https://clinicaltrials.gov/api/v2/studies" # Example V2 URL

# --- Constants ---
ASSISTANT_AVATAR = "🧑‍⚕️"
USER_AVATAR = "👤"
ASSISTANT_NAME = "Assistant"
USER_NAME = "You"

# --- Available Groq Models ---
AVAILABLE_MODELS = {
    "Llama 3 8B": "llama3-8b-8192",
    "Llama 3 70B": "llama3-70b-8192",
    "Gemma 2 Instruct": "gemma2-9b-it",
    # Add other models as needed
}
DEFAULT_MODEL_DISPLAY_NAME = "Llama 3 8B"

# --- Chatbot Questions & Stages ---
STAGES = {
    "INIT": 0,
    "GET_DIAGNOSIS": 1,
    "GET_STAGE": 2,
    "GET_BIOMARKERS": 3,
    "GET_PRIOR_TREATMENT": 4,
    "GET_IMAGING": 5,
    "PROCESS_INFO_SHOW_DRUGS": 6, # Internal processing
    "ASK_CONSENT": 7,
    "GET_NAME": 8,
    "GET_EMAIL": 9,
    "GET_PHONE": 10,
    "SAVE_CONTACT_SHOW_TRIALS": 11, # Internal processing (after contact or skip)
    "SHOW_TRIALS_NO_CONSENT": 12, # Internal processing (if consent declined)
    "FINAL_SUMMARY": 13, # Display trial results
    "END": 14
}

STAGE_PROMPTS = {
    STAGES["INIT"]: "Hi there! I'm an AI assistant designed to help you explore information about cancer treatments using public data. I work alongside your physician and **do not provide medical advice.** Let's start by gathering some information. Please answer the questions as accurately as possible.\n\nFirst: 👇",
    STAGES["GET_DIAGNOSIS"]: "Q: Kindly share the specific cancer diagnosis? (e.g., Non-small cell lung cancer, Metastatic Breast Cancer)",
    STAGES["GET_STAGE"]: "Q: Any details on the stage, progression, or spread? (e.g., Stage IV, metastatic to bones, locally advanced)",
    STAGES["GET_BIOMARKERS"]: "Q: Any known biomarker details? (e.g., EGFR Exon 19 deletion, HER2 positive, PD-L1 > 50%). Please list them or type 'None'/'Unknown'.",
    STAGES["GET_PRIOR_TREATMENT"]: "Q: What treatments have been received to date? (e.g., Chemotherapy (Carboplatin/Pemetrexed), Surgery, Immunotherapy (Pembrolizumab), Radiation)",
    STAGES["GET_IMAGING"]: "Q: Are there any recent imaging results (CT, PET, MRI) showing changes or current status? (e.g., Recent CT showed stable disease, PET scan showed progression in liver)",
    STAGES["ASK_CONSENT"]: "Q: To potentially share more tailored information or allow for follow-up, may we collect your contact details (Name, Email, Phone)? \n\n*_(For this demo, data is saved to a local text file - **this is not secure for real patient data**)._*",
    STAGES["GET_NAME"]: "Q: Please enter your First and Last Name:",
    STAGES["GET_EMAIL"]: "Q: Please enter your Email Address:",
    STAGES["GET_PHONE"]: "Q: Please enter your Phone Number (optional, press Enter if skipping):",
    STAGES["FINAL_SUMMARY"]: "Thank you for providing the information. Remember, discussing these findings with your oncologist is crucial for making informed decisions about your care.",
    STAGES["END"]: "You have reached the end of this exploration session. Feel free to restart if you want to explore a different scenario."
}

STAGE_NAMES = {v: k for k, v in STAGES.items()}

# --- Helper Functions ---

def clean_text(text):
    """Basic text cleaning: remove HTML, normalize whitespace, handle None."""
    if text is None: return "N/A"
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text) # Remove HTML tags
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    return text if text else "N/A"

def save_user_data(data):
    """Appends user data to the text file. WARNING: Not secure for real PII."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(USER_DATA_FILE, "a", encoding="utf-8") as f:
            f.write(f"--- Contact Entry: {timestamp} ---\n")
            f.write(f"  Consent Given: {data.get('ConsentGiven', 'N/A')}\n")
            if data.get('ConsentGiven'):
                f.write(f"  Name: {data.get('Name', 'N/A')}\n")
                f.write(f"  Email: {data.get('Email', 'N/A')}\n")
                f.write(f"  Phone: {data.get('Phone', 'N/A')}\n")
            f.write(f"  Context:\n")
            f.write(f"    Diagnosis: {data.get('Diagnosis', 'N/A')}\n")
            f.write(f"    Stage/Progression: {data.get('StageProgression', 'N/A')}\n")
            f.write(f"    Biomarkers: {data.get('Biomarkers', 'N/A')}\n")
            f.write(f"    Prior Treatment: {data.get('PriorTreatment', 'N/A')}\n")
            f.write(f"    Imaging: {data.get('ImagingResponse', 'N/A')}\n")
            f.write(f"--- End Entry ---\n\n")
        logging.info(f"User data appended to {USER_DATA_FILE}")
        return True
    except IOError as e:
        logging.error(f"IOError saving user data to {USER_DATA_FILE}: {e}", exc_info=True)
        st.error("Could not save contact information due to a file system error.")
        return False
    except Exception as e:
        logging.error(f"Unexpected error saving user data: {e}", exc_info=True)
        st.error("An unexpected error occurred while saving contact information.")
        return False


# --- LLM Generation Functions ---

def refine_fda_search_strategy_with_llm(diagnosis, stage_info, markers, model_id):
    """
    Uses LLM to create a prioritized, tiered search strategy for the FDA API.
    Returns a dictionary with keys like 'primary_search', 'secondary_search', etc.,
    each containing a list of search terms.
    *** Requires full implementation with actual Groq API call. ***
    """
    logging.info(f"[LLM Call - Strategy] Refining FDA search strategy for: D='{diagnosis}', S='{stage_info}', M='{markers}', Model={model_id}")
    # Default strategy as a fallback
    default_strategy = {
        "primary_search": [term.strip('",.') for term in f"{diagnosis} {markers}".split() if len(term) > 2],
        "secondary_search": [term.strip('",.') for term in diagnosis.split() if len(term) > 2] or ["cancer"],
        "tertiary_search": [],
        "fallback_search": ["cancer"]
    }
    # Basic normalization for fallback if LLM fails before call
    if diagnosis:
        diag_lower = diagnosis.lower()
        if "non-small cell lung cancer" in diag_lower or "nsclc" in diag_lower:
            default_strategy["primary_search"] = ["non-small cell lung cancer"]
            default_strategy["secondary_search"] = ["lung cancer"]
            if markers and markers.lower() not in ['none', 'unknown', '']:
                 default_strategy["primary_search"].append(markers)
                 default_strategy["secondary_search"].append(markers)
        elif "lung cancer" in diag_lower:
             default_strategy["primary_search"] = ["lung cancer"]
             if markers and markers.lower() not in ['none', 'unknown', '']:
                  default_strategy["primary_search"].append(markers)

    if not GROQ_API_KEY:
        logging.warning("Groq API key not configured. Cannot generate search strategy. Using default fallback.")
        return default_strategy

    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"""
    Analyze the following user-provided cancer information:
    Diagnosis: "{diagnosis}"
    Stage/Progression: "{stage_info}"
    Biomarkers: "{markers}"

    Your task is to create a prioritized search strategy for finding relevant drugs in the FDA 'indications_and_usage' field. The challenge is that specific terms (like 'non-small cell lung cancer') often yield no results, while broader terms (like 'lung cancer') or biomarkers ('EGFR') are more effective.

    Generate a JSON object defining a tiered search strategy with the following keys: "primary_search", "secondary_search", "tertiary_search", "fallback_search". Each key's value should be a list of strings (search terms).

    Prioritization Rules:
    1.  If a specific, actionable biomarker (e.g., EGFR, HER2-positive, BRAF V600E, ALK) is present:
        -   Primary: Combine the biomarker with the *broad* cancer type (e.g., ["lung cancer", "EGFR"]). If stage is metastatic, add "metastatic".
        -   Secondary: Use the biomarker alone (e.g., ["EGFR"]).
        -   Tertiary: Use the specific cancer subtype if available (e.g., ["non-small cell lung cancer"]). If stage is metastatic, add "metastatic".
        -   Fallback: Use the broad cancer type alone (e.g., ["lung cancer"]). If stage is metastatic, add "metastatic".
    2.  If NO specific biomarker is mentioned or markers are 'None'/'Unknown':
        -   Primary: Use the *specific* cancer subtype if available (e.g., ["non-small cell lung cancer"]). Include "metastatic" if mentioned in stage_info.
        -   Secondary: Use the *broad* cancer type (e.g., ["lung cancer"]). Include "metastatic" if relevant.
        -   Tertiary: [Empty List]
        -   Fallback: ["cancer"]
    3.  Normalize terms (e.g., "nsclc" -> "non-small cell lung cancer", "her2 pos" -> "HER2-positive"). Extract the core broad type (e.g., "lung cancer" from "non-small cell lung cancer"). Include "metastatic" if stage indicates spread (e.g., "stage IV", "metastatic", "spread"). Keep term lists concise and focused.

    Output ONLY the JSON object. Ensure the JSON is valid.

    Example Output (Input: nsclc stage 4 egfr+):
    {{
      "primary_search": ["lung cancer", "EGFR", "metastatic"],
      "secondary_search": ["EGFR"],
      "tertiary_search": ["non-small cell lung cancer", "metastatic"],
      "fallback_search": ["lung cancer", "metastatic"]
    }}

    Example Output (Input: metastatic breast cancer, markers unknown):
     {{
      "primary_search": ["breast cancer", "metastatic"],
      "secondary_search": ["breast cancer"],
      "tertiary_search": [],
      "fallback_search": ["cancer"]
    }}

    Example Output (Input: Prostate Cancer):
     {{
      "primary_search": ["prostate cancer"],
      "secondary_search": ["prostate cancer"],
      "tertiary_search": [],
      "fallback_search": ["cancer"]
    }}

    JSON Output:
    """

    strategy = default_strategy # Start with default
    try:
        logging.info(f"Sending prompt to LLM for search strategy: {prompt}")
        # *** Replace with your actual Groq API call ***
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert medical terminology assistant designing FDA search strategies. Output ONLY the valid JSON strategy object like {\"primary_search\": [], \"secondary_search\": [], ... }."},
                {"role": "user", "content": prompt}
            ],
            model=model_id,
            temperature=0.1,
            max_tokens=350,
            response_format={"type": "json_object"}
        )
        response_content = chat_completion.choices[0].message.content
        logging.info(f"LLM raw strategy response: {response_content}")
        # *** End of placeholder replacement block ***

        # Attempt to parse the JSON response
        try:
            parsed_json = json.loads(response_content)
            required_keys = ["primary_search", "secondary_search", "tertiary_search", "fallback_search"]
            if isinstance(parsed_json, dict) and all(key in parsed_json for key in required_keys):
                valid_strategy = True
                temp_strategy = {} # Build validated strategy here
                for key in required_keys:
                    if isinstance(parsed_json[key], list) and all(isinstance(term, str) for term in parsed_json[key]):
                         temp_strategy[key] = parsed_json[key] # Assign validated list
                    else:
                        valid_strategy = False
                        logging.warning(f"LLM strategy JSON invalid: key '{key}' is not a list of strings. Value: {parsed_json.get(key)}")
                        temp_strategy[key] = default_strategy.get(key, []) # Use default for this key
                if valid_strategy:
                    strategy = temp_strategy # Replace default with fully validated LLM strategy
                    logging.info(f"LLM search strategy successfully parsed and validated: {strategy}")
                else:
                    strategy = temp_strategy # Use partially validated strategy with defaults filled in
                    logging.warning(f"LLM strategy JSON structure valid, but content partially invalid. Using mix of LLM/default. Final strategy: {strategy}")
            else:
                logging.warning(f"LLM strategy JSON response did not contain all required keys or wasn't a dict. Using default. Response: {response_content}")
                strategy = default_strategy

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logging.error(f"Error parsing LLM JSON strategy response: {e}. Raw response: {response_content}", exc_info=True)
            strategy = default_strategy # Fallback to default on parsing error

    except (RateLimitError, APIError) as e:
        logging.error(f"Groq API error during strategy generation: {e}", exc_info=True)
        st.warning("Could not generate advanced search strategy due to an API issue. Using basic search.")
        strategy = default_strategy
    except Exception as e:
        logging.error(f"Unexpected error during LLM strategy generation: {e}", exc_info=True)
        st.warning("An unexpected error occurred while generating the search strategy.")
        strategy = default_strategy

    # Final check to ensure all keys exist and are lists
    for key in ["primary_search", "secondary_search", "tertiary_search", "fallback_search"]:
        if key not in strategy or not isinstance(strategy[key], list):
             strategy[key] = default_strategy.get(key, []) # Ensure lists exist

    return strategy


def generate_drug_summary_llm(diagnosis, markers, fda_drugs, model_id):
    """Generates a DIRECT drug summary. Can use LLM or direct text."""
    logging.info(f"[Summary Gen] Generating drug summary for: D='{diagnosis}', M='{markers}'")
    # Using direct text generation for simplicity and control here
    summary = ""
    if fda_drugs:
        drug_list = [f"**{d.get('brand_name', 'N/A')} ({d.get('generic_name', 'N/A')})**" for d in fda_drugs]
        summary = (f"Based on searches of FDA drug labels related to '{diagnosis}' and markers like '{markers}', the following potentially relevant drugs were identified: {', '.join(drug_list)}. "
                   f"See details below for snippets from their official labels.")
    else:
        summary = (f"An automated search of FDA drug label indication texts related to '{diagnosis}' and markers '{markers}' did not find specific matches for approved drugs based on the derived search strategy. "
                   f"Other standard treatments or drugs approved for broader indications might still apply.")
    logging.info("Direct Drug Summary generated.")
    return summary


def generate_trial_summary_llm(diagnosis, markers, stage_info, prior_treatment, clinical_trials, model_id):
    """Generates the clinical trial summary. Can use LLM or direct text."""
    logging.info(f"[Summary Gen] Generating trial summary for: D='{diagnosis}', M='{markers}', S='{stage_info}', Prior='{prior_treatment}'")
    # Using direct text generation for simplicity and control here
    summary = ""
    if clinical_trials:
        trial_titles = [f"'{t.get('title', 'N/A')}'" for t in clinical_trials[:2]]
        summary = (f"Searches on ClinicalTrials.gov for actively recruiting Phase 2 or 3 interventional studies related to '{diagnosis}' (considering stage '{stage_info}', markers '{markers}', prior treatments like '{prior_treatment}') yielded potential matches, such as: {', '.join(trial_titles)}. "
                   f"These trials are investigating newer approaches (details below).")
    else:
        summary = (f"Based on the specific criteria provided (diagnosis '{diagnosis}', stage '{stage_info}', markers '{markers}', prior treatment '{prior_treatment}') and filters (Recruiting, Phase 2+, Interventional), no matching clinical trials were found in this automated search of ClinicalTrials.gov. "
                   f"Trial availability changes frequently, and different search terms might yield results.")
    logging.info("Direct Trial Summary generated.")
    return summary


# --- API Functions ---

def search_fda_drugs_for_condition_and_markers(search_strategy, limit=15, min_results_per_tier=1):
    """
    Searches FDA drug labels via openFDA API using a tiered search strategy.
    *** Requires full implementation with actual API call and error handling. ***
    """
    logging.info(f"[API Call - Tiered] Executing FDA search strategy: {search_strategy}")
    final_results = []
    search_tier_used = "None"
    seen_generic_names = set()

    search_tiers = ["primary_search", "secondary_search", "tertiary_search", "fallback_search"]

    for tier_key in search_tiers:
        search_terms = search_strategy.get(tier_key, [])
        if not search_terms:
            logging.info(f"Skipping tier '{tier_key}' as it has no search terms.")
            continue

        logging.info(f"Attempting FDA search with tier '{tier_key}': {search_terms}")
        query_parts = []
        for term in search_terms:
            if ' ' in term: query_parts.append(f'"{term}"')
            else: query_parts.append(term)
        if not query_parts:
             logging.warning(f"No valid query parts generated for tier '{tier_key}'.")
             continue
        search_query = f'indications_and_usage:({ " AND ".join(query_parts) })'
        logging.info(f"Constructed FDA API query for tier '{tier_key}': {search_query}")

        # --- Actual API Call Implementation Needed ---
        # *** Replace this placeholder block with your real API call for THIS tier ***
        current_tier_results = []
        params = {'search': search_query, 'limit': limit}
        try:
            # >>> START REAL API CALL BLOCK <<<
            response = requests.get(FDA_API_BASE_URL, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()

            # Process results for the current tier
            for item in data.get('results', []):
                openfda_data = item.get('openfda', {})
                brand_name_list = openfda_data.get('brand_name', ['N/A'])
                generic_name_list = openfda_data.get('generic_name', ['N/A'])
                brand_name = brand_name_list[0] if brand_name_list else 'N/A'
                generic_name = generic_name_list[0] if generic_name_list else 'N/A'
                indication_list = item.get('indications_and_usage', ['N/A'])
                indication = indication_list[0] if indication_list else 'N/A'
                indication_snippet = (indication[:300] + '...') if indication and len(indication) > 300 else indication
                label_url = f"https://www.google.com/search?q={urllib.parse.quote(brand_name + ' ' + generic_name + ' FDA label')}"

                # Deduplicate based on generic name before adding
                generic_name_cleaned = clean_text(generic_name)
                if generic_name_cleaned != "N/A" and generic_name_cleaned not in seen_generic_names:
                     current_tier_results.append({
                        "brand_name": clean_text(brand_name), "generic_name": generic_name_cleaned,
                        "indication_snippet": clean_text(indication_snippet), "url": label_url
                     })
                     seen_generic_names.add(generic_name_cleaned) # Add to seen set
            # >>> END REAL API CALL BLOCK <<<

            logging.info(f"Tier '{tier_key}' search returned {len(current_tier_results)} new, unique results.")
            final_results.extend(current_tier_results) # Add new results from this tier

            # Check if we have enough results after this tier
            if len(final_results) >= min_results_per_tier:
                search_tier_used = tier_key
                logging.info(f"Met minimum results ({min_results_per_tier}) with tier '{tier_key}'. Total unique results: {len(final_results)}. Stopping tiered search.")
                break # Stop trying lower priority tiers

        # --- Error Handling for the API Call ---
        except requests.exceptions.Timeout:
            logging.error(f"FDA API request timed out for tier '{tier_key}'.")
            st.warning(f"Drug search timed out on tier '{tier_key}'. Trying next tier if available.")
        except requests.exceptions.HTTPError as e:
            logging.error(f"FDA API request failed for tier '{tier_key}' with HTTP error: {e.response.status_code} - {e.response.text}")
            # Don't show warning for 404 (Not Found), just log it
            if e.response.status_code != 404:
                st.warning(f"Drug search failed on tier '{tier_key}' (Error {e.response.status_code}). Trying next tier if available.")
            else:
                 logging.info(f"Tier '{tier_key}' search returned no results (404).")
            # Continue to the next tier regardless of 404 or other HTTP error
        except requests.exceptions.RequestException as e:
            logging.error(f"FDA API request failed for tier '{tier_key}': {e}", exc_info=True)
            st.warning("Drug search encountered a network issue. Trying next tier if available.")
        except json.JSONDecodeError as e:
             logging.error(f"Failed to decode FDA API JSON response for tier '{tier_key}': {e}", exc_info=True)
             st.warning("Received an invalid response from the drug database. Trying next tier if available.")
        except Exception as e:
             logging.error(f"Unexpected error processing FDA results for tier '{tier_key}': {e}", exc_info=True)
             st.warning("An unexpected error occurred while processing drug information. Trying next tier if available.")
        # --- End of real API call block ---

    logging.info(f"FDA Tiered Search finished. Used up to tier '{search_tier_used}'. Final unique results count: {len(final_results)}")
    return final_results


def search_filtered_clinical_trials(diagnosis, stage_info, markers, prior_treatment, limit=20):
    """
    Searches ClinicalTrials.gov V2 API with filters.
    *** Requires full implementation with actual API call and error handling. ***
    *** Fetches interventions, eligibilityCriteria, contacts, locations ***
    """
    logging.info(f"[API Call] Searching ClinicalTrials.gov for: D='{diagnosis}', S='{stage_info}', M='{markers}', Prior='{prior_treatment}'")
    results = [] # Initialize results list

    # --- Actual API Call Implementation Needed ---
    # *** Replace this placeholder block with your real API call ***
    # 1. Construct Search Query/Filters for ClinicalTrials.gov V2 API
    #    Example: Combine terms, use filters for status, phase, type
    #    query_expr = f"{diagnosis} AND {markers} AND {stage_info}" # Basic example, refine this
    #    filters = "AREA[StudyStatus]RECRUITING AND AREA[Phase]Phase 2 OR Phase 3 AND AREA[StudyType]INTERVENTIONAL"
    #    params = {'query.term': query_expr, 'filter.ids': filters, 'pageSize': limit, 'format': 'json'}
    #    api_url = CTGOV_API_BASE_URL # Use the V2 base URL

    try:
        # >>> START REAL API CALL BLOCK <<<
        # Placeholder simulation (remove when implementing real call)
        time.sleep(1) # Simulate network delay
        processed_results = [] # Use a temporary list for processing
        if diagnosis and "lung cancer" in diagnosis.lower():
            if markers and "egfr" in markers.lower():
                 processed_results.append({
                     "nct_id": "NCT0XXXXXX1", "title": "A Study of Novel Agent X in Patients With EGFR-Mutated NSCLC Progressing on Osimertinib",
                     "status": "RECRUITING", "phase": "PHASE 2", "study_type": "INTERVENTIONAL",
                     "conditions": "EGFR Positive Non-Small Cell Lung Cancer", "summary": "Evaluating efficacy and safety of Novel Agent X post-osimertinib.",
                     "interventions": "Drug: Novel Agent X; Drug: Placebo", "eligibility_snippet": "Key Inclusion: EGFR+, prior osimertinib. Exclusion: unstable brain mets.",
                     "contact_info": "Study Contact: Dr. Investigator, 1-800-555-TRIAL", "locations_snippet": "MSKCC, New York; DFCI, Boston",
                     "url": f"https://clinicaltrials.gov/study/NCT0XXXXXX1"
                     })
            processed_results.append({
                "nct_id": "NCT0YYYYYY2", "title": "Immunotherapy Combination Study for Advanced NSCLC After Chemotherapy",
                "status": "RECRUITING", "phase": "PHASE 3", "study_type": "INTERVENTIONAL",
                "conditions": "Non-Small Cell Lung Cancer Stage IV", "summary": "Comparing SOC vs. immuno combo (Drug A + Drug B) post-platinum chemo.",
                "interventions": "Drug: Drug A; Drug: Drug B; Drug: Docetaxel", "eligibility_snippet": "Inclusion: Stage IV NSCLC, prior platinum chemo. Exclusion: EGFR/ALK mut, autoimmune.",
                "contact_info": "Central Contact: research.center@example.org", "locations_snippet": "Multiple sites USA, Canada, Australia",
                "url": f"https://clinicaltrials.gov/study/NCT0YYYYYY2"
            })
        # End Placeholder simulation
        results = processed_results # Assign placeholder results

        # response = requests.get(api_url, params=params, timeout=30)
        # response.raise_for_status()
        # data = response.json()
        # # Process data['studies'] here... Extract NCT ID, title, status, phase, conditions, summary,
        # # interventions (from protocolSection.armsInterventionsModule.interventionList.intervention),
        # # eligibility (from protocolSection.eligibilityModule.eligibilityCriteria - needs parsing/summarizing),
        # # contacts (from protocolSection.contactsLocationsModule.centralContactList / overallOfficialList),
        # # locations (from protocolSection.contactsLocationsModule.locationList)
        # processed_results = []
        # for study in data.get('studies', []):
        #     protocol = study.get('protocolSection', {})
        #     # ... detailed extraction logic ...
        #     processed_results.append({ ... })
        # results = processed_results # Assign real results
        # >>> END REAL API CALL BLOCK <<<

    # --- Error Handling for the API Call ---
    except requests.exceptions.RequestException as e:
        logging.error(f"ClinicalTrials.gov API request failed: {e}", exc_info=True)
        st.warning("Could not retrieve clinical trial information due to a network or API error.")
        results = [] # Ensure results is empty list on error
    except Exception as e:
        logging.error(f"Unexpected error processing ClinicalTrials.gov results: {e}", exc_info=True)
        st.warning("An unexpected error occurred while processing clinical trial information.")
        results = []
    # --- End of real API call block ---

    logging.info(f"ClinicalTrials.gov Search returned {len(results)} results.")
    return results


# --- Streamlit App ---

st.set_page_config(
    page_title="Cancer Treatment Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
def initialize_session():
    if 'stage' not in st.session_state:
        st.session_state.stage = STAGES["INIT"]
        st.session_state.user_inputs = {}
        st.session_state.messages = [
            {"role": "assistant", "content": STAGE_PROMPTS[STAGES["INIT"]]},
            {"role": "assistant", "content": STAGE_PROMPTS[STAGES["GET_DIAGNOSIS"]]} # Add first question
        ]
        st.session_state.stage = STAGES["GET_DIAGNOSIS"] # Set stage to expect answer
        st.session_state.drug_results = None
        st.session_state.trial_results = None
        st.session_state.consent_given = None
        if 'model_id' not in st.session_state:
             st.session_state.model_id = AVAILABLE_MODELS[DEFAULT_MODEL_DISPLAY_NAME]
        logging.info("Session state initialized for a new conversation.")

initialize_session()

# --- Sidebar ---
with st.sidebar:
    st.subheader("⚙️ Configuration")
    model_display_names = list(AVAILABLE_MODELS.keys())
    current_model_id = st.session_state.get('model_id', AVAILABLE_MODELS[DEFAULT_MODEL_DISPLAY_NAME])
    current_model_display_name = next((name for name, mid in AVAILABLE_MODELS.items() if mid == current_model_id), DEFAULT_MODEL_DISPLAY_NAME)
    try: default_index = model_display_names.index(current_model_display_name)
    except ValueError: default_index = 0

    selected_model_display_name = st.selectbox(
        "AI Model:", options=model_display_names, index=default_index, key="model_select_widget",
        help="Select the AI model used for query refinement and summaries."
    )
    new_model_id = AVAILABLE_MODELS[selected_model_display_name]
    if new_model_id != st.session_state.model_id:
        st.session_state.model_id = new_model_id
        st.success(f"Model updated to: {selected_model_display_name}")

    st.divider()
    if st.button("🔄 Restart Conversation", key="restart_sidebar"):
        keys_to_clear = ['stage', 'user_inputs', 'messages', 'drug_results', 'trial_results', 'consent_given']
        for key in keys_to_clear:
            if key in st.session_state: del st.session_state[key]
        st.rerun()

    st.divider()
    st.markdown("---")
    st.caption("Debug Info:")
    current_stage_num = st.session_state.get('stage', STAGES["INIT"])
    st.write(f"Current Stage: {STAGE_NAMES.get(current_stage_num, 'Unknown')} ({current_stage_num})")
    st.write("Consent Given:", st.session_state.get('consent_given', 'N/A'))

# --- Main App Area ---
st.title("🧑‍⚕️ Cancer Treatment Explorer")
st.caption("AI-Assisted Information Retrieval (Not Medical Advice)")

# --- Main Chat Area ---
chat_container = st.container(height=450)
with chat_container:
    for message in st.session_state.get('messages', []):
        avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else USER_AVATAR
        name = ASSISTANT_NAME if message["role"] == "assistant" else USER_NAME
        with st.chat_message(name=message["role"], avatar=avatar):
            if message.get("type") == "expander":
                with st.expander(message.get("title", "Details"), expanded=False):
                    st.markdown(message["content"], unsafe_allow_html=True)
            elif message.get("type") == "buttons":
                 st.markdown(message["content"])
            else:
                st.markdown(message["content"], unsafe_allow_html=True)

# --- Input Logic ---

# Function to advance stage and add next prompt
def advance_stage(next_stage):
    st.session_state.stage = next_stage
    prompt = STAGE_PROMPTS.get(next_stage)
    if prompt:
        # Avoid adding duplicate prompts if rerun happens quickly
        if not st.session_state.messages or st.session_state.messages[-1].get("content") != prompt:
             st.session_state.messages.append({"role": "assistant", "content": prompt})
    # Mark message for button logic if it's the consent stage
    if next_stage == STAGES["ASK_CONSENT"]:
         if st.session_state.messages and st.session_state.messages[-1].get("content") == prompt:
             st.session_state.messages[-1]["type"] = "buttons"

# Handle consent buttons
if st.session_state.stage == STAGES["ASK_CONSENT"]:
    cols = st.columns(8)
    with cols[0]:
        if st.button("✔️ Yes, Agree", key="consent_yes"):
            st.session_state.consent_given = True; st.session_state.user_inputs['consent'] = True
            st.session_state.messages.append({"role": "user", "content": "A: Yes, I agree to share contact information."})
            advance_stage(STAGES["GET_NAME"]); st.rerun()
    with cols[1]:
         if st.button("❌ No, Decline", key="consent_no"):
            st.session_state.consent_given = False; st.session_state.user_inputs['consent'] = False
            st.session_state.messages.append({"role": "user", "content": "A: No, I do not wish to share contact information now."})
            context_data = {
                "Timestamp": datetime.now().isoformat(), "ConsentGiven": False,
                "Diagnosis": st.session_state.user_inputs.get("diagnosis"), "StageProgression": st.session_state.user_inputs.get("stage"),
                "Biomarkers": st.session_state.user_inputs.get("biomarkers"), "PriorTreatment": st.session_state.user_inputs.get("prior_treatment"),
                "ImagingResponse": st.session_state.user_inputs.get("imaging"),
            }; save_user_data(context_data)
            st.session_state.stage = STAGES["SHOW_TRIALS_NO_CONSENT"]; st.rerun()

# Handle text input
elif st.session_state.stage not in [STAGES["ASK_CONSENT"], STAGES["END"], STAGES["PROCESS_INFO_SHOW_DRUGS"], STAGES["SAVE_CONTACT_SHOW_TRIALS"], STAGES["SHOW_TRIALS_NO_CONSENT"]]:
    current_prompt = STAGE_PROMPTS.get(st.session_state.stage, "Enter response...")
    placeholder_match = re.search(r"\((e\.g\.,.*?)\)", current_prompt)
    placeholder = placeholder_match.group(1) if placeholder_match else "Your answer..."

    if user_input := st.chat_input(placeholder):
        st.session_state.messages.append({"role": "user", "content": f"A: {user_input}"})
        current_stage = st.session_state.stage; next_stage = None
        # --- Stage advancement logic ---
        if current_stage == STAGES["GET_DIAGNOSIS"]: st.session_state.user_inputs['diagnosis'] = user_input; next_stage = STAGES["GET_STAGE"]
        elif current_stage == STAGES["GET_STAGE"]: st.session_state.user_inputs['stage'] = user_input; next_stage = STAGES["GET_BIOMARKERS"]
        elif current_stage == STAGES["GET_BIOMARKERS"]: st.session_state.user_inputs['biomarkers'] = user_input; next_stage = STAGES["GET_PRIOR_TREATMENT"]
        elif current_stage == STAGES["GET_PRIOR_TREATMENT"]: st.session_state.user_inputs['prior_treatment'] = user_input; next_stage = STAGES["GET_IMAGING"]
        elif current_stage == STAGES["GET_IMAGING"]: st.session_state.user_inputs['imaging'] = user_input; st.session_state.stage = STAGES["PROCESS_INFO_SHOW_DRUGS"]
        elif current_stage == STAGES["GET_NAME"]: st.session_state.user_inputs['name'] = user_input; next_stage = STAGES["GET_EMAIL"]
        elif current_stage == STAGES["GET_EMAIL"]:
            if "@" not in user_input or "." not in user_input: st.session_state.messages.append({"role": "assistant", "content": "⚠️ Please enter a valid email address."})
            else: st.session_state.user_inputs['email'] = user_input; next_stage = STAGES["GET_PHONE"]
        elif current_stage == STAGES["GET_PHONE"]: st.session_state.user_inputs['phone'] = user_input if user_input else "N/A"; st.session_state.stage = STAGES["SAVE_CONTACT_SHOW_TRIALS"]
        # --- End stage advancement logic ---
        if next_stage is not None: advance_stage(next_stage)
        st.rerun()

# --- Handle Internal Processing Stages ---

if st.session_state.stage == STAGES["PROCESS_INFO_SHOW_DRUGS"]:
    logging.info("Entering stage: PROCESS_INFO_SHOW_DRUGS")
    diagnosis = st.session_state.user_inputs.get("diagnosis", ""); stage_info = st.session_state.user_inputs.get("stage", ""); markers = st.session_state.user_inputs.get("biomarkers", ""); model_id = st.session_state.model_id

    # 1. Generate Tiered Search Strategy using LLM
    search_strategy = {}
    with st.spinner("Generating advanced search strategy with AI..."):
        search_strategy = refine_fda_search_strategy_with_llm(diagnosis, stage_info, markers, model_id)

    # 2. Search FDA using the tiered strategy
    with st.spinner("Searching FDA drug database using tiered strategy..."):
        st.session_state.drug_results = search_fda_drugs_for_condition_and_markers(search_strategy)
        drug_summary = generate_drug_summary_llm(diagnosis, markers, st.session_state.drug_results, model_id) # Summary uses original terms

    # 3. Display results
    st.session_state.messages.append({"role": "assistant", "content": f"**Potential Therapeutics (Based on FDA Label Search)**\n\n{drug_summary}"})
    if st.session_state.drug_results:
        expander_content = "**Found Drug Details (Label Information):**\n\n"
        for i, drug in enumerate(st.session_state.drug_results):
            expander_content += f"**{i+1}. {drug.get('brand_name', 'N/A')} ({drug.get('generic_name', 'N/A')})**\n"
            expander_content += f"*   *Indication Snippet:* {clean_text(drug.get('indication_snippet', 'N/A'))}\n"
            if drug.get('url'): expander_content += f"*   *Label Search Link (Example):* [Search Google]({drug.get('url')})\n"
            expander_content += "---\n"
        st.session_state.messages.append({"role": "assistant", "content": expander_content.strip('---\n'), "type": "expander", "title": f"View {len(st.session_state.drug_results)} Found Drug Details"})

    # 4. Advance stage
    advance_stage(STAGES["ASK_CONSENT"]); st.rerun()


if st.session_state.stage == STAGES["SAVE_CONTACT_SHOW_TRIALS"]:
    logging.info("Entering stage: SAVE_CONTACT_SHOW_TRIALS")
    contact_data = {
        "Timestamp": datetime.now().isoformat(), "ConsentGiven": st.session_state.consent_given,
        "Name": st.session_state.user_inputs.get("name"), "Email": st.session_state.user_inputs.get("email"), "Phone": st.session_state.user_inputs.get("phone"),
        "Diagnosis": st.session_state.user_inputs.get("diagnosis"), "StageProgression": st.session_state.user_inputs.get("stage"),
        "Biomarkers": st.session_state.user_inputs.get("biomarkers"), "PriorTreatment": st.session_state.user_inputs.get("prior_treatment"),
        "ImagingResponse": st.session_state.user_inputs.get("imaging"),
    }
    if save_user_data(contact_data): st.toast("Contact info recorded (demo purposes only).", icon="✅")
    else: st.toast("Failed to save contact info.", icon="⚠️")
    st.session_state.stage = STAGES["SHOW_TRIALS_NO_CONSENT"]; st.rerun()


if st.session_state.stage == STAGES["SHOW_TRIALS_NO_CONSENT"]:
    logging.info("Entering stage: SHOW_TRIALS_NO_CONSENT (or after consent given)")
    with st.spinner("Searching ClinicalTrials.gov..."):
        diagnosis = st.session_state.user_inputs.get("diagnosis", ""); stage_info = st.session_state.user_inputs.get("stage", ""); markers = st.session_state.user_inputs.get("biomarkers", ""); prior_treatment = st.session_state.user_inputs.get("prior_treatment", ""); model_id = st.session_state.model_id
        st.session_state.trial_results = search_filtered_clinical_trials(diagnosis, stage_info, markers, prior_treatment)
        trial_summary = generate_trial_summary_llm(diagnosis, markers, stage_info, prior_treatment, st.session_state.trial_results, model_id)

    st.session_state.messages.append({"role": "assistant", "content": f"**Potential Clinical Trials (Based on Filtered Search)**\n\n{trial_summary}"})
    if st.session_state.trial_results:
        expander_content = "**Found Clinical Trial Details (Recruiting, Phase 2+, Interventional):**\n\n"
        expander_content += f"*_Search based on: {diagnosis}, {stage_info}, {markers}, {prior_treatment}. Filters applied._*\n\n"
        for i, trial in enumerate(st.session_state.trial_results):
            expander_content += f"**{i+1}. {trial.get('nct_id', 'N/A')}: {trial.get('title', 'N/A')}**\n"
            expander_content += f"*   **Phase:** {trial.get('phase','N/A')} | **Status:** {trial.get('status','N/A')}\n"
            expander_content += f"*   **Conditions:** {clean_text(trial.get('conditions','N/A'))}\n"
            expander_content += f"*   **Interventions:** {clean_text(trial.get('interventions','N/A'))}\n"
            expander_content += f"*   **Key Eligibility Snippet:** {clean_text(trial.get('eligibility_snippet','N/A'))}\n"
            expander_content += f"*   **Summary Snippet:** {clean_text(trial.get('summary','N/A'))}\n"
            expander_content += f"*   **Contact:** {trial.get('contact_info','N/A')}\n"
            expander_content += f"*   **Locations Snippet:** {clean_text(trial.get('locations_snippet','N/A'))}\n"
            if trial.get('url'): expander_content += f"*   **Link:** [View Full Details on ClinicalTrials.gov]({trial.get('url')})\n"
            expander_content += "---\n"
        st.session_state.messages.append({"role": "assistant", "content": expander_content.strip('---\n'), "type": "expander", "title": f"View {len(st.session_state.trial_results)} Found Clinical Trial Details"})
    advance_stage(STAGES["FINAL_SUMMARY"]); st.rerun()


# Handle End Stage
if st.session_state.stage == STAGES["END"]:
     # Ensure end prompt is added only once
     end_prompt_present = any(m.get('content') == STAGE_PROMPTS.get(STAGES["END"]) for m in st.session_state.messages)
     if not end_prompt_present:
         end_prompt = STAGE_PROMPTS.get(STAGES["END"])
         if end_prompt: st.session_state.messages.append({"role": "assistant", "content": end_prompt})

     st.info("Session ended. Use the sidebar or button below to start a new exploration.")
     if st.button("🔄 Start New Exploration", key="restart_main"):
        keys_to_clear = ['stage', 'user_inputs', 'messages', 'drug_results', 'trial_results', 'consent_given']
        for key in keys_to_clear:
            if key in st.session_state: del st.session_state[key]
        st.rerun()

# --- (Disclaimer remains commented out) ---