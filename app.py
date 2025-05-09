# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------
# AI Cancer Information Assistant (Production Candidate)
# --------------------------------------------------------------------------

# 1. SET PAGE CONFIG FIRST - THIS IS CRITICAL
import streamlit as st
st.set_page_config(
    page_title="AI Cancer Info Assistant",
    page_icon="🧑‍⚕️", # You can use a URL to an image too: "https://..."
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:help@example.com', # Replace with your actual help email/URL
        'Report a bug': "mailto:bugs@example.com", # Replace with your actual bug report email/URL
        'About': """
        ## AI Cancer Information Assistant
        
        This tool provides information to help you explore and learn about cancer drugs and clinical trials.
        
        
        Always discuss your specific health situation, medical concerns, and treatment options 
        directly with your doctor or other qualified healthcare provider.
        """
    }
)

# --- Standard Library Imports ---
import os
import json
import logging
import re
import time
import pickle
import math
import ast # For safely evaluating string representation of list/tuples
from datetime import datetime
import textwrap

# --- Third-party Library Imports ---
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain.pydantic_v1 import BaseModel, Field # BaseModel for tool inputs

# --- Application Configuration & Setup ---
# Configure logging
# In production, you might want to log to a file or a logging service
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - Line %(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()] # Logs to console
)
logger = logging.getLogger(__name__)

load_dotenv()

# --- API Key Check ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    # This error will be displayed on the Streamlit page thanks to st.set_page_config being first
    st.error(
        "🚨 **CRITICAL ERROR: ANTHROPIC_API_KEY is not configured!**\n\n"
        "The application cannot connect to the AI model. Please ensure the `ANTHROPIC_API_KEY` "
        "environment variable is set correctly in your deployment environment or `.env` file and restart."
    )
    logger.critical("ANTHROPIC_API_KEY not found. Application cannot proceed.")
    st.stop() # Stop execution if the key is missing

# --- File Paths & Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # For robust file pathing
DRUG_DATA_CSV = os.path.join(BASE_DIR, 'drug_data.csv')
TRIAL_DATA_XLSX = os.path.join(BASE_DIR, 'trials_filtered_with_coordinates.xlsx')
DRUG_EMBEDDINGS_FILE = os.path.join(BASE_DIR, 'drug_embeddings.npy')
TRIAL_EMBEDDINGS_FILE = os.path.join(BASE_DIR, 'trial_embeddings.npy')
TRIAL_INDEX_MAP_FILE = os.path.join(BASE_DIR, 'trial_index_map.pkl')
GEOCODE_CACHE_FILE_PATH = os.path.join(BASE_DIR, "geocode_cache.json")
TEMP_GEOCODE_CACHE_FILE_PATH = os.path.join(BASE_DIR, "geocode_cache.tmp.json")

EMBEDDING_MODEL_NAME = 'neuml/pubmedbert-base-embeddings'
DRUG_TEXT_COLUMNS_FOR_EMBEDDING = ['Cancer Type']
TRIAL_TEXT_COLUMNS_FOR_EMBEDDING = ['Study Type', 'Conditions']

TRIAL_FILTER_PRIMARY_OUTCOME_COLUMN = 'Primary Outcome Measures'
TRIAL_FILTER_PRIMARY_OUTCOME_TERM = 'Overall Survival'
TRIAL_FILTER_PHASES_COLUMN = 'Phases'
TRIAL_ACCEPTABLE_PHASES_STR = ['PHASE1|PHASE2', 'PHASE2', 'PHASE2|PHASE3', 'PHASE3', 'PHASE4']
TRIAL_ACCEPTABLE_INDIVIDUAL_PHASES = set()
for phase_combo in TRIAL_ACCEPTABLE_PHASES_STR:
    for phase_str_part in re.split(r'[|/,\s]+', phase_combo): # Renamed 'phase' to 'phase_str_part'
        if phase_str_part: TRIAL_ACCEPTABLE_INDIVIDUAL_PHASES.add(phase_str_part.strip().upper())
TRIAL_FILTER_STUDY_TYPE_COLUMN = 'Study Type'
TRIAL_FILTER_STUDY_TYPE_VALUE = 'INTERVENTIONAL'

CLAUDE_MODEL_NAME = "claude-3-haiku-latest" # Balance of speed and capability
DEFAULT_SEARCH_RADIUS_MILES = 50
# !! IMPORTANT: Update NOMINATIM_USER_AGENT for geocoding to avoid being blocked !!
# Use a unique app name and your contact email, e.g., "MyCancerInfoApp/1.0 contact@example.com"
NOMINATIM_USER_AGENT = "AI_Cancer_Info_Assistant_Prod/1.0"
API_REQUEST_DELAY_SECONDS = 1.05 # For Nominatim rate limiting
API_TIMEOUT_SECONDS = 15

ASSISTANT_AVATAR = "🧑‍⚕️"
USER_AVATAR = "👤"

# --- Helper Functions ---
def parse_time_to_months(time_str):
    if isinstance(time_str, (int, float)) and not math.isnan(time_str): return float(time_str)
    if not isinstance(time_str, str): return None
    time_str_cleaned = str(time_str).strip().lower()
    if time_str_cleaned in ['n/a', 'not applicable', 'not reported', 'not reached', 'nr', '', 'nan']: return None
    match_months = re.match(r'(\d+(\.\d+)?)\s*m', time_str_cleaned)
    if match_months: return float(match_months.group(1))
    match_years = re.match(r'(\d+(\.\d+)?)\s*y', time_str_cleaned)
    if match_years: return float(match_years.group(1)) * 12
    try: return float(time_str_cleaned)
    except ValueError: return None

def sort_key_with_none(value, reverse=True):
    is_none_or_nan = value is None or (isinstance(value, float) and math.isnan(value))
    if is_none_or_nan: return float('-inf') if reverse else float('inf')
    try: return float(value)
    except (ValueError, TypeError): return float('-inf') if reverse else float('inf')

def check_phases(trial_phases_raw_str): # Renamed variable for clarity
    if not isinstance(trial_phases_raw_str, str) or not str(trial_phases_raw_str).strip(): return False
    trial_individual_phases = re.split(r'[|/,\s]+', str(trial_phases_raw_str).strip())
    return any(phase_part.strip().upper() in TRIAL_ACCEPTABLE_INDIVIDUAL_PHASES for phase_part in trial_individual_phases if phase_part) # Renamed 'phase'

# --- Data Loading and Preprocessing Functions ---
@st.cache_resource(show_spinner="Loading AI model components...")
def load_sentence_transformer_model(model_name=EMBEDDING_MODEL_NAME):
    try:
        model = SentenceTransformer(model_name)
        logger.info(f"Embedding model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Fatal error loading Sentence Transformer model '{model_name}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to load essential AI model component: {e}")

@st.cache_data(show_spinner="Loading drug information database...")
def load_and_preprocess_drug_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        # Ensure required columns for embedding exist
        if not all(col in df.columns for col in DRUG_TEXT_COLUMNS_FOR_EMBEDDING):
            raise ValueError(f"Missing required columns for drug embedding in {csv_path}. Expected: {DRUG_TEXT_COLUMNS_FOR_EMBEDDING}")
        df['combined_text_for_embedding'] = df[DRUG_TEXT_COLUMNS_FOR_EMBEDDING].fillna('').astype(str).agg(' '.join, axis=1)
        # Add other common parsings upfront if frequently used by LLM, or let LLM handle raw data
        df['Treatment_OS_Months_Parsed'] = df.get('Treatment_OS', pd.Series([None]*len(df))).apply(parse_time_to_months)
        df['Control_OS_Months_Parsed'] = df.get('Control_OS', pd.Series([None]*len(df))).apply(parse_time_to_months)
        df = df.fillna('N/A') # Fill all NaNs after specific parsing
        logger.info(f"Drug data loaded and preprocessed from '{csv_path}'. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Fatal error loading or preprocessing drug data from '{csv_path}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to load drug information database: {e}")

@st.cache_data(show_spinner="Preparing drug information for search...")
def get_or_generate_drug_embeddings(_drug_df, _model, embeddings_path=DRUG_EMBEDDINGS_FILE):
    if os.path.exists(embeddings_path):
        try:
            embeddings = np.load(embeddings_path)
            if embeddings.shape[0] == len(_drug_df) and embeddings.shape[1] == _model.get_sentence_embedding_dimension():
                logger.info("Drug embeddings loaded from cached file.")
                return embeddings
            logger.warning("Cached drug embeddings shape mismatch with data. Regenerating.")
        except Exception as e:
            logger.warning(f"Error loading drug embeddings from cache: {e}. Regenerating.", exc_info=True)
    try:
        texts_to_embed = _drug_df['combined_text_for_embedding'].tolist()
        if not texts_to_embed:
            logger.warning("No text found in drug data for embedding.")
            return np.array([])
        embeddings = _model.encode(texts_to_embed, show_progress_bar=False, convert_to_numpy=True)
        np.save(embeddings_path, embeddings)
        logger.info("Drug embeddings generated and saved to file.")
        return embeddings
    except Exception as e:
        logger.error(f"Fatal error generating drug embeddings: {e}", exc_info=True)
        raise RuntimeError(f"Failed to prepare drug information for search: {e}")

@st.cache_data(show_spinner="Loading clinical trial information database...")
def load_and_preprocess_trial_data(xlsx_path):
    try:
        df = pd.read_excel(xlsx_path)
        if not all(col in df.columns for col in TRIAL_TEXT_COLUMNS_FOR_EMBEDDING):
            raise ValueError(f"Missing required columns for trial embedding in {xlsx_path}. Expected: {TRIAL_TEXT_COLUMNS_FOR_EMBEDDING}")
        df['combined_text_for_embedding'] = df[TRIAL_TEXT_COLUMNS_FOR_EMBEDDING].fillna('').astype(str).agg(' '.join, axis=1)

        if 'location_coordinates' in df.columns:
            def parse_excel_coords_robust(coord_str_val):
                if pd.isna(coord_str_val) or not isinstance(coord_str_val, str) or not coord_str_val.strip(): return []
                try:
                    # Attempt to fix common JSON-like issues if ast.literal_eval fails
                    try:
                        parsed_list = ast.literal_eval(coord_str_val)
                    except (SyntaxError, ValueError):
                        # Try to treat as JSON if ast fails (e.g. if it's actual JSON string)
                        coord_str_val_fixed = coord_str_val.replace("'", "\"") # Replace single with double quotes
                        parsed_list = json.loads(coord_str_val_fixed)

                    if isinstance(parsed_list, list):
                        valid_coords = []
                        for item in parsed_list:
                            if isinstance(item, (list, tuple)) and len(item) == 2 and \
                               all(isinstance(num, (int, float)) and not math.isnan(num) for num in item):
                                valid_coords.append(tuple(item))
                        return valid_coords
                    return []
                except Exception as parse_err: # Catch any error during parsing
                    logger.warning(f"Could not parse coordinate string: '{str(coord_str_val)[:50]}...'. Error: {parse_err}")
                    return []
            df['parsed_location_coordinates'] = df['location_coordinates'].apply(parse_excel_coords_robust)
        else:
            logger.warning("'location_coordinates' column not found in trial data. Location-based search will be impaired.")
            df['parsed_location_coordinates'] = pd.Series([[] for _ in range(len(df))]) # Ensure column exists
        
        df = df.fillna('N/A')
        logger.info(f"Trial data loaded and preprocessed from '{xlsx_path}'. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Fatal error loading or preprocessing trial data from '{xlsx_path}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to load clinical trial information database: {e}")

@st.cache_data(show_spinner="Preparing trial information for search...")
def get_or_generate_trial_embeddings(_trial_df, _model, embeddings_path=TRIAL_EMBEDDINGS_FILE, map_path=TRIAL_INDEX_MAP_FILE):
    # ... (Embedding generation logic - keep as is from previous correct version)
    if os.path.exists(embeddings_path) and os.path.exists(map_path):
        try:
            embeddings = np.load(embeddings_path)
            with open(map_path, 'rb') as f: index_map = pickle.load(f)
            if (embeddings is not None and index_map is not None and
                embeddings.shape[0] == len(index_map) and
                embeddings.shape[1] == _model.get_sentence_embedding_dimension() and
                (max(index_map.keys(), default=-1) < len(_trial_df) if index_map else True)): # Check map keys if map not empty
                logger.info("Trial embeddings and index map loaded from cached files.")
                return embeddings, index_map
            logger.warning("Cached trial embeddings/map mismatch or corrupt. Regenerating.")
        except Exception as e:
            logger.warning(f"Error loading trial embeddings/map from cache: {e}. Regenerating.", exc_info=True)
    try:
        non_empty_mask = _trial_df['combined_text_for_embedding'].astype(str).str.strip() != ''
        non_empty_indices = _trial_df.index[non_empty_mask].tolist()
        texts_to_embed = _trial_df.loc[non_empty_indices, 'combined_text_for_embedding'].tolist()
        if not texts_to_embed: 
            logger.warning("No non-empty text in trial data for embedding.")
            return np.array([]), {}
        embeddings = _model.encode(texts_to_embed, show_progress_bar=False, convert_to_numpy=True)
        index_map = {original_idx: emb_idx for emb_idx, original_idx in enumerate(non_empty_indices)}
        
        np.save(embeddings_path, embeddings)
        with open(map_path, 'wb') as f: pickle.dump(index_map, f)
        logger.info("Trial embeddings and index map generated and saved to files.")
        return embeddings, index_map
    except Exception as e:
        logger.error(f"Fatal error generating trial embeddings: {e}", exc_info=True)
        raise RuntimeError(f"Failed to prepare trial information for search: {e}")


# --- Global Data Initialization ---
DATA_LOADED_SUCCESSFULLY = False
embedding_model_global = None
df_drugs_processed_global = None
drug_embeddings_array_global = None
df_trials_processed_global = None
trial_embeddings_array_global = None
trial_index_map_global = None

try:
    # Check if Nominatim user agent is set to default, if so, warn user.
    if NOMINATIM_USER_AGENT == "AI_Cancer_Info_Assistant_Prod/1.0 your_contact_email@example.com":
        st.sidebar.warning(
            "**Geocoding Service Alert:**\n\n"
            "The `NOMINATIM_USER_AGENT` is using a default placeholder value. "
            "For reliable geocoding (finding nearby trials), please update this constant "
            "in the script with your unique application name and contact email "
            "(e.g., 'MyCancerApp/1.0 admin@myapp.com'). Using the default may lead to service interruption."
        )
    embedding_model_global = load_sentence_transformer_model()
    df_drugs_processed_global = load_and_preprocess_drug_data(DRUG_DATA_CSV)
    drug_embeddings_array_global = get_or_generate_drug_embeddings(df_drugs_processed_global, embedding_model_global)
    df_trials_processed_global = load_and_preprocess_trial_data(TRIAL_DATA_XLSX)
    trial_embeddings_array_global, trial_index_map_global = get_or_generate_trial_embeddings(df_trials_processed_global, embedding_model_global)
    DATA_LOADED_SUCCESSFULLY = True
    logger.info("All critical data and AI models initialized successfully.")
except RuntimeError as e: # Catch specific RuntimeError from loading functions
    # Error already logged by the failing function
    st.error(
        f"🚨 **APPLICATION STARTUP FAILED** 🚨\n\n"
        f"A critical error occurred during data or model initialization:\n\n"
        f"`{e}`\n\n"
        "Please check the application logs for more details and ensure all required data files "
        "are correctly placed and formatted. The application cannot continue without these resources."
    )
    # DATA_LOADED_SUCCESSFULLY remains False, app will not proceed to main logic


# --- Dynamic Top-N Selection & Geocoding Cache (Keep as is from previous correct version) ---
def _determine_top_n_results(scored_results_list):
    if not scored_results_list: return 0
    count = len(scored_results_list)
    if count == 0: return 0 # Handle empty list explicitly
    if count <= 10: return count
    
    # Score of the 10th item (0-indexed), ensure it exists
    score_at_10th = scored_results_list[9]['semantic_similarity'] if count > 9 else 0 

    if score_at_10th >= 0.7: return min(count, 20)
    if score_at_10th < 0.5: return 10
    return min(count, 15)

def load_persistent_geocode_cache(filepath=GEOCODE_CACHE_FILE_PATH):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f: return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Geocode cache file '{filepath}' is corrupted. Starting with an empty cache.")
            return {}
        except Exception as e:
            logger.warning(f"Error loading geocode cache '{filepath}': {e}. Starting with an empty cache.")
            return {}
    return {}

def save_persistent_geocode_cache(cache_data, target_path=GEOCODE_CACHE_FILE_PATH, temp_path=TEMP_GEOCODE_CACHE_FILE_PATH):
    try:
        with open(temp_path, 'w') as f: json.dump(cache_data, f, indent=2)
        if os.path.exists(target_path): os.remove(target_path) # Atomic write
        os.rename(temp_path, target_path)
    except Exception as e:
        logger.warning(f"Could not save geocode cache to '{target_path}': {e}", exc_info=True)

# --- Langchain Tools Definition ---
# Ensure all tool functions have clear docstrings for the agent.
class FindDrugsInput(BaseModel):
    diagnosis: str = Field(description="The primary diagnosis, e.g., 'Metastatic Breast Cancer'")
    stage: str = Field(description="The cancer stage, e.g., 'Stage IV', 'Recurrent', 'Advanced'")
    biomarkers: str = Field(description="Known biomarkers, comma-separated or 'None', e.g., 'HER2-positive, ER-negative', 'EGFR mutation'")

@tool("find_drugs_tool", args_schema=FindDrugsInput)
def find_drugs_tool(diagnosis: str, stage: str, biomarkers: str) -> str:
    """
    Finds relevant drug studies based on the patient's diagnosis, stage, and biomarkers.
    It returns a JSON string containing a list of potentially relevant drugs with key details.
    The AI assistant should then review this data for relevance and summarize the most important findings
    for the user in patient-friendly language. Technical details like 'semantic_similarity'
    from the tool's output are for internal ranking and should NOT be shown directly to the user.
    """
    logger.info(f"Executing find_drugs_tool: Dx='{diagnosis}', Stg='{stage}', Bio='{biomarkers}'")
    if drug_embeddings_array_global is None or drug_embeddings_array_global.size == 0:
        logger.error("Drug embeddings not available for find_drugs_tool.")
        return json.dumps({"error": "Drug information database is currently unavailable.", "drugs": []})

    query_text = f"{diagnosis} {stage} {biomarkers}".strip()
    if not query_text:
        logger.warning("Empty query received for find_drugs_tool.")
        return json.dumps({"error": "Please provide a diagnosis to search for drugs.", "drugs": []})

    try:
        query_embedding = embedding_model_global.encode(query_text, convert_to_numpy=True)
        similarities = cosine_similarity([query_embedding], drug_embeddings_array_global)[0]
    except Exception as e:
        logger.error(f"Error during drug embedding or similarity calculation: {e}", exc_info=True)
        return json.dumps({"error": "An internal error occurred while searching for drugs.", "drugs": []})

    potential_results = []
    for idx, row_series in df_drugs_processed_global.iterrows(): # Renamed 'row' to 'row_series'
        if idx >= len(similarities): continue
        similarity_score = similarities[idx] # Renamed 'sim' to 'similarity_score'
        if similarity_score >= 0.4: # Pool threshold
            potential_results.append({
                'drug_name': row_series.get('Drug Name', 'N/A'),
                'cancer_type_studied': row_series.get('Cancer Type', 'N/A'),
                'treatment_os_data': row_series.get('Treatment_OS', 'N/A'),
                'control_os_data': row_series.get('Control_OS', 'N/A'),
                'os_improvement_percent_data': row_series.get('OS_Improvement (%)', 'N/A'),
                'treatment_pfs_data': row_series.get('Treatment_PFS', 'N/A'),
                'control_pfs_data': row_series.get('Control_PFS', 'N/A'),
                'pfs_improvement_percent_data': row_series.get('PFS_Improvement (%)', 'N/A'),
                'brief_summary_from_study': textwrap.shorten(str(row_series.get('Brief Study Summary', '')), 250, placeholder="..."),
                'semantic_similarity': round(float(similarity_score), 3),
            })
    potential_results.sort(key=lambda x: x['semantic_similarity'], reverse=True)
    
    num_to_select_count = _determine_top_n_results(potential_results) # Renamed 'num_to_select' to 'num_to_select_count'
    final_selected_results = potential_results[:num_to_select_count] # Renamed 'final_results' to 'final_selected_results'
    
    if not final_selected_results:
        return json.dumps({"message": "No drug studies closely matched your specific criteria at this time.", "drugs": []})
    return json.dumps({"drugs": final_selected_results, "count": len(final_selected_results)})


class FindClinicalTrialsInput(BaseModel):
    diagnosis: str = Field(description="Primary diagnosis, e.g., 'Non-Small Cell Lung Cancer'")
    stage: str = Field(description="Cancer stage, e.g., 'Stage IV', 'Metastatic'")
    biomarkers: str = Field(description="Biomarkers, comma-separated or 'None', e.g., 'ALK-positive'")
    user_latitude: float = Field(None, description="User's latitude. Optional for location-based search.")
    user_longitude: float = Field(None, description="User's longitude. Optional for location-based search.")
    radius_miles: int = Field(DEFAULT_SEARCH_RADIUS_MILES, description="Search radius in miles if location is provided.")

@tool("find_clinical_trials_tool", args_schema=FindClinicalTrialsInput)
def find_clinical_trials_tool(diagnosis: str, stage: str, biomarkers: str, user_latitude: float = None, user_longitude: float = None, radius_miles: int = DEFAULT_SEARCH_RADIUS_MILES) -> str:
    """
    Searches for relevant clinical trials based on diagnosis, stage, biomarkers, and optionally, location.
    This tool applies initial filters for 'Overall Survival' as a primary outcome, specific trial phases,
    and 'INTERVENTIONAL' study type. It returns a JSON string of top matching trials with key details.
    The AI assistant should review these for relevance and summarize important findings for the user
    in patient-friendly language. Technical details like 'semantic_similarity' from the tool's output
    are for internal ranking and should NOT be shown directly to the user. If location was used for the search,
    the distance to trial sites should be mentioned if available and relevant.
    """
    logger.info(f"Executing find_clinical_trials_tool: Dx='{diagnosis}', Stg='{stage}', Bio='{biomarkers}', Loc=({user_latitude},{user_longitude}), Rad={radius_miles}")
    # ... (tool logic with improved variable names and checks)
    if trial_embeddings_array_global is None or trial_embeddings_array_global.size == 0 or \
       trial_index_map_global is None: # Check trial_index_map_global as well
        logger.error("Trial embeddings or index map not available for find_clinical_trials_tool.")
        return json.dumps({"error": "Clinical trial information database is currently unavailable.", "trials": []})

    query_text = f"{diagnosis} {stage} {biomarkers}".strip()
    if not query_text:
        logger.warning("Empty query received for find_clinical_trials_tool.")
        return json.dumps({"error": "Please provide a diagnosis to search for clinical trials.", "trials": []})
    
    user_coordinates = (user_latitude, user_longitude) if user_latitude is not None and user_longitude is not None else None # Renamed 'user_loc' to 'user_coordinates'

    try:
        query_embedding = embedding_model_global.encode(query_text, convert_to_numpy=True)
    except Exception as e:
        logger.error(f"Error embedding trial query: {e}", exc_info=True)
        return json.dumps({"error": "An internal error occurred while preparing your trial search.", "trials": []})

    potential_results = []
    for original_idx, row_data in df_trials_processed_global.iterrows(): # Renamed 'i' to 'original_idx', 'row' to 'row_data'
        # Apply hard filters first
        primary_outcome_str = str(row_data.get(TRIAL_FILTER_PRIMARY_OUTCOME_COLUMN, '')).lower()
        trial_phases_str = str(row_data.get(TRIAL_FILTER_PHASES_COLUMN, ''))
        study_type_str = str(row_data.get(TRIAL_FILTER_STUDY_TYPE_COLUMN, '')).upper()

        if not (TRIAL_FILTER_PRIMARY_OUTCOME_TERM.lower() in primary_outcome_str and \
                check_phases(trial_phases_str) and \
                study_type_str == TRIAL_FILTER_STUDY_TYPE_VALUE.upper() and \
                original_idx in trial_index_map_global and \
                trial_index_map_global[original_idx] < len(trial_embeddings_array_global)):
            continue
        
        embedding_idx = trial_index_map_global[original_idx]
        similarity_score = cosine_similarity([query_embedding], [trial_embeddings_array_global[embedding_idx]])[0][0]
        
        if similarity_score < 0.4: continue # Pool threshold

        trial_details = { # Renamed 'trial_info' to 'trial_details'
            'nct_id': row_data.get('NCT Number', 'N/A'),
            'title_from_study': textwrap.shorten(str(row_data.get('Study Title', 'N/A')), 150, placeholder="..."),
            'status': row_data.get('Study Status', 'N/A'), 
            'phases': trial_phases_str, # Use the string directly
            'conditions_studied': textwrap.shorten(str(row_data.get('Conditions', 'N/A')), 100, placeholder="..."),
            'interventions_studied': textwrap.shorten(str(row_data.get('Interventions', 'N/A')), 100, placeholder="..."),
            'brief_summary_from_study': textwrap.shorten(str(row_data.get('Brief Summary', '')), 250, placeholder="..."),
            'official_url': f"https://clinicaltrials.gov/study/{row_data.get('NCT Number', '')}" if row_data.get('NCT Number', 'N/A') != 'N/A' else None,
            'semantic_similarity': round(float(similarity_score), 3),
            'distance_miles': None, # Initialize
        }

        if user_coordinates:
            trial_site_coordinates_list = row_data.get('parsed_location_coordinates', [])
            min_distance_miles = float('inf') # Renamed 'min_dist' to 'min_distance_miles'
            if trial_site_coordinates_list: # Check if list is not empty
                for site_coords_tuple in trial_site_coordinates_list: # Renamed 'site_coords' to 'site_coords_tuple'
                    try:
                        current_distance = geodesic(user_coordinates, site_coords_tuple).miles # Renamed 'dist' to 'current_distance'
                        if current_distance < min_distance_miles:
                            min_distance_miles = current_distance
                    except Exception as dist_err:
                        logger.debug(f"Could not calculate distance for site {site_coords_tuple} in trial {trial_details['nct_id']}: {dist_err}")
            
            if min_distance_miles <= radius_miles:
                trial_details['distance_miles'] = round(min_distance_miles, 1)
                potential_results.append(trial_details)
            # If location search but no results yet in radius, consider closest ones
            elif not any(res.get('distance_miles') is not None for res in potential_results) and min_distance_miles != float('inf'):
                 trial_details['distance_miles'] = round(min_distance_miles, 1)
                 potential_results.append(trial_details)
        else: # No location search
            potential_results.append(trial_details)
            
    # Sort results
    if user_coordinates:
        potential_results.sort(key=lambda x_item: (sort_key_with_none(x_item['distance_miles'], False), -x_item['semantic_similarity'])) # Renamed 'x' to 'x_item'
    else:
        potential_results.sort(key=lambda x_item: x_item['semantic_similarity'], reverse=True)

    num_to_select_count = _determine_top_n_results(potential_results)
    final_selected_results = potential_results[:num_to_select_count]

    if not final_selected_results:
        return json.dumps({"message": "No clinical trials closely matched your specific criteria at this time.", "trials": []})
    return json.dumps({"trials": final_selected_results, "count": len(final_selected_results)})

class ZipToCoordinatesInput(BaseModel):
    zip_code: str = Field(description="User's zip code, e.g., '90210'")
    country_code: str = Field("US", description="Country code for the zip code, e.g., 'US', 'CA'. Defaults to 'US'.")

@tool("zip_to_coordinates_tool", args_schema=ZipToCoordinatesInput)
def zip_to_coordinates_tool(zip_code: str, country_code: str = "US") -> str:
    """
    Converts a user's zip code and country code into geographic coordinates (latitude and longitude).
    Uses a cache to speed up responses for previously geocoded zip codes.
    Returns a JSON string with 'latitude', 'longitude', and 'status'.
    The AI assistant uses these coordinates for location-specific searches, like nearby clinical trials.
    """
    logger.info(f"Executing zip_to_coordinates_tool: Zip='{zip_code}', Country='{country_code}'")
    # ... (Tool logic - keep as is from previous correct version, ensure logging and error handling)
    geocode_cache = load_persistent_geocode_cache() # Renamed 'cache' to 'geocode_cache'
    cache_lookup_key = f"{zip_code}_{country_code}".lower() # Renamed 'key' to 'cache_lookup_key'
    if cache_lookup_key in geocode_cache and geocode_cache[cache_lookup_key]:
        logger.info(f"Zip cache hit for '{cache_lookup_key}'. Coords: {geocode_cache[cache_lookup_key]}")
        cached_coords = geocode_cache[cache_lookup_key]
        return json.dumps({"latitude": cached_coords[0], "longitude": cached_coords[1], "status": "success_cache"})

    geolocator_service = Nominatim(user_agent=NOMINATIM_USER_AGENT) # Renamed 'geolocator' to 'geolocator_service'
    nominatim_warning_message = "" # Renamed 'warning_msg' to 'nominatim_warning_message'
    if NOMINATIM_USER_AGENT == "AI_Cancer_Info_Assistant_Prod/1.0 your_contact_email@example.com": # Default placeholder check
         nominatim_warning_message = "Warning: Nominatim user agent is using a default placeholder. Geocoding accuracy and availability may be affected. Please update the agent string in the application code. "

    try:
        time.sleep(API_REQUEST_DELAY_SECONDS)
        location_data = geolocator_service.geocode(f"{zip_code}, {country_code}", timeout=API_TIMEOUT_SECONDS) # Renamed 'loc' to 'location_data'
        if location_data:
            coordinates = (location_data.latitude, location_data.longitude) # Renamed 'coords' to 'coordinates'
            geocode_cache[cache_lookup_key] = coordinates
            save_persistent_geocode_cache(geocode_cache)
            logger.info(f"Geocoded '{cache_lookup_key}' to {coordinates}. Saved to cache.")
            return json.dumps({"latitude": coordinates[0], "longitude": coordinates[1], "status": "success_api"})
        
        logger.warning(f"Could not geocode zip: {zip_code}, country: {country_code} using Nominatim.")
        return json.dumps({"error": f"{nominatim_warning_message}Could not find location for zip code: {zip_code}.", "status": "error_not_found"})
    except (GeocoderTimedOut, GeocoderUnavailable) as ge_service_error: # Renamed 'e' to 'ge_service_error'
        logger.error(f"Nominatim service issue for zip {zip_code}: {ge_service_error}", exc_info=True)
        return json.dumps({"error": f"{nominatim_warning_message}The location service is temporarily unavailable or timed out. Please try again later.", "status": "error_service_unavailable"})
    except Exception as general_ge_error: # Renamed 'e' to 'general_ge_error'
        logger.error(f"Unexpected error geocoding zip {zip_code}: {general_ge_error}", exc_info=True)
        return json.dumps({"error": f"{nominatim_warning_message}An unexpected error occurred while trying to find the location for zip code: {zip_code}.", "status": "error_unknown"})

available_tools = [find_drugs_tool, find_clinical_trials_tool, zip_to_coordinates_tool]

# --- Langchain Agent Setup ---
@st.cache_resource(show_spinner="Initializing AI Assistant...")
def get_langchain_agent_executor():
    # This system prompt is CRITICAL for patient-friendly interaction and accurate information presentation.
    # It needs to be very directive.
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an AI Patient Information Assistant specializing in cancer drugs and clinical trials.
Your primary goal is to have a helpful, empathetic, and clear conversation with users (patients or their loved ones) seeking information. You must adhere to the following principles and output formats STRICTLY.

**Core Principles:**
1.  **Empathy and Patient-Centric Language:**
    *   Always begin by acknowledging the user's query, especially if it's about a diagnosis. E.g., "I understand you're looking for information about [diagnosis]. I'll do my best to help you explore this."
    *   Use simple, clear language. AVOID MEDICAL JARGON. If a medical term is unavoidable (e.g., 'metastatic'), explain it briefly in plain terms (e.g., "metastatic, which means the cancer has spread").
    *   Maintain a supportive, calm, and professional tone. You are an information assistant, not a doctor.

2.  **Information Gathering (Be Conversational):**
    *   If a user's query is too general (e.g., "cancer drugs"), gently ask for more specifics: "To help me find the most relevant information, could you please tell me the specific type of cancer you're interested in? Information like the cancer stage or any known biomarkers can also be very helpful."
    *   Before using tools, confirm your understanding if the query was complex: "Okay, just to confirm, you're asking about [paraphrased query]. Is that right?"

3.  **Using Your Tools (`find_drugs_tool`, `find_clinical_trials_tool`, `zip_to_coordinates_tool`):**
    *   These tools return JSON data. You MUST process this JSON data.
    *   If discussing trials and location is relevant, ask: "I can also look for trials near a specific area. Would you like me to do that? If so, please provide a zip code."
    *   If they provide a zip code, use `zip_to_coordinates_tool` first, then use the latitude/longitude with `find_clinical_trials_tool`.

4.  **Presenting Information (Your Most Important Task - Follow Formats EXACTLY):**
    *   **NEVER directly output the raw JSON from tools.**
    *   **Critical Review:** From the tool's JSON output, select only the 1-2 MOST relevant items for the user's specific situation. If there are more, you can mention "There are a few other possibilities, we can explore them if you like." If no results are highly relevant, state that clearly and gently.
    *   **DO NOT MENTION "semantic similarity" or any internal scores to the user.**
    *   **Use Markdown for ALL formatted output to the user.**

    *   **Drug Information Presentation Format (Strict Adherence):**
        ```markdown
        Based on your information about [User's Diagnosis/Context], here's some information about a drug that might be discussed for such conditions:

        **Drug Option: [Drug Name]** 
        *   **What it is (Simplified):** [e.g., "This is a type of medication called a 'targeted therapy' that works by..."]
        *   **Commonly Used For:** [e.g., "Certain types of [Cancer Type] that have [specific biomarker, if relevant and mentioned by user]"]
        *   **What Studies Suggest (Simplified):** [e.g., "In some clinical studies for [cancer type studied in data], this drug, when [used how - e.g., combined with X / used after Y], showed it could help [key patient-friendly outcome - e.g., slow down cancer growth for a period / shrink tumors in some patients]."] 
        *   **Important to Know:** "All medications have potential benefits and risks, and whether this or any drug is appropriate depends on many individual factors."
        
        (If presenting a second drug, use a separator like "---" and repeat the format)
        ```

    *   **Clinical Trial Information Presentation Format (Strict Adherence):**
        ```markdown
        Regarding clinical trials for [User's Diagnosis/Context], here's one that appears relevant:

        **Clinical Trial Focus: [Brief, patient-friendly summary of trial's main goal, e.g., "Testing a new drug combination"]**
        *   **Official Title Snippet:** "[Shortened, understandable part of the title_from_study]"
        *   **Trial ID:** [NCT ID]
        *   **Purpose of this Study (Simplified):** [e.g., "Researchers are looking to see if [intervention] is more helpful than the usual treatment for people with [condition studied]."]
        *   **Current Status:** [Status - e.g., "Recruiting participants", "Active, but not recruiting"]
        *   **Trial Phase:** [Phase - e.g., "Phase 3". Briefly explain if not common knowledge: e.g., "(Phase 3 trials compare new treatments to standard ones in larger groups of people)"]
        *   **Key Things Being Studied:** [Interventions_studied, simplified]
        *   **Approx. Distance (if applicable):** [e.g., "One or more study sites are located about X miles from the provided zip code."] (ONLY if location search was done by user request and distance is known. Otherwise, omit this line.)
        *   **More Information:** "You can find more detailed official information about this trial at: [Official URL]"
        
        (If presenting a second trial, use a separator like "---" and repeat the format)
        ```

5.  **MANDATORY DISCLAIMER (Verbatim at the end of relevant responses):**
    *   After providing any drug or trial information, ALWAYS conclude with:
        "**Important Reminder:** This information is for educational purposes only and is NOT medical advice. It's essential to discuss all of this with your doctor or a qualified healthcare professional. They are the only ones who can understand your complete medical situation and provide guidance on your care."

6.  **Follow-up Suggestions (JSON Format - Strict Adherence):**
    *   After the disclaimer, if appropriate, provide 2-3 *relevant and conversational* follow-up questions the user might have.
    *   Format them EXACTLY like this in a JSON block. The application will parse this:
        `[PATIENT_SUGGESTIONS_JSON_START]`
        `{"suggestions": ["Tell me more about the side effects of [Drug Name mentioned].", "What does 'Phase 3' mean for that trial?", "Are there trials for [different but related aspect]?"]}`
        `[PATIENT_SUGGESTIONS_JSON_END]`
        (These suggestions MUST be tailored to what was just discussed.)

7.  **Error/No Results Handling:**
    *   If tools return errors or no relevant data: "I wasn't able to find specific [drug/trial] information matching [details of query] right now. This can sometimes happen if the criteria are very specific or if data isn't available in the resources I can access. Would you like me to try a broader search, or perhaps refine the search terms?"

8.  **Scope:** Stick to cancer drug and clinical trial information. For other topics, politely state it's outside your scope and suggest consulting a relevant professional or their doctor.

Remember, clarity, empathy, and accuracy (within the bounds of an informational assistant) are paramount.
You have access to tools to fetch data. Use them thoughtfully. Your final output to the user should always be a complete, well-formatted, and patient-friendly response.
""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"), # For tool calls and observations
        ]
    )
    # max_tokens_to_sample for Claude via Anthropic SDK, use max_tokens for some other Langchain integrations
    llm = ChatAnthropic(
        model=CLAUDE_MODEL_NAME, 
        temperature=0.2, # Lower temperature for more factual, less creative responses
        api_key=ANTHROPIC_API_KEY, 
        # max_tokens_to_sample=4096, # Increased for potentially long formatted outputs + tool use
        # default_request_timeout=30.0 # seconds
    )
    
    agent = create_tool_calling_agent(llm, available_tools, prompt_template)
    # handle_parsing_errors can take a string message, a custom function, or True/False
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=available_tools, 
        verbose=True, # Set to False for production deployment unless debugging
        handle_parsing_errors="I apologize, I had a bit of trouble understanding that. Could you please rephrase?", # More user-friendly
        max_iterations=10 # Prevent runaway loops
    )
    logger.info("Langchain agent executor initialized.")
    return agent_executor

# --- Streamlit UI ---
# Initialize session state variables
if "messages" not in st.session_state: 
    st.session_state.messages = []
if "chat_history_for_agent" not in st.session_state: 
    st.session_state.chat_history_for_agent = []
if "agent_executor" not in st.session_state and DATA_LOADED_SUCCESSFULLY: # Lazy load agent
    st.session_state.agent_executor = get_langchain_agent_executor()
if "current_input" not in st.session_state: # Used to manage clicked suggestions
    st.session_state.current_input = ""
if "clicked_suggestion" not in st.session_state:
    st.session_state.clicked_suggestion = None

# --- Sidebar ---
with st.sidebar:
    st.title("AI Assistant Controls")
    st.markdown("---") # Visual separator
    
    if DATA_LOADED_SUCCESSFULLY:
        if st.button("🔄 Start New Conversation", use_container_width=True, type="primary", help="Clears the current chat and starts fresh."):
            st.session_state.messages = []
            st.session_state.chat_history_for_agent = []
            # Re-initialize agent to ensure fresh state if it has any internal memory not passed via invoke
            st.session_state.agent_executor = get_langchain_agent_executor() 
            st.session_state.current_input = ""
            st.session_state.clicked_suggestion = None
            logger.info("Conversation restarted by user.")
            st.success("New conversation started!")
            st.rerun() # Force rerun to clear main page content
    else:
        st.warning("Application data failed to load. Restart is unavailable.")

    st.markdown("---")
    st.subheader("Important Note")
    st.info(
        "This AI provides information to help you learn. It is **not a substitute for medical advice** from "
        "your doctor or other qualified healthcare professionals. Always discuss your health and treatment "
        "options with them."
    )
    st.caption(f"AI Model: `{CLAUDE_MODEL_NAME}`")

# --- Main Chat Interface ---
st.title("AI Cancer Information Assistant 🧑‍⚕️")
st.caption("Your guide for exploring information on cancer drugs and clinical trials. Remember to always consult your doctor for medical advice.")
st.markdown("---") # Visual separator

if DATA_LOADED_SUCCESSFULLY:
    # Display chat messages from history
    for message_data in st.session_state.messages: # Renamed 'message' to 'message_data'
        avatar_icon = ASSISTANT_AVATAR if message_data["role"] == "assistant" else USER_AVATAR # Renamed 'avatar' to 'avatar_icon'
        with st.chat_message(message_data["role"], avatar=avatar_icon):
            st.markdown(message_data["content"], unsafe_allow_html=True) # Allow LLM-generated Markdown

    # Handle if a suggestion was clicked
    if st.session_state.clicked_suggestion:
        st.session_state.current_input = st.session_state.clicked_suggestion
        st.session_state.clicked_suggestion = None # Reset after transferring
        # This will be picked up by the process_prompt logic below
    
    # Get user input (either typed or from a clicked suggestion that populates current_input)
    user_typed_prompt_str = st.chat_input("Ask me about cancer treatments, drugs, or trials...", key="main_chat_input") # Renamed 'user_prompt' to 'user_typed_prompt_str'

    prompt_to_process = None # Renamed 'process_prompt' to 'prompt_to_process'
    if st.session_state.current_input: # Input came from a clicked suggestion
        prompt_to_process = st.session_state.current_input
        st.session_state.current_input = "" # Clear after use
    elif user_typed_prompt_str: # Input came from user typing
        prompt_to_process = user_typed_prompt_str
    
    # If there's a prompt to process (either typed or from suggestion)
    if prompt_to_process:
        st.session_state.messages.append({"role": "user", "content": prompt_to_process})
        # Display user message immediately
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt_to_process)

        # Process with LLM agent
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            message_response_placeholder = st.empty() # Renamed 'message_placeholder' to 'message_response_placeholder'
            final_assistant_response = "" # Renamed 'full_response_content' to 'final_assistant_response'
            
            with st.spinner("🧑‍⚕️ Thinking and gathering information for you..."):
                try:
                    agent_invocation_input = { # Renamed 'agent_input' to 'agent_invocation_input'
                        "input": prompt_to_process, 
                        "chat_history": st.session_state.chat_history_for_agent
                    }
                    
                    # Ensure agent_executor is initialized (important if app was reloaded without full data load)
                    if 'agent_executor' not in st.session_state:
                         logger.warning("Agent executor was not in session state. Re-initializing.")
                         st.session_state.agent_executor = get_langchain_agent_executor()

                    agent_response_data = st.session_state.agent_executor.invoke(agent_invocation_input) # Renamed 'response_data' to 'agent_response_data'
                    
                    # Robustly extract the string output from the agent
                    if isinstance(agent_response_data, dict) and "output" in agent_response_data:
                        raw_ai_output = agent_response_data["output"]
                    elif isinstance(agent_response_data, str):
                        raw_ai_output = agent_response_data
                    else:
                        logger.error(f"Unexpected agent output structure: {type(agent_response_data)} - {agent_response_data}")
                        raw_ai_output = "I apologize, I encountered an issue formulating a complete response. Could you try asking in a different way?"

                    # Ensure it's a string before proceeding
                    if not isinstance(raw_ai_output, str):
                        logger.warning(f"Agent output ('raw_ai_output') is not a string: {type(raw_ai_output)} - {raw_ai_output}. Attempting to coerce.")
                        # Attempt to handle common non-string outputs, e.g., list of dicts from some tool formats if parsing fails higher up
                        if isinstance(raw_ai_output, list) and len(raw_ai_output) > 0 and isinstance(raw_ai_output[0], dict) and 'text' in raw_ai_output[0]:
                            final_assistant_response = raw_ai_output[0]['text']
                        else:
                            final_assistant_response = str(raw_ai_output) # Last resort
                            logger.error(f"Could not convert agent output to a clean string. Final raw form: {final_assistant_response}")
                    else:
                        final_assistant_response = raw_ai_output
                    
                    # Update Langchain-specific chat history
                    st.session_state.chat_history_for_agent.append(HumanMessage(content=prompt_to_process))
                    st.session_state.chat_history_for_agent.append(AIMessage(content=final_assistant_response)) # Store the final string
                    
                    # Limit history length
                    max_history_pairs_count = 7  # Renamed 'max_history_pairs' to 'max_history_pairs_count'
                    if len(st.session_state.chat_history_for_agent) > max_history_pairs_count * 2:
                        st.session_state.chat_history_for_agent = st.session_state.chat_history_for_agent[-(max_history_pairs_count * 2):]

                except Exception as agent_error: # Renamed 'e' to 'agent_error'
                    logger.error(f"Critical error during agent invocation: {agent_error}", exc_info=True)
                    error_detail_snippet = str(agent_error)[:150] # Truncate long error messages for display
                    final_assistant_response = (
                        "I'm truly sorry, but I encountered a significant technical issue while trying to process your request. "
                        "The team has been notified. Please try again in a little while, or rephrase your question.\n\n"
                        f"(Technical detail: {error_detail_snippet}...)"
                    )
                    # Still add to history for context if a retry happens
                    st.session_state.chat_history_for_agent.append(HumanMessage(content=prompt_to_process))
                    st.session_state.chat_history_for_agent.append(AIMessage(content=f"System error during processing: {agent_error}")) # Log system error internally
            
            message_response_placeholder.markdown(final_assistant_response, unsafe_allow_html=True)
        
        # Add assistant's final response to the display message list
        st.session_state.messages.append({"role": "assistant", "content": final_assistant_response})

        # --- Parse and Display Suggestions ---
        extracted_suggestions_list = [] # Renamed 'parsed_suggestions' to 'extracted_suggestions_list'
        try:
            if isinstance(final_assistant_response, str): # Ensure it's a string before regex
                # Regex to find the JSON block for suggestions based on new prompt
                suggestion_json_block_match = re.search( # Renamed 'suggestion_json_match' to 'suggestion_json_block_match'
                    r"\[PATIENT_SUGGESTIONS_JSON_START\](.*?)\[PATIENT_SUGGESTIONS_JSON_END\]", 
                    final_assistant_response, 
                    re.DOTALL | re.IGNORECASE
                )
                if suggestion_json_block_match:
                    json_content_str = suggestion_json_block_match.group(1).strip() # Renamed 'json_str' to 'json_content_str'
                    try:
                        suggestions_json_data = json.loads(json_content_str) # Renamed 'suggestions_data' to 'suggestions_json_data'
                        if isinstance(suggestions_json_data, dict) and \
                           "suggestions" in suggestions_json_data and \
                           isinstance(suggestions_json_data["suggestions"], list):
                            extracted_suggestions_list = [str(sug) for sug in suggestions_json_data["suggestions"] if str(sug).strip()] # Ensure suggestions are strings and not empty
                    except json.JSONDecodeError as json_decode_err: # Renamed 'je' to 'json_decode_err'
                        logger.warning(f"Failed to parse suggestions JSON: {json_decode_err}. JSON string was: '{json_content_str}'")
            else:
                logger.warning(f"final_assistant_response was not a string, skipping suggestion parsing. Type: {type(final_assistant_response)}")
        
        except TypeError as type_err_suggestions: # Renamed 'te' to 'type_err_suggestions'
             logger.error(f"TypeError while searching for suggestions; final_assistant_response might not be a string: {type_err_suggestions}. Content type: {type(final_assistant_response)}", exc_info=True)

        if extracted_suggestions_list:
            st.markdown("---") 
            st.subheader("💭 What would you like to explore next?")
            
            num_suggestions_to_display = min(len(extracted_suggestions_list), 3) # Renamed 'num_suggestions_to_show' to 'num_suggestions_to_display'
            if num_suggestions_to_display > 0:
                suggestion_columns = st.columns(num_suggestions_to_display) # Renamed 'cols' to 'suggestion_columns'
                for idx, suggestion_item_text in enumerate(extracted_suggestions_list[:num_suggestions_to_display]): # Renamed 'i' to 'idx', 'suggestion_text' to 'suggestion_item_text'
                    # Create a more robust unique key for buttons to prevent state issues on reruns
                    button_key_str = f"suggestion_btn_{idx}_{int(time.time())}_{hash(suggestion_item_text)}" # Renamed 'button_key' to 'button_key_str'
                    if suggestion_columns[idx].button(f"{suggestion_item_text}", key=button_key_str, use_container_width=True, help=f"Click to ask: {suggestion_item_text}"):
                        st.session_state.clicked_suggestion = suggestion_item_text
                        logger.info(f"Suggestion clicked: '{suggestion_item_text}'")
                        st.rerun() # Rerun to process the clicked suggestion as the new user input
        
        # Rerun to update the full chat display if a prompt was processed (either user typed or from suggestion)
        st.rerun()


    # Initial greeting message if no messages exist and data loaded successfully
    if not st.session_state.messages and DATA_LOADED_SUCCESSFULLY:
        initial_greeting_text = ( # Renamed 'initial_greeting' to 'initial_greeting_text'
            "Hello! I'm an AI assistant here to help you explore information about cancer treatments, including "
            "drugs and clinical trials. My goal is to provide clear and understandable information.\n\n"
            "**How can I help you today?** For example, you could tell me about a specific diagnosis you're "
            "interested in, like 'Stage IV Lung Cancer with EGFR mutation'."
        )
        st.session_state.messages.append({"role": "assistant", "content": initial_greeting_text})
        st.session_state.chat_history_for_agent.append(AIMessage(content=initial_greeting_text))
        logger.info("Initial greeting displayed to user.")
        st.rerun() # Rerun to display the initial greeting

elif not DATA_LOADED_SUCCESSFULLY:
    # This message is displayed if DATA_LOADED_SUCCESSFULLY is False after the initial load attempt.
    # The more specific error from the loading block should already be visible via st.error.
    st.warning(
        "The Cancer Information Assistant could not start correctly due to issues loading essential data or AI models. "
        "Please review any error messages above and ensure your setup is correct. If the problem persists, "
        "please contact support."
    )

st.markdown("---") # Final separator
st.caption(
    "This AI tool is for informational purposes and does not provide medical advice. "
    "Always consult your doctor or healthcare team for any health-related decisions."
)