# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------
# AI Drug & Trial Match (Interactive Assistant)
# --------------------------------------------------------------------------

# 1. SET PAGE CONFIG FIRST - THIS IS CRITICAL
import streamlit as st
st.set_page_config(
    page_title="AI Drug & Trial Match",
    page_icon="🧑‍⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None, # Replace with actual help URL or remove
        'Report a bug': None, # Replace with actual bug report URL or remove
        'About': """
        ## AI Drug & Trial Match Assistant
        This application helps explore information about cancer drugs and clinical trials.
        **Disclaimer:** This tool is for informational purposes only and does not provide medical advice.
        Always consult with a qualified healthcare professional for medical concerns.
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
import ast
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- API Key Check ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    st.error(
        "🚨 **FATAL ERROR: ANTHROPIC_API_KEY not found!**\n\n"
        "Please set the `ANTHROPIC_API_KEY` environment variable in your `.env` file "
        "or system environment and restart the application."
    )
    logging.critical("ANTHROPIC_API_KEY not found. Application cannot proceed.")
    st.stop()

# --- File Paths & Constants ---
# (Keep existing file paths and constants as they were, e.g., DRUG_DATA_CSV, TRIAL_DATA_XLSX, etc.)
# --- File Paths ---
DRUG_DATA_CSV = 'drug_data.csv'
TRIAL_DATA_XLSX = 'trials_filtered_with_coordinates.xlsx' # Ensure this has 'location_coordinates'
DRUG_EMBEDDINGS_FILE = 'drug_embeddings.npy'
TRIAL_EMBEDDINGS_FILE = 'trial_embeddings.npy'
TRIAL_INDEX_MAP_FILE = 'trial_index_map.pkl'
GEOCODE_CACHE_FILE_PATH = "geocode_cache.json"
TEMP_GEOCODE_CACHE_FILE_PATH = "geocode_cache.tmp.json"

# --- Constants ---
EMBEDDING_MODEL_NAME = 'neuml/pubmedbert-base-embeddings'
DRUG_TEXT_COLUMNS_FOR_EMBEDDING = ['Cancer Type']
TRIAL_TEXT_COLUMNS_FOR_EMBEDDING = ['Study Type', 'Conditions']

TRIAL_FILTER_PRIMARY_OUTCOME_COLUMN = 'Primary Outcome Measures'
TRIAL_FILTER_PRIMARY_OUTCOME_TERM = 'Overall Survival'
TRIAL_FILTER_PHASES_COLUMN = 'Phases'
TRIAL_ACCEPTABLE_PHASES_STR = ['PHASE1|PHASE2', 'PHASE2', 'PHASE2|PHASE3', 'PHASE3', 'PHASE4']
TRIAL_ACCEPTABLE_INDIVIDUAL_PHASES = set()
for phase_combo in TRIAL_ACCEPTABLE_PHASES_STR:
    for phase in re.split(r'[|/,\s]+', phase_combo):
        if phase: TRIAL_ACCEPTABLE_INDIVIDUAL_PHASES.add(phase.strip().upper())
TRIAL_FILTER_STUDY_TYPE_COLUMN = 'Study Type'
TRIAL_FILTER_STUDY_TYPE_VALUE = 'INTERVENTIONAL'

CLAUDE_MODEL_NAME = "claude-3-haiku-20240307" # Fast and capable
DEFAULT_SEARCH_RADIUS_MILES = 50

# IMPORTANT: Update NOMINATIM_USER_AGENT for geocoding
NOMINATIM_USER_AGENT = "ai_health_connect/1.2" # PLEASE UPDATE to unique string
API_REQUEST_DELAY_SECONDS = 1.05
API_TIMEOUT_SECONDS = 15

ASSISTANT_AVATAR = "🧑‍⚕️"
USER_AVATAR = "👤"
# --- Helper Functions (Parsing, Sorting - Keep as is from previous correct version) ---
def parse_time_to_months(time_str):
    if isinstance(time_str, (int, float)): return float(time_str)
    if not isinstance(time_str, str): return None
    time_str = str(time_str).strip().lower()
    if time_str in ['n/a', 'not applicable', 'not reported', 'not reached', 'nr', '', 'nan']: return None
    match_months = re.match(r'(\d+(\.\d+)?)\s*m', time_str)
    if match_months: return float(match_months.group(1))
    match_years = re.match(r'(\d+(\.\d+)?)\s*y', time_str)
    if match_years: return float(match_years.group(1)) * 12
    try: return float(time_str)
    except ValueError: return None

def parse_improvement_percentage(perc_str):
    if isinstance(perc_str, (int, float)): return float(perc_str)
    if not isinstance(perc_str, str): return None
    perc_str = str(perc_str).strip().lower()
    if perc_str in ['n/a', 'not applicable', 'not reported', 'not statistically significant', 'nss', '', 'nan']: return None
    match = re.match(r'(-?\d+(\.\d+)?)\s*%', perc_str)
    if match: return float(match.group(1))
    try: return float(perc_str)
    except ValueError: return None

def sort_key_with_none(value, reverse=True):
    is_none_or_nan = value is None or (isinstance(value, float) and math.isnan(value))
    if is_none_or_nan:
        return float('-inf') if reverse else float('inf')
    try: return float(value)
    except (ValueError, TypeError): return float('-inf') if reverse else float('inf')

def check_phases(trial_phases_raw):
    if not isinstance(trial_phases_raw, str) or not str(trial_phases_raw).strip(): return False
    trial_individual_phases = re.split(r'[|/,\s]+', str(trial_phases_raw).strip())
    return any(phase.strip().upper() in TRIAL_ACCEPTABLE_INDIVIDUAL_PHASES for phase in trial_individual_phases if phase)


# --- Data Loading and Preprocessing Functions (Keep as is) ---
@st.cache_resource(show_spinner="Initializing AI model...")
def load_sentence_transformer_model(model_name=EMBEDDING_MODEL_NAME):
    try:
        model = SentenceTransformer(model_name)
        logging.info(f"Embedding model '{model_name}' loaded.")
        return model
    except Exception as e:
        logging.error(f"Fatal: Error loading Sentence Transformer model '{model_name}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to load embedding model: {e}")

@st.cache_data(show_spinner="Loading drug data...")
def load_and_preprocess_drug_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if not all(col in df.columns for col in DRUG_TEXT_COLUMNS_FOR_EMBEDDING):
            raise ValueError(f"Missing required columns in {csv_path}")
        df['combined_text_for_embedding'] = df[DRUG_TEXT_COLUMNS_FOR_EMBEDDING].fillna('').astype(str).agg(' '.join, axis=1)
        df['Treatment_OS_Months_Parsed'] = df.get('Treatment_OS', pd.Series([None]*len(df))).apply(parse_time_to_months)
        df['Control_OS_Months_Parsed'] = df.get('Control_OS', pd.Series([None]*len(df))).apply(parse_time_to_months)
        df = df.fillna('N/A')
        logging.info(f"Drug data loaded and preprocessed. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Fatal: Error loading/processing drug data '{csv_path}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to load drug data: {e}")

@st.cache_data(show_spinner="Generating/loading drug embeddings...")
def get_or_generate_drug_embeddings(_drug_df, _model, embeddings_path=DRUG_EMBEDDINGS_FILE):
    if os.path.exists(embeddings_path):
        try:
            embeddings = np.load(embeddings_path)
            if embeddings.shape[0] == len(_drug_df) and embeddings.shape[1] == _model.get_sentence_embedding_dimension():
                logging.info("Drug embeddings loaded from cache.")
                return embeddings
            logging.warning("Drug embeddings shape mismatch. Regenerating.")
        except Exception as e:
            logging.error(f"Error loading drug embeddings: {e}. Regenerating.", exc_info=True)
    try:
        texts = _drug_df['combined_text_for_embedding'].tolist()
        if not texts: return np.array([])
        embeddings = _model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        np.save(embeddings_path, embeddings)
        logging.info("Drug embeddings generated and saved.")
        return embeddings
    except Exception as e:
        logging.error(f"Fatal: Error generating drug embeddings: {e}", exc_info=True)
        raise RuntimeError(f"Failed to generate drug embeddings: {e}")

@st.cache_data(show_spinner="Loading clinical trial data...")
def load_and_preprocess_trial_data(xlsx_path):
    try:
        df = pd.read_excel(xlsx_path)
        if not all(col in df.columns for col in TRIAL_TEXT_COLUMNS_FOR_EMBEDDING):
            raise ValueError(f"Missing required columns in {xlsx_path}")
        df['combined_text_for_embedding'] = df[TRIAL_TEXT_COLUMNS_FOR_EMBEDDING].fillna('').astype(str).agg(' '.join, axis=1)

        if 'location_coordinates' in df.columns:
            def parse_excel_coords(coord_str_val):
                if pd.isna(coord_str_val) or not isinstance(coord_str_val, str) or not coord_str_val.strip(): return []
                try:
                    parsed = ast.literal_eval(coord_str_val)
                    if isinstance(parsed, list):
                        return [tuple(item) for item in parsed if isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(num, (int, float)) for num in item)]
                    return []
                except (ValueError, SyntaxError, TypeError): return []
            df['parsed_location_coordinates'] = df['location_coordinates'].apply(parse_excel_coords)
        else:
            logging.warning("'location_coordinates' column not found in trial data. Location-based search will be impaired.")
            df['parsed_location_coordinates'] = pd.Series([[] for _ in range(len(df))])
        
        df = df.fillna('N/A')
        logging.info(f"Trial data loaded and preprocessed. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Fatal: Error loading/processing trial data '{xlsx_path}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to load trial data: {e}")

@st.cache_data(show_spinner="Generating/loading trial embeddings...")
def get_or_generate_trial_embeddings(_trial_df, _model, embeddings_path=TRIAL_EMBEDDINGS_FILE, map_path=TRIAL_INDEX_MAP_FILE):
    if os.path.exists(embeddings_path) and os.path.exists(map_path):
        try:
            embeddings = np.load(embeddings_path)
            with open(map_path, 'rb') as f: index_map = pickle.load(f)
            if (embeddings is not None and index_map is not None and
                embeddings.shape[0] == len(index_map) and
                embeddings.shape[1] == _model.get_sentence_embedding_dimension() and
                (max(index_map.keys(), default=-1) < len(_trial_df) if index_map else True)):
                logging.info("Trial embeddings and index map loaded from cache.")
                return embeddings, index_map
            logging.warning("Trial embeddings/map mismatch. Regenerating.")
        except Exception as e:
            logging.error(f"Error loading trial embeddings/map: {e}. Regenerating.", exc_info=True)
    try:
        mask = _trial_df['combined_text_for_embedding'].str.strip() != ''
        indices = _trial_df.index[mask].tolist()
        texts = _trial_df.loc[indices, 'combined_text_for_embedding'].tolist()
        if not texts: return np.array([]), {}
        embeddings = _model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        index_map = {original_idx: emb_idx for emb_idx, original_idx in enumerate(indices)}
        np.save(embeddings_path, embeddings)
        with open(map_path, 'wb') as f: pickle.dump(index_map, f)
        logging.info("Trial embeddings/map generated and saved.")
        return embeddings, index_map
    except Exception as e:
        logging.error(f"Fatal: Error generating trial embeddings: {e}", exc_info=True)
        raise RuntimeError(f"Failed to generate trial embeddings: {e}")

# --- Global Data Initialization (Keep as is) ---
DATA_LOADED_SUCCESSFULLY = False
try:
    embedding_model_global = load_sentence_transformer_model()
    df_drugs_processed_global = load_and_preprocess_drug_data(DRUG_DATA_CSV)
    drug_embeddings_array_global = get_or_generate_drug_embeddings(df_drugs_processed_global, embedding_model_global)
    df_trials_processed_global = load_and_preprocess_trial_data(TRIAL_DATA_XLSX)
    trial_embeddings_array_global, trial_index_map_global = get_or_generate_trial_embeddings(df_trials_processed_global, embedding_model_global)
    DATA_LOADED_SUCCESSFULLY = True
    logging.info("All critical data and models initialized successfully.")
    if NOMINATIM_USER_AGENT == "ai_drug_trial_matcher_app/1.2": # Default placeholder
        st.sidebar.warning(
            "**Geocoding Alert:**\n\n"
            "The `NOMINATIM_USER_AGENT` is set to a default value. "
            "Please update it in the script to a unique identifier for your application "
            "(e.g., 'YourAppName/1.0 yourcontact@example.com') to ensure reliable geocoding services."
        )
except RuntimeError as e:
    st.error(
        f"🚨 **APPLICATION STARTUP FAILED** 🚨\n\n"
        f"A critical error occurred during data or model initialization:\n\n`{e}`\n\n"
        "Please check the console logs for more details. The application cannot continue."
    )
    logging.critical(f"Application startup failed due to: {e}", exc_info=True)


# --- Dynamic Top-N Selection & Geocoding Cache (Keep as is) ---
def _determine_top_n_results(scored_results_list):
    if not scored_results_list: return 0
    count = len(scored_results_list)
    if count <= 10: return count
    score_at_10th = scored_results_list[9]['semantic_similarity'] if count > 9 else 0 
    if score_at_10th >= 0.7: return min(count, 20)
    if score_at_10th < 0.5: return 10
    return min(count, 15)

def load_persistent_geocode_cache(filepath=GEOCODE_CACHE_FILE_PATH):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f: return json.load(f)
        except: pass
    return {}

def save_persistent_geocode_cache(cache_data, target_path=GEOCODE_CACHE_FILE_PATH, temp_path=TEMP_GEOCODE_CACHE_FILE_PATH):
    try:
        with open(temp_path, 'w') as f: json.dump(cache_data, f, indent=2)
        if os.path.exists(target_path): os.remove(target_path)
        os.rename(temp_path, target_path)
    except Exception as e:
        logging.warning(f"Could not save geocode cache: {e}")

# --- Langchain Tools Definition (Keep schema, adjust tool descriptions for clarity if needed) ---
class FindDrugsInput(BaseModel):
    diagnosis: str = Field(description="The primary diagnosis, e.g., Breast Cancer")
    stage: str = Field(description="The stage or progression, e.g., Stage IV, Metastatic")
    biomarkers: str = Field(description="Known biomarkers, comma-separated, or 'None', e.g., HR-positive, PD-L1 high")

@tool("find_drugs_tool", args_schema=FindDrugsInput)
def find_drugs_tool(diagnosis: str, stage: str, biomarkers: str) -> str:
    """
    Finds relevant drug studies. Input: diagnosis, stage, biomarkers.
    Returns a JSON string of top matching drugs with details.
    The LLM should review these for relevance and summarize key findings for the user.
    DO NOT present similarity scores to the user.
    """
    logging.info(f"Tool: find_drugs_tool: Dx='{diagnosis}', Stg='{stage}', Bio='{biomarkers}'")
    if drug_embeddings_array_global.size == 0:
        return json.dumps({"error": "Drug data or embeddings are not available.", "drugs": []})

    query = f"{diagnosis} {stage} {biomarkers}".strip()
    if not query: return json.dumps({"error": "Drug search query is empty.", "drugs": []})

    try:
        q_emb = embedding_model_global.encode(query, convert_to_numpy=True)
        sims = cosine_similarity([q_emb], drug_embeddings_array_global)[0]
    except Exception as e:
        logging.error(f"Drug embedding/similarity error: {e}")
        return json.dumps({"error": f"Error during drug similarity calculation: {e}", "drugs": []})

    results = []
    for i, row in df_drugs_processed_global.iterrows():
        if i >= len(sims): continue
        # The LLM will decide final relevance. Provide a reasonable pool.
        # We can keep a slightly lower threshold here as LLM does final filtering.
        if sims[i] >= 0.4: # Adjusted threshold for pooling if LLM reviews
            results.append({
                'drug_name': row.get('Drug Name', 'N/A'),
                'cancer_type_studied': row.get('Cancer Type', 'N/A'),
                'treatment_os': row.get('Treatment_OS', 'N/A'),
                'control_os': row.get('Control_OS', 'N/A'),
                'os_improvement_percent': row.get('OS_Improvement (%)', 'N/A'), # LLM can interpret this
                'treatment_pfs': row.get('Treatment_PFS', 'N/A'),
                'control_pfs': row.get('Control_PFS', 'N/A'),
                'pfs_improvement_percent': row.get('PFS_Improvement (%)', 'N/A'),
                'brief_summary': textwrap.shorten(str(row.get('Brief Study Summary', '')), 250, placeholder="..."),
                'semantic_similarity': round(float(sims[i]), 3), # For LLM's internal use if needed
                '_source_data_index': i # Internal reference
            })
    results.sort(key=lambda x: x['semantic_similarity'], reverse=True)
    
    num_to_select = _determine_top_n_results(results)
    final_results = results[:num_to_select]
    
    if not final_results:
        return json.dumps({"message": "No drug studies found closely matching the criteria after initial selection.", "drugs": []})
    # Remove similarity score before sending to LLM if it shouldn't influence its summary other than ranking
    # for r in final_results: del r['semantic_similarity'] # Or keep it for LLM to use for relevance judgment
    return json.dumps({"drugs": final_results, "count": len(final_results)})

class FindClinicalTrialsInput(BaseModel):
    diagnosis: str = Field(description="Primary diagnosis, e.g., Lung Cancer")
    stage: str = Field(description="Stage/progression, e.g., Advanced, Recurrent")
    biomarkers: str = Field(description="Biomarkers (comma-separated or 'None'), e.g., EGFR mutation")
    user_latitude: float = Field(None, description="User's latitude. Optional.")
    user_longitude: float = Field(None, description="User's longitude. Optional.")
    radius_miles: int = Field(DEFAULT_SEARCH_RADIUS_MILES, description="Search radius in miles. Optional.")

@tool("find_clinical_trials_tool", args_schema=FindClinicalTrialsInput)
def find_clinical_trials_tool(diagnosis: str, stage: str, biomarkers: str, user_latitude: float = None, user_longitude: float = None, radius_miles: int = DEFAULT_SEARCH_RADIUS_MILES) -> str:
    """
    Finds relevant clinical trials. Input: diagnosis, stage, biomarkers, optional location.
    Filters for Overall Survival, specific phases, interventional type.
    Returns JSON string of top trials with details.
    The LLM should review for relevance and summarize key findings for the user.
    DO NOT present similarity scores to the user. Include distance if location was used.
    """
    logging.info(f"Tool: find_clinical_trials_tool: Dx='{diagnosis}', Stg='{stage}', Bio='{biomarkers}', Loc=({user_latitude},{user_longitude}), Rad={radius_miles}")
    if trial_embeddings_array_global.size == 0 or not trial_index_map_global:
        return json.dumps({"error": "Trial data, embeddings, or map unavailable.", "trials": []})

    query = f"{diagnosis} {stage} {biomarkers}".strip()
    if not query: return json.dumps({"error": "Trial search query empty.", "trials": []})
    
    user_loc = (user_latitude, user_longitude) if user_latitude is not None and user_longitude is not None else None

    try: q_emb = embedding_model_global.encode(query, convert_to_numpy=True)
    except Exception as e:
        logging.error(f"Trial query embedding error: {e}")
        return json.dumps({"error": f"Error embedding trial query: {e}", "trials": []})

    results = []
    for i, row in df_trials_processed_global.iterrows():
        if not (str(row.get(TRIAL_FILTER_PRIMARY_OUTCOME_COLUMN, '')).lower().count(TRIAL_FILTER_PRIMARY_OUTCOME_TERM.lower()) > 0 and \
                check_phases(str(row.get(TRIAL_FILTER_PHASES_COLUMN, ''))) and \
                str(row.get(TRIAL_FILTER_STUDY_TYPE_COLUMN, '')).upper() == TRIAL_FILTER_STUDY_TYPE_VALUE.upper() and \
                i in trial_index_map_global and trial_index_map_global[i] < len(trial_embeddings_array_global)):
            continue
        
        sim = cosine_similarity([q_emb], [trial_embeddings_array_global[trial_index_map_global[i]]])[0][0]
        if sim < 0.4: continue # Adjusted threshold for pooling

        trial_info = {
            'nct_id': row.get('NCT Number', 'N/A'),
            'title': textwrap.shorten(str(row.get('Study Title', 'N/A')), 150, placeholder="..."),
            'status': row.get('Study Status', 'N/A'), 
            'phases': row.get(TRIAL_FILTER_PHASES_COLUMN, 'N/A'),
            'conditions': textwrap.shorten(str(row.get('Conditions', 'N/A')), 100, placeholder="..."),
            'interventions': textwrap.shorten(str(row.get('Interventions', 'N/A')), 100, placeholder="..."),
            'brief_summary': textwrap.shorten(str(row.get('Brief Summary', '')), 250, placeholder="..."),
            'url': f"https://clinicaltrials.gov/study/{row.get('NCT Number', '')}" if row.get('NCT Number', 'N/A') != 'N/A' else None,
            'semantic_similarity': round(float(sim), 3), # For LLM's internal use
            'distance_miles': None, 
            '_source_data_index': i
        }

        if user_loc:
            trial_sites = row.get('parsed_location_coordinates', [])
            min_dist = float('inf')
            for site_coords in trial_sites:
                try:
                    dist = geodesic(user_loc, site_coords).miles
                    if dist < min_dist: min_dist = dist
                except: pass 
            
            if min_dist <= radius_miles:
                trial_info['distance_miles'] = round(min_dist, 1)
                results.append(trial_info)
            elif not any(r.get('distance_miles') is not None for r in results) and min_dist != float('inf'): # If no matches yet in radius, consider closest
                 trial_info['distance_miles'] = round(min_dist, 1)
                 results.append(trial_info)
        else:
            results.append(trial_info)
            
    if user_loc: results.sort(key=lambda x: (sort_key_with_none(x['distance_miles'], False), -x['semantic_similarity']))
    else: results.sort(key=lambda x: x['semantic_similarity'], reverse=True)

    num_to_select = _determine_top_n_results(results)
    final_results = results[:num_to_select]

    if not final_results:
        return json.dumps({"message": "No trials found closely matching criteria after dynamic selection.", "trials": []})
    return json.dumps({"trials": final_results, "count": len(final_results)})


class ZipToCoordinatesInput(BaseModel):
    zip_code: str = Field(description="User's zip code, e.g., 90210")
    country_code: str = Field("US", description="Country code for the zip code, e.g., US, CA. Defaults to US.")

@tool("zip_to_coordinates_tool", args_schema=ZipToCoordinatesInput)
def zip_to_coordinates_tool(zip_code: str, country_code: str = "US") -> str:
    """
    Converts zip code to geographic coordinates (latitude, longitude).
    Uses a persistent cache. Returns JSON with coordinates or error.
    """
    # This tool remains the same as it's utility function
    logging.info(f"Tool: zip_to_coordinates_tool: Zip='{zip_code}', Country='{country_code}'")
    cache = load_persistent_geocode_cache()
    key = f"{zip_code}_{country_code}".lower()
    if key in cache and cache[key]:
        logging.info(f"Zip cache hit for '{key}'. Coords: {cache[key]}")
        return json.dumps({"latitude": cache[key][0], "longitude": cache[key][1], "status": "success_cache"})

    geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT)
    warning_msg = ""
    if NOMINATIM_USER_AGENT == "ai_drug_trial_matcher_app/1.2": # Default placeholder
         warning_msg = "Warning: Nominatim user agent is default. Geocoding may be unreliable. "

    try:
        time.sleep(API_REQUEST_DELAY_SECONDS)
        loc = geolocator.geocode(f"{zip_code}, {country_code}", timeout=API_TIMEOUT_SECONDS)
        if loc:
            coords = (loc.latitude, loc.longitude)
            cache[key] = coords
            save_persistent_geocode_cache(cache)
            logging.info(f"Geocoded '{key}' to {coords}. Saved to cache.")
            return json.dumps({"latitude": coords[0], "longitude": coords[1], "status": "success_api"})
        logging.warning(f"Could not geocode zip: {zip_code}, {country_code}")
        return json.dumps({"error": f"{warning_msg}Could not geocode zip code: {zip_code}", "status": "error_not_found"})
    except (GeocoderTimedOut, GeocoderUnavailable) as e:
        logging.error(f"Geocoding service issue for zip {zip_code}: {e}")
        return json.dumps({"error": f"{warning_msg}Geocoding service issue: {e}", "status": "error_service_unavailable"})
    except Exception as e:
        logging.error(f"Error geocoding zip {zip_code}: {e}")
        return json.dumps({"error": f"{warning_msg}Unexpected geocoding error: {e}", "status": "error_unknown"})

available_tools = [find_drugs_tool, find_clinical_trials_tool, zip_to_coordinates_tool]

# --- Langchain Agent Setup ---
@st.cache_resource
def get_langchain_agent_executor():
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a highly specialized AI assistant for exploring information about cancer drugs and clinical trials. Your primary goal is to provide relevant, summarized information to users based on their input.

Key Responsibilities:
1.  **Information Gathering:** Collect necessary details from the user: diagnosis, stage, and any known biomarkers. Be clear if information is missing. If the query is too vague, ask for clarification.
2.  **Tool Utilization:**
    *   Use `find_drugs_tool` to find drug studies.
    *   Use `find_clinical_trials_tool` to find clinical trials. This tool applies initial filters (Overall Survival outcome, specific phases, interventional type).
    *   If trials are discussed and the user expresses interest in location, ask for their zip code (and country, defaulting to US). Then, use `zip_to_coordinates_tool` to get coordinates. Re-invoke `find_clinical_trials_tool` with these coordinates.
3.  **Critical Review & Summarization (VERY IMPORTANT):**
    *   After a tool returns data (JSON string of drugs/trials), YOU MUST carefully review the items for relevance to the user's stated diagnosis, stage, and biomarkers.
    *   Do NOT just dump the raw tool output. Select the most relevant items based on the context.
    *   Explain your reasoning if you choose to highlight certain results or if many results seem weakly relevant.
    *   Summarize key findings from the selected items. DO NOT MENTION "semantic similarity" scores to the user. These are for your internal relevance assessment if provided by the tool.
4.  **Markdown Formatting for User:**
    *   Present information clearly using Markdown (bolding, bullet points, etc.).
    *   For drugs:
        ```markdown
        **Drug: [Drug Name]**
        *   Studied for: [Cancer Type from study]
        *   Key Outcome Data (if available): Treatment OS: [OS], Control OS: [OS], OS Improvement: [X%] (or similar for PFS)
        *   Brief Summary: [Summary from tool, rephrased for clarity]
        ```
    *   For trials:
        ```markdown
        **Trial: [NCT ID] - [Shortened Title]**
        *   Status: [Status], Phase: [Phase]
        *   Conditions: [Conditions from tool]
        *   Interventions: [Interventions from tool]
        *   Distance: [e.g., Approx. X miles away] (Only if location search was performed and distance is available)
        *   Summary: [Brief summary from tool, rephrased for clarity]
        *   Link: [URL from tool] (If available)
        ```
5.  **Disclaimer:** ALWAYS remind the user that you are NOT providing medical advice. All information is for exploration and MUST be discussed with a qualified healthcare provider. This is critical.
6.  **Follow-up Suggestions:** At the end of your substantive responses, suggest 2-3 relevant follow-up questions. Format them EXACTLY like this, with each suggestion on a new line:
    [SUGGESTIONS_START]
    1. What are the main side effects of [specific drug mentioned]?
    2. Can you tell me more about trial [NCT ID mentioned]?
    3. How does [biomarker] affect treatment options for [diagnosis]?
    [SUGGESTIONS_END]
    (Adapt the example suggestions to be relevant to the current conversation context.)
7.  **Tone:** Maintain a friendly, empathetic, professional, and cautious tone.
8.  **Error Handling:** If a tool returns an error or no results, inform the user gracefully. Suggest trying different phrasing or providing more details.
9.  **Scope:** Do not answer questions outside the scope of drug and clinical trial information for cancer.
You have access to tools. Use them when appropriate. Your final answer to the user should be a complete thought.
""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    # Max tokens to sample should be high enough for complex responses + tool use
    llm = ChatAnthropic(model=CLAUDE_MODEL_NAME, temperature=0.2, api_key=ANTHROPIC_API_KEY, max_tokens_to_sample=3500)
    agent = create_tool_calling_agent(llm, available_tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=available_tools, verbose=True, handle_parsing_errors=True, max_iterations=10)
    return agent_executor

# --- Streamlit UI ---
# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history_for_agent" not in st.session_state:
    st.session_state.chat_history_for_agent = []
if "agent_executor" not in st.session_state and DATA_LOADED_SUCCESSFULLY:
    st.session_state.agent_executor = get_langchain_agent_executor()
if "current_input" not in st.session_state:
    st.session_state.current_input = ""
if "clicked_suggestion" not in st.session_state:
    st.session_state.clicked_suggestion = None

# --- Sidebar ---
with st.sidebar:
    st.title("AI Assistant Controls")
    st.divider()
    
    if DATA_LOADED_SUCCESSFULLY:
        if st.button("🔄 Restart Conversation", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.chat_history_for_agent = []
            st.session_state.agent_executor = get_langchain_agent_executor() 
            st.session_state.current_input = ""
            st.session_state.clicked_suggestion = None
            st.success("Conversation restarted!")
            st.rerun()
    else:
        st.warning("Application data failed to load. Restart is unavailable.")

    st.divider()
    st.subheader("About this Assistant")
    st.info(
        "This AI assistant helps you explore information about cancer drugs and clinical trials. "
    )
    # st.warning(
    #     "**Disclaimer:** The information provided is for general knowledge and informational purposes only, "
    #     "and does not constitute medical advice. It is essential to consult with a qualified healthcare "
    #     "professional for any health concerns or before making any decisions related to your health or treatment."
    # )
    st.caption(f"Claude Model: `{CLAUDE_MODEL_NAME}` | Embedding: `{EMBEDDING_MODEL_NAME}`")

# --- Main Chat Interface ---
st.header("AI Drug & Trial Information Assistant")

if DATA_LOADED_SUCCESSFULLY:
    # Display chat messages
    for message in st.session_state.messages:
        avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else USER_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"], unsafe_allow_html=True)

    # Handle clicked suggestion
    if st.session_state.clicked_suggestion:
        st.session_state.current_input = st.session_state.clicked_suggestion
        st.session_state.clicked_suggestion = None # Reset
        # We will process this current_input as if user typed it
    
    # Get user input (either typed or from a clicked suggestion)
    user_typed_prompt = st.chat_input("Ask about drugs or clinical trials for a diagnosis...", key="main_chat_input")

    process_prompt = None
    if st.session_state.current_input: # A suggestion was clicked and set
        process_prompt = st.session_state.current_input
        st.session_state.current_input = "" # Clear after use
    elif user_typed_prompt:
        process_prompt = user_typed_prompt

    if process_prompt:
        st.session_state.messages.append({"role": "user", "content": process_prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(process_prompt)

        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            message_placeholder = st.empty()
            full_response_content = ""
            with st.spinner("🧑‍⚕️ Thinking..."):
                try:
                    agent_input = {
                        "input": process_prompt,
                        "chat_history": st.session_state.chat_history_for_agent
                    }
                    # Ensure agent_executor is initialized
                    if 'agent_executor' not in st.session_state:
                         st.session_state.agent_executor = get_langchain_agent_executor()

                    response_data = st.session_state.agent_executor.invoke(agent_input)
                    ai_response_content = response_data.get("output", "I apologize, I couldn't process that request properly.")
                    full_response_content = ai_response_content
                    
                    st.session_state.chat_history_for_agent.append(HumanMessage(content=process_prompt))
                    st.session_state.chat_history_for_agent.append(AIMessage(content=ai_response_content))
                    
                    max_history_pairs = 7 
                    if len(st.session_state.chat_history_for_agent) > max_history_pairs * 2:
                        st.session_state.chat_history_for_agent = st.session_state.chat_history_for_agent[-(max_history_pairs * 2):]

                except Exception as e:
                    logging.error(f"Error during agent invocation: {e}", exc_info=True)
                    err_msg = f"Sorry, I encountered an error processing your request. Details: {str(e)[:200]}..."
                    if "Input to ChatPromptTemplate is missing variables" in str(e):
                        err_msg += "\nThis might be a prompt configuration issue. Please notify the administrator."
                    full_response_content = err_msg
                    st.session_state.chat_history_for_agent.append(HumanMessage(content=process_prompt))
                    st.session_state.chat_history_for_agent.append(AIMessage(content=f"System error processing request: {e}"))
            
            message_placeholder.markdown(full_response_content, unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response_content})

        # Extract and display suggested prompts using the new tags
        # REGEX to find content between [SUGGESTIONS_START] and [SUGGESTIONS_END]
        suggestion_block_match = re.search(r"\[SUGGESTIONS_START\](.*?)\[SUGGESTIONS_END\]", full_response_content, re.DOTALL | re.IGNORECASE)
        
        parsed_suggestions = []
        if suggestion_block_match:
            suggestions_text = suggestion_block_match.group(1).strip()
            # Split by newline and filter out empty lines and numbered prefixes
            raw_lines = suggestions_text.split('\n')
            for line in raw_lines:
                cleaned_line = re.sub(r"^\s*\d+\.\s*", "", line).strip() # Remove "1. ", "2. ", etc.
                if cleaned_line:
                    parsed_suggestions.append(cleaned_line)
        
        if parsed_suggestions:
            st.subheader("Suggested Follow-up Questions:")
            # Display max 3 suggestions even if more are parsed
            cols = st.columns(min(len(parsed_suggestions), 3)) 
            for i, suggestion_text in enumerate(parsed_suggestions[:3]):
                # Use a more robust unique key including a timestamp component
                button_key = f"suggestion_{i}_{int(time.time())}_{hash(suggestion_text)}"
                if cols[i].button(suggestion_text, key=button_key, use_container_width=True):
                    st.session_state.clicked_suggestion = suggestion_text
                    st.rerun() # Rerun to process the clicked suggestion

        # Only rerun if a prompt was actually processed to update the display fully.
        # This includes user typed or clicked suggestion.
        st.rerun()


    # Initial greeting
    if not st.session_state.messages and DATA_LOADED_SUCCESSFULLY:
        initial_greeting = (
            "Hello! I'm your AI assistant for exploring information about cancer drugs and clinical trials. "
            "How can I assist you today? For example, you can tell me about a diagnosis like 'Stage IV Lung Cancer with EGFR mutation'."
        )
        st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
        st.session_state.chat_history_for_agent.append(AIMessage(content=initial_greeting))
        st.rerun()

elif not DATA_LOADED_SUCCESSFULLY:
    st.warning("The application could not start correctly. Please review error messages and check your setup.")

st.markdown("---")
st.caption("This is an AI-powered informational tool. Always consult a healthcare professional for medical advice.")