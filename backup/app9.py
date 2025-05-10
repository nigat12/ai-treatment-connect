# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------
# AI Cancer Information Assistant (Production Candidate - Robust Suggestions)
# v2.0 - Enhanced Conversational AI & UI Rendering
# --------------------------------------------------------------------------

# 1. SET PAGE CONFIG FIRST
import streamlit as st
st.set_page_config(
    page_title="AI Cancer Info Assistant",
    page_icon="🧑‍⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:help@example.com', # Replace
        'Report a bug': "mailto:bugs@example.com", # Replace
        'About': """
        ## AI Cancer Information Assistant
        This tool helps explore information about cancer drugs and clinical trials.
        **Important Disclaimer:** This tool is for informational purposes ONLY and is NOT medical advice.
        Always consult with your doctor or a qualified healthcare professional.
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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.pydantic_v1 import BaseModel, Field

# --- Application Configuration & Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - Line %(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
load_dotenv()

# --- API Key Check ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    st.error("🚨 **CRITICAL ERROR: ANTHROPIC_API_KEY is not configured!**")
    logger.critical("ANTHROPIC_API_KEY not found. Application cannot proceed.")
    st.stop()

# --- File Paths & Constants (Keep as is from previous version) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRUG_DATA_CSV = os.path.join(BASE_DIR, 'drug_data.csv')
TRIAL_DATA_XLSX = os.path.join(BASE_DIR, 'trials_filtered_with_coordinates.xlsx')
DRUG_EMBEDDINGS_FILE = os.path.join(BASE_DIR, 'drug_embeddings.npy')
TRIAL_EMBEDDINGS_FILE = os.path.join(BASE_DIR, 'trial_embeddings.npy')
TRIAL_INDEX_MAP_FILE = os.path.join(BASE_DIR, 'trial_index_map.pkl')
GEOCODE_CACHE_FILE_PATH = os.path.join(BASE_DIR, "geocode_cache.json")
TEMP_GEOCODE_CACHE_FILE_PATH = os.path.join(BASE_DIR, "geocode_cache.tmp.json")

EMBEDDING_MODEL_NAME = 'neuml/pubmedbert-base-embeddings'
DRUG_TEXT_COLUMNS_FOR_EMBEDDING = ['Cancer Type', 'Drug Name'] # Added more context
TRIAL_TEXT_COLUMNS_FOR_EMBEDDING = ['Study Title', 'Conditions'] # Added more context

TRIAL_FILTER_PRIMARY_OUTCOME_COLUMN = 'Primary Outcome Measures'
TRIAL_FILTER_PRIMARY_OUTCOME_TERM = 'Overall Survival' # Can be broadened later if needed
TRIAL_FILTER_PHASES_COLUMN = 'Phases'
TRIAL_ACCEPTABLE_PHASES_STR = ['PHASE1|PHASE2', 'PHASE2', 'PHASE2|PHASE3', 'PHASE3', 'PHASE4'] # Adjusted for consistency
TRIAL_ACCEPTABLE_INDIVIDUAL_PHASES = set()
for phase_combo in TRIAL_ACCEPTABLE_PHASES_STR:
    for phase_str_part in re.split(r'[|/,\s]+', phase_combo):
        if phase_str_part: TRIAL_ACCEPTABLE_INDIVIDUAL_PHASES.add(phase_str_part.strip().upper())
TRIAL_FILTER_STUDY_TYPE_COLUMN = 'Study Type'
TRIAL_FILTER_STUDY_TYPE_VALUE = 'INTERVENTIONAL'

CLAUDE_MODEL_NAME = "claude-3-5-haiku-20241022" # Using Opus for better reasoning if available & budget allows, else Haiku is fine.
DEFAULT_SEARCH_RADIUS_MILES = 50
NOMINATIM_USER_AGENT = "AI_Cancer_Info_Assistant_Prod/1.0" # !! PLEASE UPDATE !!
API_REQUEST_DELAY_SECONDS = 1.05
API_TIMEOUT_SECONDS = 15

ASSISTANT_AVATAR = "🧑‍⚕️"
USER_AVATAR = "👤"

# --- Helper Functions (Keep as is) ---
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

def check_phases(trial_phases_raw_str):
    if not isinstance(trial_phases_raw_str, str) or not str(trial_phases_raw_str).strip(): return False
    trial_individual_phases = re.split(r'[|/,\s]+', str(trial_phases_raw_str).strip())
    return any(phase_part.strip().upper() in TRIAL_ACCEPTABLE_INDIVIDUAL_PHASES for phase_part in trial_individual_phases if phase_part)

# --- Data Loading Functions (Minor adjustments for embedding columns) ---
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
        if not all(col in df.columns for col in DRUG_TEXT_COLUMNS_FOR_EMBEDDING):
            missing_cols = [col for col in DRUG_TEXT_COLUMNS_FOR_EMBEDDING if col not in df.columns]
            raise ValueError(f"Missing required columns for drug embedding in {csv_path}. Expected: {DRUG_TEXT_COLUMNS_FOR_EMBEDDING}, Missing: {missing_cols}")
        df['combined_text_for_embedding'] = df[DRUG_TEXT_COLUMNS_FOR_EMBEDDING].fillna('').astype(str).agg(' '.join, axis=1)
        df['Treatment_OS_Months_Parsed'] = df.get('Treatment_OS', pd.Series([None]*len(df))).apply(parse_time_to_months)
        df['Control_OS_Months_Parsed'] = df.get('Control_OS', pd.Series([None]*len(df))).apply(parse_time_to_months)
        df['Treatment_PFS_Months_Parsed'] = df.get('Treatment_PFS', pd.Series([None]*len(df))).apply(parse_time_to_months) # Added PFS parsing
        df['Control_PFS_Months_Parsed'] = df.get('Control_PFS', pd.Series([None]*len(df))).apply(parse_time_to_months) # Added PFS parsing
        df = df.fillna('N/A')
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
            missing_cols = [col for col in TRIAL_TEXT_COLUMNS_FOR_EMBEDDING if col not in df.columns]
            raise ValueError(f"Missing required columns for trial embedding in {xlsx_path}. Expected: {TRIAL_TEXT_COLUMNS_FOR_EMBEDDING}, Missing: {missing_cols}")
        df['combined_text_for_embedding'] = df[TRIAL_TEXT_COLUMNS_FOR_EMBEDDING].fillna('').astype(str).agg(' '.join, axis=1)
        if 'location_coordinates' in df.columns:
            def parse_excel_coords_robust(coord_str_val):
                if pd.isna(coord_str_val) or not isinstance(coord_str_val, str) or not coord_str_val.strip(): return []
                try:
                    try: parsed_list = ast.literal_eval(coord_str_val)
                    except (SyntaxError, ValueError):
                        coord_str_val_fixed = coord_str_val.replace("'", "\"")
                        parsed_list = json.loads(coord_str_val_fixed)
                    if isinstance(parsed_list, list):
                        return [tuple(item) for item in parsed_list if isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(num, (int, float)) and not math.isnan(num) for num in item)]
                    return []
                except Exception as parse_err:
                    logger.warning(f"Could not parse coordinate string: '{str(coord_str_val)[:50]}...'. Error: {parse_err}")
                    return []
            df['parsed_location_coordinates'] = df['location_coordinates'].apply(parse_excel_coords_robust)
        else:
            logger.warning("'location_coordinates' column not found in trial data. Location-based search will be impaired.")
            df['parsed_location_coordinates'] = pd.Series([[] for _ in range(len(df))])
        df = df.fillna('N/A')
        logger.info(f"Trial data loaded and preprocessed from '{xlsx_path}'. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Fatal error loading or preprocessing trial data from '{xlsx_path}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to load clinical trial information database: {e}")

@st.cache_data(show_spinner="Preparing trial information for search...")
def get_or_generate_trial_embeddings(_trial_df, _model, embeddings_path=TRIAL_EMBEDDINGS_FILE, map_path=TRIAL_INDEX_MAP_FILE):
    if os.path.exists(embeddings_path) and os.path.exists(map_path):
        try:
            embeddings = np.load(embeddings_path)
            with open(map_path, 'rb') as f: index_map = pickle.load(f)
            if (embeddings is not None and index_map is not None and
                embeddings.shape[0] == len(index_map) and
                embeddings.shape[1] == _model.get_sentence_embedding_dimension() and
                (max(index_map.keys(), default=-1) < len(_trial_df) if index_map else True)):
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
embedding_model_global, df_drugs_processed_global, drug_embeddings_array_global = None, None, None
df_trials_processed_global, trial_embeddings_array_global, trial_index_map_global = None, None, None

try:
    if NOMINATIM_USER_AGENT == "AI_Cancer_Info_Assistant_Prod/1.5 your_contact_email@example.com": # Default check
        st.sidebar.warning(
            "**Geocoding Service Alert:** Update `NOMINATIM_USER_AGENT` in the script "
            "with your unique app name & email for reliable geocoding."
        )
    embedding_model_global = load_sentence_transformer_model()
    df_drugs_processed_global = load_and_preprocess_drug_data(DRUG_DATA_CSV)
    drug_embeddings_array_global = get_or_generate_drug_embeddings(df_drugs_processed_global, embedding_model_global)
    df_trials_processed_global = load_and_preprocess_trial_data(TRIAL_DATA_XLSX)
    trial_embeddings_array_global, trial_index_map_global = get_or_generate_trial_embeddings(df_trials_processed_global, embedding_model_global)
    DATA_LOADED_SUCCESSFULLY = True
    logger.info("All critical data and AI models initialized successfully.")
except RuntimeError as e:
    st.error(f"🚨 **APPLICATION STARTUP FAILED:** `{e}`\n\nPlease check logs. The app cannot continue.")
    logger.critical(f"Startup failed: {e}", exc_info=True)

# --- Dynamic Top-N & Geocoding Cache (Keep as is) ---
def _determine_top_n_results(scored_results_list):
    if not scored_results_list: return 0
    count = len(scored_results_list)
    if count == 0: return 0
    if count <= 5: return count # Show fewer by default, emphasize quality
    score_at_3rd = scored_results_list[2]['semantic_similarity'] if count > 2 else 0
    if score_at_3rd >= 0.75: return min(count, 5)
    if score_at_3rd < 0.6: return 3
    return min(count, 3) # Default to top 3 unless very good matches

def load_persistent_geocode_cache(filepath=GEOCODE_CACHE_FILE_PATH):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f: return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Geocode cache file '{filepath}' is corrupted. Starting empty.")
            return {}
        except Exception as e:
            logger.warning(f"Error loading geocode cache '{filepath}': {e}. Starting empty.")
            return {}
    return {}

def save_persistent_geocode_cache(cache_data, target_path=GEOCODE_CACHE_FILE_PATH, temp_path=TEMP_GEOCODE_CACHE_FILE_PATH):
    try:
        with open(temp_path, 'w') as f: json.dump(cache_data, f, indent=2)
        if os.path.exists(target_path): os.remove(target_path)
        os.rename(temp_path, target_path)
    except Exception as e:
        logger.warning(f"Could not save geocode cache to '{target_path}': {e}", exc_info=True)


# --- Langchain Tools ---
class FindDrugsInput(BaseModel):
    diagnosis: str = Field(description="The primary diagnosis, e.g., 'Metastatic Breast Cancer', 'Lung Adenocarcinoma'")
    stage: str = Field(description="The cancer stage, e.g., 'Stage IV', 'Recurrent', 'Advanced', 'Early Stage'")
    biomarkers: str = Field(description="Known biomarkers or genetic mutations, comma-separated or 'None', e.g., 'HER2-positive, ER-negative', 'EGFR exon 19 deletion', 'KRAS G12C'")
    line_of_therapy: str = Field(None, description="Optional. The line of therapy if known, e.g., 'first-line', 'second-line', 'previously treated with chemotherapy'")
@tool("find_drugs_tool", args_schema=FindDrugsInput)
def find_drugs_tool(diagnosis: str, stage: str, biomarkers: str, line_of_therapy: str = None) -> str:
    """
    Finds relevant drug information based on cancer diagnosis, stage, biomarkers, and optionally, line of therapy.
    Returns a JSON string containing a list of potential drugs with details like name, cancer type studied,
    survival data (OS/PFS if available), and a brief summary.
    The AI assistant MUST review this data for relevance to the user's specific query, select only the most pertinent
    information, summarize it in a patient-friendly way using the specified Markdown format, and MUST NOT show
    'semantic_similarity' scores or raw data directly to the user.
    """
    logger.info(f"Executing find_drugs_tool: Dx='{diagnosis}', Stg='{stage}', Bio='{biomarkers}', Line='{line_of_therapy}'")
    if drug_embeddings_array_global is None or drug_embeddings_array_global.size == 0:
        logger.error("Drug embeddings not available for find_drugs_tool.")
        return json.dumps({"error": "Drug information database is currently unavailable.", "drugs": []})

    query_parts = [diagnosis, stage, biomarkers, line_of_therapy]
    query_text = " ".join(filter(None, query_parts)).strip()

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
    for idx, row_series in df_drugs_processed_global.iterrows():
        if idx >= len(similarities): continue
        similarity_score = similarities[idx]
        if similarity_score >= 0.5: # Adjusted threshold for potentially better initial filtering
            potential_results.append({
                'drug_name': row_series.get('Drug Name', 'N/A'),
                'cancer_type_studied': row_series.get('Cancer Type', 'N/A'), # Cancer type in the study
                'biomarker_info_from_study': row_series.get('Biomarker', 'N/A'), # Biomarker info from the study
                'treatment_os_months': row_series.get('Treatment_OS_Months_Parsed', None),
                'control_os_months': row_series.get('Control_OS_Months_Parsed', None),
                'treatment_pfs_months': row_series.get('Treatment_PFS_Months_Parsed', None),
                'control_pfs_months': row_series.get('Control_PFS_Months_Parsed', None),
                'brief_study_summary_text': textwrap.shorten(str(row_series.get('Brief Study Summary', '')), 300, placeholder="..."),
                'source_of_data_id': row_series.get('Source ID', 'N/A'), # e.g., study ID or reference
                'semantic_similarity': round(float(similarity_score), 3),
            })
    potential_results.sort(key=lambda x: x['semantic_similarity'], reverse=True)
    num_to_select_count = _determine_top_n_results(potential_results)
    final_selected_results = potential_results[:num_to_select_count]

    if not final_selected_results:
        return json.dumps({"message": "No drug studies closely matched your specific criteria at this time. This could be due to the specificity of your request or data availability.", "drugs": []})
    return json.dumps({"drugs": final_selected_results, "count": len(final_selected_results)})

class FindClinicalTrialsInput(BaseModel):
    diagnosis: str = Field(description="Primary diagnosis, e.g., 'Non-Small Cell Lung Cancer'")
    stage: str = Field(description="Cancer stage, e.g., 'Stage IV', 'Metastatic'")
    biomarkers: str = Field(description="Biomarkers or genetic mutations, comma-separated or 'None', e.g., 'ALK-positive', 'EGFR exon 20 insertion'")
    line_of_therapy: str = Field(None, description="Optional. The line of therapy if known, e.g., 'first-line', 'treatment-naive', 'progressed on Osimertinib'")
    user_latitude: float = Field(None, description="User's latitude. Optional for location-based search.")
    user_longitude: float = Field(None, description="User's longitude. Optional for location-based search.")
    radius_miles: int = Field(DEFAULT_SEARCH_RADIUS_MILES, description="Search radius in miles if location is provided.")
@tool("find_clinical_trials_tool", args_schema=FindClinicalTrialsInput)
def find_clinical_trials_tool(diagnosis: str, stage: str, biomarkers: str, line_of_therapy: str = None, user_latitude: float = None, user_longitude: float = None, radius_miles: int = DEFAULT_SEARCH_RADIUS_MILES) -> str:
    """
    Searches for clinical trials based on diagnosis, stage, biomarkers, and optionally line of therapy and location.
    Returns a JSON string of trials with details like NCT ID, title, status, phase, conditions, interventions,
    a brief summary, and an official URL. Distance is included if location search was performed.
    The AI assistant MUST review this data for relevance to the user's specific query, select only the most pertinent
    information, summarize it in a patient-friendly way using the specified Markdown format, and MUST NOT show
    'semantic_similarity' scores or raw data directly to the user. It should also highlight if a trial specifically
    mentions the user's biomarker or if it's more general.
    """
    logger.info(f"Executing find_clinical_trials_tool: Dx='{diagnosis}', Stg='{stage}', Bio='{biomarkers}', Line='{line_of_therapy}', Loc=({user_latitude},{user_longitude}), Rad={radius_miles}")
    if trial_embeddings_array_global is None or trial_embeddings_array_global.size == 0 or trial_index_map_global is None:
        logger.error("Trial embeddings or index map not available for find_clinical_trials_tool.")
        return json.dumps({"error": "Clinical trial information database is currently unavailable.", "trials": []})

    query_parts = [diagnosis, stage, biomarkers, line_of_therapy]
    query_text = " ".join(filter(None, query_parts)).strip()

    if not query_text:
        logger.warning("Empty query received for find_clinical_trials_tool.")
        return json.dumps({"error": "Please provide a diagnosis to search for clinical trials.", "trials": []})
    user_coordinates = (user_latitude, user_longitude) if user_latitude is not None and user_longitude is not None else None
    try:
        query_embedding = embedding_model_global.encode(query_text, convert_to_numpy=True)
    except Exception as e:
        logger.error(f"Error embedding trial query: {e}", exc_info=True)
        return json.dumps({"error": "An internal error occurred while preparing your trial search.", "trials": []})

    potential_results = []
    for original_idx, row_data in df_trials_processed_global.iterrows():
        # Pre-filtering based on structured data (can be expanded)
        # primary_outcome_str = str(row_data.get(TRIAL_FILTER_PRIMARY_OUTCOME_COLUMN, '')).lower()
        trial_phases_str = str(row_data.get(TRIAL_FILTER_PHASES_COLUMN, ''))
        study_type_str = str(row_data.get(TRIAL_FILTER_STUDY_TYPE_COLUMN, '')).upper()

        # Basic mandatory filters
        if not ( # TRIAL_FILTER_PRIMARY_OUTCOME_TERM.lower() in primary_outcome_str and # Temporarily relax outcome term filter for broader results
                check_phases(trial_phases_str) and \
                study_type_str == TRIAL_FILTER_STUDY_TYPE_VALUE.upper() and \
                original_idx in trial_index_map_global and \
                trial_index_map_global[original_idx] < len(trial_embeddings_array_global)):
            continue

        embedding_idx = trial_index_map_global[original_idx]
        similarity_score = cosine_similarity([query_embedding], [trial_embeddings_array_global[embedding_idx]])[0][0]

        if similarity_score < 0.5: continue # Adjusted threshold

        trial_details = {
            'nct_id': row_data.get('NCT Number', 'N/A'),
            'official_study_title': str(row_data.get('Study Title', 'N/A')), # Full title for AI context
            'status': row_data.get('Study Status', 'N/A'),
            'phases': trial_phases_str,
            'conditions_studied_text': str(row_data.get('Conditions', 'N/A')), # Full conditions for AI context
            'interventions_studied_text': str(row_data.get('Interventions', 'N/A')), # Full interventions for AI context
            'brief_summary_text': textwrap.shorten(str(row_data.get('Brief Summary', '')), 350, placeholder="..."),
            'official_url': f"https://clinicaltrials.gov/study/{row_data.get('NCT Number', '')}" if row_data.get('NCT Number', 'N/A') != 'N/A' else None,
            'semantic_similarity': round(float(similarity_score), 3),
            'distance_miles': None,
            'trial_locations_text': str(row_data.get('Locations', 'N/A')) # For AI to parse if needed
        }

        if user_coordinates:
            trial_site_coordinates_list = row_data.get('parsed_location_coordinates', [])
            min_distance_miles = float('inf')
            if trial_site_coordinates_list:
                for site_coords_tuple in trial_site_coordinates_list:
                    try:
                        current_distance = geodesic(user_coordinates, site_coords_tuple).miles
                        if current_distance < min_distance_miles:
                            min_distance_miles = current_distance
                    except Exception as dist_err:
                        logger.debug(f"Could not calculate distance for site {site_coords_tuple} in trial {trial_details['nct_id']}: {dist_err}")

            if min_distance_miles <= radius_miles:
                trial_details['distance_miles'] = round(min_distance_miles, 1)
                potential_results.append(trial_details)
            # If no trials within radius, still consider adding the closest ones if list is empty.
            # This part needs careful balancing or a flag to the AI. For now, strict radius.
        else:
            potential_results.append(trial_details)

    if user_coordinates:
        potential_results.sort(key=lambda x_item: (sort_key_with_none(x_item['distance_miles'], False), -x_item['semantic_similarity']))
    else:
        potential_results.sort(key=lambda x_item: x_item['semantic_similarity'], reverse=True)

    num_to_select_count = _determine_top_n_results(potential_results)
    final_selected_results = potential_results[:num_to_select_count]

    if not final_selected_results:
        return json.dumps({"message": "No clinical trials closely matched your specific criteria or location at this time. This could be due to the specificity of your request or data availability.", "trials": []})
    return json.dumps({"trials": final_selected_results, "count": len(final_selected_results)})


class ZipToCoordinatesInput(BaseModel):
    zip_code: str = Field(description="User's zip code, e.g., '90210'")
    country_code: str = Field("US", description="Country code for the zip code, e.g., 'US', 'CA'. Defaults to 'US'.")
@tool("zip_to_coordinates_tool", args_schema=ZipToCoordinatesInput)
def zip_to_coordinates_tool(zip_code: str, country_code: str = "US") -> str:
    """
    Converts a zip code and country code to geographic coordinates (latitude and longitude).
    Returns a JSON string with 'latitude', 'longitude', and 'status'.
    The AI assistant uses these coordinates for location-specific searches, primarily for clinical trials.
    """
    logger.info(f"Executing zip_to_coordinates_tool: Zip='{zip_code}', Country='{country_code}'")
    geocode_cache = load_persistent_geocode_cache()
    cache_lookup_key = f"{zip_code}_{country_code}".lower()
    if cache_lookup_key in geocode_cache and geocode_cache[cache_lookup_key]:
        logger.info(f"Zip cache hit for '{cache_lookup_key}'. Coords: {geocode_cache[cache_lookup_key]}")
        cached_coords = geocode_cache[cache_lookup_key]
        return json.dumps({"latitude": cached_coords[0], "longitude": cached_coords[1], "status": "success_cache"})

    geolocator_service = Nominatim(user_agent=NOMINATIM_USER_AGENT)
    nominatim_warning_message = ""
    if NOMINATIM_USER_AGENT == "AI_Cancer_Info_Assistant_Prod/1.5 your_contact_email@example.com": # Default check
         nominatim_warning_message = "Warning: Nominatim user agent is default. Geocoding may be less reliable. "
    try:
        time.sleep(API_REQUEST_DELAY_SECONDS) # Respect API rate limits
        location_data = geolocator_service.geocode(f"{zip_code}, {country_code}", timeout=API_TIMEOUT_SECONDS)
        if location_data:
            coordinates = (location_data.latitude, location_data.longitude)
            geocode_cache[cache_lookup_key] = coordinates
            save_persistent_geocode_cache(geocode_cache)
            logger.info(f"Geocoded '{cache_lookup_key}' to {coordinates}. Saved to cache.")
            return json.dumps({"latitude": coordinates[0], "longitude": coordinates[1], "status": "success_api"})
        logger.warning(f"Could not geocode zip: {zip_code}, country: {country_code} using Nominatim.")
        return json.dumps({"error": f"{nominatim_warning_message}Could not find location for zip code: {zip_code}. Please check the zip code and country.", "status": "error_not_found"})
    except (GeocoderTimedOut, GeocoderUnavailable) as ge_service_error:
        logger.error(f"Nominatim service issue for zip {zip_code}: {ge_service_error}", exc_info=True)
        return json.dumps({"error": f"{nominatim_warning_message}The location service is temporarily unavailable or timed out. Please try again later.", "status": "error_service_unavailable"})
    except Exception as general_ge_error:
        logger.error(f"Unexpected error geocoding zip {zip_code}: {general_ge_error}", exc_info=True)
        return json.dumps({"error": f"{nominatim_warning_message}An unexpected error occurred while trying to find the location for zip code: {zip_code}.", "status": "error_unknown"})

available_tools = [find_drugs_tool, find_clinical_trials_tool, zip_to_coordinates_tool]

# --- Function for Separate Suggestion Generation ---
def generate_follow_up_suggestions(user_query: str, ai_main_response: str, llm_client: ChatAnthropic) -> list[str]:
    """
    Generates 2-3 relevant follow-up suggestions based on the last interaction.
    """
    user_query = str(user_query) if user_query is not None else "The user asked a question."
    ai_main_response = str(ai_main_response) if ai_main_response is not None else "I provided some information."

    max_len = 700
    truncated_user_query = textwrap.shorten(user_query, width=max_len, placeholder="...")
    truncated_ai_response = textwrap.shorten(ai_main_response, width=max_len, placeholder="...")

    suggestion_prompt_messages = [
        SystemMessage(content="You are an expert at crafting helpful follow-up questions for a user exploring cancer information. Based on the user's last query and the AI assistant's response, provide exactly 2 or 3 concise, patient-friendly follow-up questions. These questions should anticipate what the user might naturally want to know next, such as asking for more details on a specific drug/trial, side effects, alternative options, understanding a term, or how to discuss the information with a doctor. Output ONLY the suggestions as a JSON list of strings. Example: {\"suggestions\": [\"What are common side effects of [Drug Name]?\", \"Tell me more about the eligibility for [Trial ID].\", \"Are there other types of treatments for this situation?\"]}"),
        HumanMessage(content=f"User's last query: \"{truncated_user_query}\"\n\nAI Assistant's last response: \"{truncated_ai_response}\"\n\nGenerate 2-3 distinct, relevant follow-up questions based on this exchange. Ensure questions are actionable for the user. Format as a JSON list of strings under a 'suggestions' key.")
    ]
    try:
        response = llm_client.invoke(suggestion_prompt_messages)
        content = response.content
        logger.debug(f"Raw suggestions LLM response: {content}")

        json_match = re.search(r"{\s*\"suggestions\"\s*:\s*\[.*?\]\s*}", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                suggestions_data = json.loads(json_str)
                if isinstance(suggestions_data.get("suggestions"), list):
                    return [str(s).strip() for s in suggestions_data["suggestions"] if str(s).strip()][:3]
            except json.JSONDecodeError:
                logger.warning(f"Could not decode JSON for suggestions: {json_str}")
        else:
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            parsed_lines = []
            for line in lines:
                cleaned_line = re.sub(r"^\s*[\d\.\-\*]+\s*", "", line).strip().replace("\"", "")
                if cleaned_line and len(cleaned_line) > 10 and cleaned_line.endswith("?"): # Ensure it's a question
                    parsed_lines.append(cleaned_line)
            if parsed_lines:
                logger.info(f"Parsed suggestions using line splitting fallback: {parsed_lines[:3]}")
                return parsed_lines[:3]
    except Exception as e:
        logger.error(f"Error generating follow-up suggestions: {e}", exc_info=True)
    return []


# --- Langchain Agent Setup ---
@st.cache_resource(show_spinner="Initializing AI Assistant...")
def get_langchain_agent_executor_and_llm():
    # System Prompt: Heavily revised for patient interaction, formatting, and NO embedded suggestions.
    system_prompt_content = """You are an AI Patient Information Assistant specializing in cancer drugs and clinical trials.
Your primary goal is to have a helpful, empathetic, and clear conversation with users (patients or their loved ones) seeking information. You must adhere to the following principles and output formats STRICTLY.

**Core Principles:**
1.  **Empathy and Patient-Centric Language:**
    *   Always begin by acknowledging the user's query, especially if it's about a diagnosis. E.g., "I understand you're looking for information about [diagnosis]. I'll do my best to help you explore this."
    *   Use simple, clear language. AVOID MEDICAL JARGON. If a medical term is unavoidable (e.g., 'metastatic', 'Progression-Free Survival'), explain it briefly in plain terms (e.g., "metastatic, which means the cancer has spread to other parts of the body.", "Progression-Free Survival, or PFS, is the length of time patients lived without their cancer getting worse.").
    *   Maintain a supportive, calm, and professional tone. You are an information assistant, not a doctor.

2.  **Information Gathering & Clarification (Be Conversational):**
    *   If a user's query is too general (e.g., "cancer drugs"), gently ask for more specifics: "To help me find the most relevant information, could you please tell me the specific type of cancer you're interested in? Information like the cancer stage or any known biomarkers (like specific gene mutations) can also be very helpful."
    *   **EGFR Mutation Specificity:** If the user mentions "EGFR mutation" for lung cancer, ask for clarification: "EGFR mutations can be of different types, for example, common ones like exon 19 deletions or L858R, or rarer ones like exon 20 insertions. The best treatment can vary depending on the type. Do you have information on the specific type of EGFR mutation? If not, I can provide information that's generally applicable or for the more common types." State your assumption if you proceed without this detail.
    *   **Line of Therapy:** If not specified for a treatment query, you might ask: "Are you interested in initial (first-line) treatments, or options that might be considered if a previous treatment has been tried?"
    *   Before using tools for a complex query, confirm understanding: "Okay, just to confirm, you're looking for [paraphrased query including diagnosis, stage, biomarkers, line of therapy if provided]. Is that right?"

3.  **Using Your Tools (`find_drugs_tool`, `find_clinical_trials_tool`, `zip_to_coordinates_tool`):**
    *   These tools return JSON data. You MUST process this JSON data.
    *   If discussing trials and location is relevant and not yet provided, ask: "I can also look for trials near a specific area. Would you like me to do that? If so, please provide a zip code (e.g., 90210)."
    *   If they provide a zip code, use `zip_to_coordinates_tool` first, then use the latitude/longitude with `find_clinical_trials_tool`.

4.  **Presenting Information (Your Most Important Task - Follow Formats EXACTLY):**
    *   **NEVER directly output the raw JSON from tools.**
    *   **Critical Review & Selection:** From the tool's JSON output, select only the 1-2 MOST relevant items for the user's specific query. If there are other potentially relevant items, you can mention: "There are a few other options that came up in the search. We can explore those if you'd like." If no results are highly relevant, state that clearly and gently.
    *   **DO NOT MENTION "semantic similarity" or any internal scores to the user.**
    *   **Use Markdown for ALL formatted output to the user.**

    *   **Drug Information Presentation Format (Strict Adherence):**
        ```markdown
        Based on your information about [User's Diagnosis, Stage, Biomarkers], here's some information about a drug that might be discussed for such conditions:

        **Drug Option: [Drug Name]**
        *   **What it is (Simplified):** [e.g., "This is a type of medication called a 'targeted therapy'. Targeted therapies are designed to attack specific features of cancer cells, often related to genetic mutations like the [biomarker if relevant, e.g., EGFR mutation] you mentioned."] (Use `brief_study_summary_text` from tool to help formulate this and the 'What Studies Suggest' part).
        *   **Commonly Used For:** [e.g., "Non-small cell lung cancer (NSCLC) that has a specific type of EGFR mutation, such as [biomarker_info_from_study if specific, or 'common EGFR mutations'] and has spread (metastasized)."] (Refer to `cancer_type_studied` and `biomarker_info_from_study` from tool output).
        *   **What Studies Suggest (Simplified):** [Paraphrase `brief_study_summary_text`. Include key outcomes like PFS or OS if available and clearly explained. E.g., "In one study involving patients with [cancer_type_studied from tool] with [biomarker_info_from_study], [Drug Name] was looked at. For patients taking this drug, the median Progression-Free Survival (PFS), which is the average time they lived without their cancer getting worse, was reported as [treatment_pfs_months] months. This was compared to [control_pfs_months] months for patients receiving [describe control group if simple, e.g., standard chemotherapy]."] (Use `treatment_pfs_months`, `control_pfs_months`, `treatment_os_months`, `control_os_months` from tool output if available and the values are not null/None. Always explain PFS/OS.)
        *   **Line of Therapy Context (if applicable):** [e.g., "This drug is often considered as an initial (first-line) treatment." or "It might be used after other treatments have been tried."]
        *   **Important to Know:** "All medications have potential benefits and risks. Whether this or any drug is appropriate depends on many individual factors, including the specific type of cancer, overall health, and other medical conditions."

        (If presenting a second drug, use a separator like "---" and repeat the format)
        ```

    *   **Clinical Trial Information Presentation Format (Strict Adherence):**
        ```markdown
        Regarding clinical trials for [User's Diagnosis, Stage, Biomarkers], here's one that appears potentially relevant. Please note that eligibility for trials is very specific.

        **Clinical Trial Focus: [Create a patient-friendly summary of the trial's main goal based on `official_study_title` and `brief_summary_text` from tool output, e.g., "Testing a new drug combination for advanced lung cancer with EGFR mutations"]**
        *   **Official Title Snippet:** "[Provide a concise, understandable snippet from `official_study_title`]"
        *   **Trial ID:** [nct_id]
        *   **Purpose of this Study (Simplified):** [Paraphrase `brief_summary_text` from tool output. E.g., "Researchers are investigating if adding a new drug, [Intervention Drug Name if clear], to the standard treatment [Standard Drug Name if clear] is more effective for patients with [conditions_studied_text, simplified]."] (Use `interventions_studied_text` and `conditions_studied_text` for context).
        *   **Current Status:** [status - e.g., "Recruiting participants", "Active, but not currently recruiting"]
        *   **Trial Phase:** [phases - e.g., "Phase 3". Briefly explain if not common knowledge: e.g., "(Phase 3 trials compare new treatments to standard ones in larger groups of people to confirm effectiveness and monitor side effects)"]
        *   **Key Things Being Studied:** [Summarize `interventions_studied_text` simply, e.g., "The main drug being tested is [Drug X], possibly in combination with [Drug Y]."]
        *   **Biomarker Relevance (If applicable):** [e.g., "This trial is specifically for patients with the [user's biomarker] mutation." OR "This trial is for [general condition], and may include patients with [user's biomarker]. You'd need to check the detailed criteria."]
        *   **Approx. Distance (if applicable):** [e.g., "One or more study sites are located about [distance_miles] miles from the zip code you provided."] (ONLY if location search was done and distance is known. Otherwise, omit this line.)
        *   **More Information:** "You can find more detailed official information about this trial, including full eligibility criteria and study locations, at its ClinicalTrials.gov page: [official_url]"

        (If presenting a second trial, use a separator like "---" and repeat the format)
        ```

5.  **MANDATORY DISCLAIMER (Verbatim at the end of relevant responses):**
    *   After providing ANY drug or trial information, ALWAYS conclude with:
        "**Important Reminder:** This information is for educational purposes only and is NOT medical advice. It's essential to discuss all of this with your doctor or a qualified healthcare professional. They are the only ones who can understand your complete medical situation and provide guidance on your care."

6.  **Concluding Your Turn (Interaction Flow):**
    *   After the disclaimer, end your response naturally by inviting further questions. DO NOT generate follow-up suggestions yourself; a separate system will display those.
    *   Examples: "Is there anything specific about [drug/trial name] you'd like me to elaborate on?" or "What other aspects of [user's topic] are you interested in exploring?" or "Do you have any initial questions about what I've shared?"

7.  **Error/No Results Handling:**
    *   If tools return errors or no relevant data after filtering: "I've searched based on the information you provided ([mention key criteria like diagnosis, specific biomarker if any]), but I wasn't able to find specific [drug/trial] information that closely matches right now. This can sometimes happen if the criteria are very specific, or if data isn't available in the resources I can access. Would you like me to try a broader search (e.g., for more general [cancer type] without focusing on [specific biomarker]), or perhaps we can refine the search terms?"

8.  **Scope:** Stick to cancer drug and clinical trial information. For other topics (e.g., diet, alternative therapies not in clinical trials, detailed medical prognosis), politely state it's outside your informational scope and recommend discussion with their healthcare team. E.g., "That's an important question, but detailed advice on [topic] is best discussed with your doctor or a specialist, as it's outside the scope of the drug and trial information I can provide."

Remember, clarity, empathy, and accuracy (within the bounds of an informational assistant) are paramount.
You have access to tools to fetch data. Use them thoughtfully. Your final output to the user should always be a complete, well-formatted, and patient-friendly response.
"""
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt_content),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    llm = ChatAnthropic(
        model=CLAUDE_MODEL_NAME,
        temperature=0.1, # Lower for more factual, less "creative" responses for medical info
        api_key=ANTHROPIC_API_KEY,
        max_tokens_to_sample=4000, # Increased max tokens for potentially longer, well-formatted responses
        default_request_timeout=60.0 # Increased timeout for complex queries
    )
    agent = create_tool_calling_agent(llm, available_tools, prompt_template)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=available_tools,
        verbose=False, # Set to True for detailed debugging logs
        handle_parsing_errors="I apologize, I had a slight difficulty processing that. Could you please try rephrasing your request or asking in a different way?",
        max_iterations=10, # Increased iterations for complex multi-tool calls if needed
        return_intermediate_steps=False
    )
    logger.info("Langchain agent executor and LLM client initialized.")
    return agent_executor, llm

# --- Streamlit UI ---
if "messages" not in st.session_state: st.session_state.messages = []
if "chat_history_for_agent" not in st.session_state: st.session_state.chat_history_for_agent = []
if "agent_executor" not in st.session_state and DATA_LOADED_SUCCESSFULLY:
    st.session_state.agent_executor, st.session_state.llm_client = get_langchain_agent_executor_and_llm()
elif "llm_client" not in st.session_state and DATA_LOADED_SUCCESSFULLY: # If only agent_executor was there
     _, st.session_state.llm_client = get_langchain_agent_executor_and_llm()


if "current_input" not in st.session_state: st.session_state.current_input = ""
if "clicked_suggestion" not in st.session_state: st.session_state.clicked_suggestion = None

# --- Sidebar ---
with st.sidebar:
    st.title("AI Assistant Controls")
    st.markdown("---")
    if DATA_LOADED_SUCCESSFULLY:
        if st.button("🔄 Start New Conversation", use_container_width=True, type="primary", help="Clears the current chat and starts fresh."):
            st.session_state.messages = []
            st.session_state.chat_history_for_agent = []
            # Re-initialize agent to clear its internal state if any, and get fresh LLM client
            st.session_state.agent_executor, st.session_state.llm_client = get_langchain_agent_executor_and_llm()
            st.session_state.current_input = ""
            st.session_state.clicked_suggestion = None
            logger.info("Conversation restarted by user.")
            st.success("New conversation started!")
            st.rerun()
    else:
        st.warning("Data loading failed. Restart unavailable.")
    st.markdown("---")
    st.subheader("Important Note")
    st.info(
        "This AI provides information to help you learn. It is **not a substitute for medical advice** from "
        "your doctor or other qualified healthcare professionals. Always discuss your health and treatment "
        "options with them."
    )
    st.caption(f"AI Model: `{CLAUDE_MODEL_NAME}`")
    st.caption(f"Embedding Model: `{EMBEDDING_MODEL_NAME}`")


# --- Main Chat Interface ---
st.title("AI Cancer Information Assistant 🧑‍⚕️")
st.caption("Your guide for exploring information on cancer drugs and clinical trials. Remember to always consult your doctor for medical advice.")
st.markdown("---")

if DATA_LOADED_SUCCESSFULLY:
    for message_data in st.session_state.messages:
        avatar_icon = ASSISTANT_AVATAR if message_data["role"] == "assistant" else USER_AVATAR
        with st.chat_message(message_data["role"], avatar=avatar_icon):
            st.markdown(message_data["content"], unsafe_allow_html=True)

    # Handle clicked suggestion by setting it as the current input
    if st.session_state.clicked_suggestion:
        st.session_state.current_input = st.session_state.clicked_suggestion
        st.session_state.clicked_suggestion = None # Clear after use
        # No st.rerun() here, let the normal input processing flow handle it

    user_typed_prompt_str = st.chat_input("Ask me about cancer treatments, drugs, or trials...", key="main_chat_input_v3")

    prompt_to_process = None
    if st.session_state.current_input: # Prioritize suggestion input
        prompt_to_process = st.session_state.current_input
        st.session_state.current_input = "" # Clear after use
    elif user_typed_prompt_str:
        prompt_to_process = user_typed_prompt_str

    if prompt_to_process:
        st.session_state.messages.append({"role": "user", "content": prompt_to_process})
        # No immediate display of user message here, will be handled by st.rerun() at the end of this block

        # Add to agent history immediately
        st.session_state.chat_history_for_agent.append(HumanMessage(content=prompt_to_process))

        # Temporary display for AI thinking
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            message_response_placeholder = st.empty()
            message_response_placeholder.markdown("🧑‍⚕️ Thinking and gathering information for you...")

        final_assistant_response = ""
        try:
            agent_invocation_input = {"input": prompt_to_process, "chat_history": st.session_state.chat_history_for_agent}

            if 'agent_executor' not in st.session_state or 'llm_client' not in st.session_state:
                 logger.warning("Agent executor or LLM client not in session state. Re-initializing.")
                 st.session_state.agent_executor, st.session_state.llm_client = get_langchain_agent_executor_and_llm()

            agent_response_data = st.session_state.agent_executor.invoke(agent_invocation_input)

            if isinstance(agent_response_data, dict) and "output" in agent_response_data:
                raw_ai_output = agent_response_data["output"]
            elif isinstance(agent_response_data, str): raw_ai_output = agent_response_data
            else:
                logger.error(f"Unexpected agent output structure: {type(agent_response_data)} - {agent_response_data}")
                raw_ai_output = "I apologize, I encountered an issue formulating a complete response. Could you try rephrasing?"

            final_assistant_response = str(raw_ai_output)

            st.session_state.chat_history_for_agent.append(AIMessage(content=final_assistant_response))

            max_history_pairs_count = 7
            if len(st.session_state.chat_history_for_agent) > max_history_pairs_count * 2:
                st.session_state.chat_history_for_agent = st.session_state.chat_history_for_agent[-(max_history_pairs_count * 2):]

        except Exception as agent_error:
            logger.error(f"Critical error during agent invocation: {agent_error}", exc_info=True)
            error_detail_snippet = str(agent_error)[:150]
            final_assistant_response = (
                "I'm truly sorry, but I encountered a significant technical issue while trying to process your request. "
                "The team has been notified. Please try again in a little while, or rephrase your question.\n\n"
                f"(Technical detail for admin: {error_detail_snippet}...)"
            )
            # Add error to history as AI response
            st.session_state.chat_history_for_agent.append(AIMessage(content=f"System error during processing: {final_assistant_response}"))

        st.session_state.messages.append({"role": "assistant", "content": final_assistant_response})

        # --- Generate and Store Suggestions Separately ---
        # These will be displayed on the *next* rerun after the main response is shown.
        st.session_state.follow_up_suggestions_list = [] # Clear previous suggestions
        if final_assistant_response and "technical issue" not in final_assistant_response.lower() and "error" not in final_assistant_response.lower() :
            # Generate suggestions based on the latest interaction
            # Don't show spinner here, it's a background task for next display
            st.session_state.follow_up_suggestions_list = generate_follow_up_suggestions(
                user_query=prompt_to_process,
                ai_main_response=final_assistant_response,
                llm_client=st.session_state.llm_client
            )
        st.rerun() # Rerun to display the new user message, AI response, and then suggestions.

    # Display follow-up suggestions if they exist from the previous turn
    if "follow_up_suggestions_list" in st.session_state and st.session_state.follow_up_suggestions_list:
        st.markdown("---")
        st.subheader("💭 What would you like to explore next?")
        num_suggestions_to_display = min(len(st.session_state.follow_up_suggestions_list), 3)
        if num_suggestions_to_display > 0:
            suggestion_columns = st.columns(num_suggestions_to_display)
            for idx, suggestion_item_text in enumerate(st.session_state.follow_up_suggestions_list[:num_suggestions_to_display]):
                # Create unique key for each button using timestamp and hash to ensure re-creation
                button_key_str = f"suggestion_btn_{idx}_{int(time.time())}_{hash(suggestion_item_text)}"
                if suggestion_columns[idx].button(f"{suggestion_item_text}", key=button_key_str, use_container_width=True, help=f"Click to ask: {suggestion_item_text}"):
                    st.session_state.clicked_suggestion = suggestion_item_text # Set flag
                    st.session_state.follow_up_suggestions_list = [] # Clear suggestions after click
                    logger.info(f"Suggestion clicked: '{suggestion_item_text}'")
                    st.rerun() # Rerun to process the clicked suggestion as new input

    # Initial greeting
    if not st.session_state.messages and DATA_LOADED_SUCCESSFULLY:
        initial_greeting_text = (
            "Hello! I'm an AI assistant here to help you explore information about cancer treatments, including "
            "drugs and clinical trials. My goal is to provide clear and understandable information.\n\n"
            "**How can I help you today?** For example, you could tell me about a specific diagnosis you're "
            "interested in, like 'Stage IV Lung Cancer with EGFR mutation'."
        )
        st.session_state.messages.append({"role": "assistant", "content": initial_greeting_text})
        st.session_state.chat_history_for_agent.append(AIMessage(content=initial_greeting_text))
        logger.info("Initial greeting displayed to user.")
        st.rerun()

elif not DATA_LOADED_SUCCESSFULLY:
    st.warning("The Cancer Information Assistant could not start. Please review error messages and application setup. Check logs for details.")

st.markdown("---")
st.caption("This AI tool provides information, not medical advice. Always consult your healthcare team for medical guidance.")