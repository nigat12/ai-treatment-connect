# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
from groq import Groq, RateLimitError, APIError
from dotenv import load_dotenv
import json
import logging
from datetime import datetime
import textwrap
import math
import time
import pickle # For saving/loading index map

# --- Configuration & Setup ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# CRITICAL: Check if Groq API key is loaded
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 FATAL ERROR: GROQ_API_KEY not found. Please set it in your .env file or environment variables and restart.")
    logging.critical("GROQ_API_KEY not found. Application cannot proceed.")
    st.stop()

# --- File Paths ---
DRUG_DATA_CSV = 'drug_data.csv'
TRIAL_DATA_XLSX = 'trials.xlsx'
DRUG_EMBEDDINGS_FILE = 'drug_embeddings.npy'
TRIAL_EMBEDDINGS_FILE = 'trial_embeddings.npy'
TRIAL_INDEX_MAP_FILE = 'trial_index_map.pkl'

# Insecure contact saving path (function retained, warnings removed from UI)
USER_DATA_FILE = "user_contact_data_demo.txt"

# --- Constants for Matching & UI ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
DRUG_TEXT_COLUMNS_FOR_EMBEDDING = ['Cancer Type']
TRIAL_TEXT_COLUMNS_FOR_EMBEDDING = ['Conditions']

# Default Thresholds (User configurable via sidebar)
DRUG_RELEVANCE_THRESHOLD_DEFAULT = 0.50
TRIAL_RELEVANCE_THRESHOLD_DEFAULT = 0.50

# Trial Filtering Criteria
TRIAL_FILTER_PRIMARY_OUTCOME_COLUMN = 'Primary Outcome Measures'
TRIAL_FILTER_PRIMARY_OUTCOME_TERM = 'Overall Survival'
TRIAL_FILTER_PHASES_COLUMN = 'Phases'
TRIAL_ACCEPTABLE_PHASES = ['PHASE1|PHASE2', 'PHASE2', 'PHASE2|PHASE3', 'PHASE3', 'PHASE4']
TRIAL_ACCEPTABLE_INDIVIDUAL_PHASES = set()
for phase_combo in TRIAL_ACCEPTABLE_PHASES:
    for phase in re.split(r'[|/,\s]+', phase_combo):
        if phase: TRIAL_ACCEPTABLE_INDIVIDUAL_PHASES.add(phase.strip().upper())

TRIAL_FILTER_STUDY_TYPE_COLUMN = 'Study Type'
TRIAL_FILTER_STUDY_TYPE_VALUE = 'INTERVENTIONAL'

# Display Limits
MAX_DRUGS_TO_DISPLAY = 10
MAX_TRIALS_TO_DISPLAY = 10

# UI Constants
ASSISTANT_AVATAR = "🧑‍⚕️"
USER_AVATAR = "👤"
ASSISTANT_NAME = "Assistant"
USER_NAME = "You"
# CHAT_CONTAINER_HEIGHT removed - using single page scroll

# --- Available Groq Models ---
AVAILABLE_MODELS = {
    "Llama 3 70B": "llama3-70b-8192",
    "Llama 3 8B": "llama3-8b-8192",
    "DeepSeek R1 Distill Llama 70B": "deepseek-r1-distill-llama-70b",
    "Gemma 2 Instruct": "gemma2-9b-it",
    "Mistral Saba 24B": "mistral-saba-24b",
}
DEFAULT_MODEL_DISPLAY_NAME = "Llama 3 70B"
DEFAULT_MODEL_ID = AVAILABLE_MODELS.get(DEFAULT_MODEL_DISPLAY_NAME, list(AVAILABLE_MODELS.values())[0])

# --- Chatbot Questions & Stages ---
STAGES = {
    "INIT": 0, "GET_DIAGNOSIS": 1, "GET_STAGE": 2, "GET_BIOMARKERS": 3,
    "GET_PRIOR_TREATMENT": 4, "GET_IMAGING": 5, "PROCESS_INFO_SHOW_DRUGS": 6,
    "ASK_CONSENT": 7, "GET_NAME": 8, "GET_EMAIL": 9, "GET_PHONE": 10,
    "SAVE_CONTACT_SHOW_TRIALS": 11, "SHOW_TRIALS_NO_CONSENT": 12,
    "FINAL_SUMMARY": 13, "END": 14
}

STAGE_PROMPTS = {
    STAGES["INIT"]: "Welcome! I can help explore potential treatment options based on study data. Please provide some details to begin.\n\nFirst: 👇",
    STAGES["GET_DIAGNOSIS"]: "Q: What is the primary diagnosis? (e.g., Breast Cancer)",
    STAGES["GET_STAGE"]: "Q: What is the stage or progression? (e.g., Stage IV, Metastatic)",
    STAGES["GET_BIOMARKERS"]: "Q: Are there any known biomarkers? (e.g., HR-positive). List them or type 'None'.",
    STAGES["GET_PRIOR_TREATMENT"]: "Q: What treatments have been received previously? (e.g., Chemotherapy, Immunotherapy)",
    STAGES["GET_IMAGING"]: "Q: Any recent imaging results indicating current status? (e.g., Stable disease, Progression)",
    STAGES["ASK_CONSENT"]: "Q: Would you like to save your query context and contact information for potential follow-up?",
    STAGES["GET_NAME"]: "Q: Please enter your First and Last Name:",
    STAGES["GET_EMAIL"]: "Q: Please enter your Email Address:",
    STAGES["GET_PHONE"]: "Q: Please enter your Phone Number (optional):",
    STAGES["END"]: "Exploration complete. Please remember to discuss all findings and options with your healthcare provider."
    # Internal stages removed, handled by logic/spinners
}

STAGE_NAMES = {v: k for k, v in STAGES.items()}

# --- Helper Functions (Parsing, Sorting, Saving) ---
# (parse_time_to_months, parse_improvement_percentage, sort_key_with_none, check_phases, save_user_data, get_llm_client remain unchanged from previous version)

def parse_time_to_months(time_str):
    if isinstance(time_str, (int, float)): return float(time_str)
    if not isinstance(time_str, str): return None
    time_str = time_str.strip().lower()
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
    perc_str = perc_str.strip().lower()
    if perc_str in ['n/a', 'not applicable', 'not reported', 'not statistically significant', 'nss', '', 'nan']: return None
    match = re.match(r'(-?\d+(\.\d+)?)\s*%', perc_str)
    if match: return float(match.group(1))
    try: return float(perc_str)
    except ValueError: return None

def sort_key_with_none(value, reverse=True):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float('-inf') if reverse else float('inf')
    try: return float(value)
    except (ValueError, TypeError): return float('-inf') if reverse else float('inf')

def check_phases(trial_phases_raw):
    if not isinstance(trial_phases_raw, str) or not trial_phases_raw.strip(): return False
    trial_individual_phases = re.split(r'[|/,\s]+', trial_phases_raw.strip())
    for phase in trial_individual_phases:
        if phase.strip().upper() in TRIAL_ACCEPTABLE_INDIVIDUAL_PHASES:
            return True
    return False

def save_user_data(data):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(USER_DATA_FILE, "a", encoding="utf-8") as f:
            f.write(f"--- Contact Entry: {timestamp} ---\n")
            f.write(f"  Consent Given: {data.get('ConsentGiven', 'N/A')}\n")
            if data.get('ConsentGiven') is True:
                f.write(f"  Name: {data.get('Name', 'N/A')}\n")
                f.write(f"  Email: {data.get('Email', 'N/A')}\n")
                f.write(f"  Phone: {data.get('Phone', 'N/A')}\n")
            f.write(f"  Context:\n")
            f.write(f"    Diagnosis: {data.get('Diagnosis', 'N/A')}\n")
            f.write(f"    StageProgression: {data.get('StageProgression', 'N/A')}\n")
            f.write(f"    Biomarkers: {data.get('Biomarkers', 'N/A')}\n")
            f.write(f"    PriorTreatment: {data.get('PriorTreatment', 'N/A')}\n")
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

def get_llm_client():
    try:
        return Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        logging.critical(f"Failed to initialize Groq client: {e}", exc_info=True)
        st.error("FATAL ERROR: Could not initialize AI client.")
        st.stop()

# --- Caching and Data/Embedding Loading Functions ---
# (load_sentence_transformer_model, load_and_preprocess_drug_data, get_or_generate_drug_embeddings,
#  load_and_preprocess_trial_data, get_or_generate_trial_embeddings remain unchanged from previous version)

@st.cache_resource(show_spinner="Loading embedding model...")
def load_sentence_transformer_model(model_name=EMBEDDING_MODEL_NAME):
    try:
        logging.info(f"Loading Sentence Transformer model: {model_name}")
        model = SentenceTransformer(model_name)
        logging.info(f"Model {model_name} loaded.")
        return model
    except Exception as e:
        logging.error(f"Error loading Sentence Transformer model '{model_name}': {e}", exc_info=True)
        st.error(f"Failed to load the embedding model ({model_name}).")
        st.stop()

@st.cache_data(show_spinner="Loading drug data...")
def load_and_preprocess_drug_data(csv_path):
    try:
        logging.info(f"Loading drug data from {csv_path}")
        df = pd.read_csv(csv_path)
        logging.info(f"Drug data loaded. Shape: {df.shape}")
        missing_cols = [col for col in DRUG_TEXT_COLUMNS_FOR_EMBEDDING if col not in df.columns]
        if missing_cols: raise ValueError(f"Missing required columns in {csv_path}: {missing_cols}")
        df['combined_text_for_embedding'] = df[DRUG_TEXT_COLUMNS_FOR_EMBEDDING].fillna('').astype(str).agg(' '.join, axis=1)
        df['Treatment_OS_Months_Parsed'] = df.get('Treatment_OS', pd.Series([None]*len(df))).apply(parse_time_to_months)
        df['Control_OS_Months_Parsed'] = df.get('Control_OS', pd.Series([None]*len(df))).apply(parse_time_to_months)
        df['OS_Improvement_Percentage_Parsed'] = df.get('OS_Improvement (%)', pd.Series([None]*len(df))).apply(parse_improvement_percentage)
        df['Treatment_PFS_Months_Parsed'] = df.get('Treatment_PFS', pd.Series([None]*len(df))).apply(parse_time_to_months)
        df['Control_PFS_Months_Parsed'] = df.get('Control_PFS', pd.Series([None]*len(df))).apply(parse_time_to_months)
        df['PFS_Improvement_Percentage_Parsed'] = df.get('PFS_Improvement (%)', pd.Series([None]*len(df))).apply(parse_improvement_percentage)
        df['Calculated_OS_Improvement_Months'] = df.apply(lambda row: row['Treatment_OS_Months_Parsed'] - row['Control_OS_Months_Parsed'] if pd.notna(row['Treatment_OS_Months_Parsed']) and pd.notna(row['Control_OS_Months_Parsed']) else None, axis=1)
        df['Calculated_PFS_Improvement_Months'] = df.apply(lambda row: row['Treatment_PFS_Months_Parsed'] - row['Control_PFS_Months_Parsed'] if pd.notna(row['Treatment_PFS_Months_Parsed']) and pd.notna(row['Control_PFS_Months_Parsed']) else None, axis=1)
        logging.info("Drug data preprocessing complete.")
        return df.fillna('N/A')
    except FileNotFoundError: logging.error(f"Drug data file not found: {csv_path}"); st.error(f"ERROR: Drug data file ({csv_path}) not found."); st.stop()
    except ValueError as ve: logging.error(f"ValueError processing drug data: {ve}", exc_info=True); st.error(f"ERROR: Problem processing drug data file ({csv_path}). Details: {ve}"); st.stop()
    except Exception as e: logging.error(f"Unexpected error loading/processing drug data {csv_path}: {e}", exc_info=True); st.error(f"An unexpected error occurred loading drug data from {csv_path}."); st.stop()

def get_or_generate_drug_embeddings(_drug_df, _model, embeddings_path=DRUG_EMBEDDINGS_FILE):
    if os.path.exists(embeddings_path):
        try:
            logging.info(f"Loading existing drug embeddings from {embeddings_path}")
            embeddings = np.load(embeddings_path)
            if embeddings.shape[0] == len(_drug_df) and embeddings.shape[1] == _model.get_sentence_embedding_dimension():
                 logging.info("Drug embeddings loaded successfully.")
                 return embeddings
            else: logging.warning(f"Loaded drug embeddings shape mismatch. Regenerating.")
        except Exception as e: logging.error(f"Error loading drug embeddings: {e}. Regenerating.", exc_info=True)
    logging.info("Generating new drug embeddings...")
    try:
        texts_to_embed = _drug_df['combined_text_for_embedding'].tolist()
        if not texts_to_embed: logging.warning("No text in drug data for embedding."); return np.array([])
        embeddings = _model.encode(texts_to_embed, show_progress_bar=True, convert_to_numpy=True)
        logging.info("Drug embeddings generated.")
        try: np.save(embeddings_path, embeddings); logging.info(f"Drug embeddings saved to {embeddings_path}")
        except Exception as e: logging.error(f"Error saving drug embeddings: {e}", exc_info=True); st.warning(f"Could not save generated drug embeddings.")
        return embeddings
    except Exception as e: logging.error(f"Error generating drug embeddings: {e}", exc_info=True); st.error("Error generating drug embeddings."); return np.array([])

@st.cache_data(show_spinner="Loading trial data...")
def load_and_preprocess_trial_data(xlsx_path):
    try:
        logging.info(f"Loading trial data from {xlsx_path}")
        df = pd.read_excel(xlsx_path)
        logging.info(f"Trial data loaded. Shape: {df.shape}")
        missing_cols = [col for col in TRIAL_TEXT_COLUMNS_FOR_EMBEDDING if col not in df.columns]
        if missing_cols: raise ValueError(f"Missing required columns in {xlsx_path}: {missing_cols}")
        df['combined_text_for_embedding'] = df[TRIAL_TEXT_COLUMNS_FOR_EMBEDDING].fillna('').astype(str).agg(' '.join, axis=1)
        if TRIAL_FILTER_PRIMARY_OUTCOME_COLUMN in df.columns: df[TRIAL_FILTER_PRIMARY_OUTCOME_COLUMN] = df[TRIAL_FILTER_PRIMARY_OUTCOME_COLUMN].fillna('').astype(str)
        if TRIAL_FILTER_PHASES_COLUMN in df.columns: df[TRIAL_FILTER_PHASES_COLUMN] = df[TRIAL_FILTER_PHASES_COLUMN].fillna('').astype(str)
        if TRIAL_FILTER_STUDY_TYPE_COLUMN in df.columns: df[TRIAL_FILTER_STUDY_TYPE_COLUMN] = df[TRIAL_FILTER_STUDY_TYPE_COLUMN].fillna('').astype(str)
        logging.info("Trial data preprocessing complete.")
        return df.fillna('N/A')
    except FileNotFoundError: logging.error(f"Trial data file not found: {xlsx_path}"); st.error(f"ERROR: Trial data file ({xlsx_path}) not found."); st.stop()
    except ValueError as ve: logging.error(f"ValueError processing trial data: {ve}", exc_info=True); st.error(f"ERROR: Problem processing trial data file ({xlsx_path}). Details: {ve}"); st.stop()
    except Exception as e: logging.error(f"Unexpected error loading/processing trial data {xlsx_path}: {e}", exc_info=True); st.error(f"An unexpected error occurred loading trial data from {xlsx_path}."); st.stop()

def get_or_generate_trial_embeddings(_trial_df, _model, embeddings_path=TRIAL_EMBEDDINGS_FILE, map_path=TRIAL_INDEX_MAP_FILE):
    embeddings, index_map = None, None
    if os.path.exists(embeddings_path) and os.path.exists(map_path):
        try:
            logging.info(f"Loading existing trial embeddings/map")
            embeddings = np.load(embeddings_path)
            with open(map_path, 'rb') as f: index_map = pickle.load(f)
            if embeddings is not None and index_map is not None and \
               embeddings.shape[0] == len(index_map) and \
               embeddings.shape[1] == _model.get_sentence_embedding_dimension() and \
               max(index_map.keys(), default=-1) < len(_trial_df):
                 logging.info("Trial embeddings/map loaded successfully.")
                 return embeddings, index_map
            else: logging.warning(f"Trial embeddings/map shape mismatch or invalid data. Regenerating."); embeddings, index_map = None, None
        except Exception as e: logging.error(f"Error loading trial embeddings/map: {e}. Regenerating.", exc_info=True); embeddings, index_map = None, None
    logging.info("Generating new trial embeddings and index map...")
    try:
        non_empty_mask = _trial_df['combined_text_for_embedding'].str.strip() != ''
        non_empty_indices = _trial_df.index[non_empty_mask].tolist()
        texts_to_embed = _trial_df.loc[non_empty_indices, 'combined_text_for_embedding'].tolist()
        if not texts_to_embed: logging.warning("No non-empty text in trial data for embedding."); return np.array([]), {}
        embeddings = _model.encode(texts_to_embed, show_progress_bar=True, convert_to_numpy=True)
        index_map = {original_idx: emb_idx for emb_idx, original_idx in enumerate(non_empty_indices)}
        logging.info("Trial embeddings/map generated.")
        try:
            np.save(embeddings_path, embeddings)
            with open(map_path, 'wb') as f: pickle.dump(index_map, f)
            logging.info(f"Trial embeddings/map saved.")
        except Exception as e: logging.error(f"Error saving trial embeddings/map: {e}", exc_info=True); st.warning("Could not save generated trial embeddings/map.")
        return embeddings, index_map
    except Exception as e: logging.error(f"Error generating trial embeddings: {e}", exc_info=True); st.error("Error generating trial embeddings."); return np.array([]), {}


# --- Local Matching Functions ---
# Pass session_state threshold values to these functions

def find_relevant_drugs_local(df_drugs: pd.DataFrame, drug_embeddings: np.ndarray, model: SentenceTransformer,
                              user_cancer_type_raw: str, user_stage_raw: str, user_biomarkers_raw: str,
                              relevance_threshold: float, # Now passed as argument
                              max_results: int = MAX_DRUGS_TO_DISPLAY):
    logging.info("Starting local drug search...")
    if drug_embeddings.size == 0: logging.warning("Drug embeddings empty."); return []
    user_biomarkers_list = [b.strip() for b in user_biomarkers_raw.split(',') if b.strip()]
    user_query_text = f"{user_cancer_type_raw} {user_stage_raw} {' '.join(user_biomarkers_list)}".strip()
    logging.info(f"Drug query: '{user_query_text}', Threshold: {relevance_threshold:.2f}")
    if not user_query_text: logging.warning("User drug query empty."); return []
    try: user_embedding = model.encode(user_query_text, convert_to_numpy=True)
    except Exception as e: logging.error(f"Error generating drug query embedding: {e}", exc_info=True); st.warning("Could not generate query embedding."); return []
    potential_results = []
    try: similarities = cosine_similarity([user_embedding], drug_embeddings)[0]
    except ValueError as e: logging.error(f"Error calculating drug similarities: {e}", exc_info=True); st.error("Error comparing query to drug data."); return []
    for index, row in df_drugs.iterrows():
        if index >= len(similarities): logging.warning(f"Drug index {index} out of bounds for similarities."); continue
        semantic_sim = similarities[index]
        if semantic_sim >= relevance_threshold:
            result_dict = {'index': index, 'semantic_similarity': semantic_sim, **row.to_dict()} # Include all columns
            potential_results.append(result_dict)
    potential_results.sort(key=lambda x: (
        sort_key_with_none(x['semantic_similarity'], reverse=True),
        sort_key_with_none(x.get('Calculated_OS_Improvement_Months'), reverse=True),
        sort_key_with_none(x.get('OS_Improvement_Percentage_Parsed') is not None, reverse=True),
        sort_key_with_none(x.get('OS_Improvement_Percentage_Parsed'), reverse=True),
        sort_key_with_none(x.get('PFS_Improvement_Percentage_Parsed') is not None, reverse=True),
        sort_key_with_none(x.get('PFS_Improvement_Percentage_Parsed'), reverse=True),
    ), reverse=False)
    potential_results.reverse()
    logging.info(f"Found {len(potential_results)} relevant drugs passing threshold {relevance_threshold:.2f}.")
    return potential_results[:max_results]

def find_relevant_trials_local(df_trials: pd.DataFrame, trial_embeddings: np.ndarray, index_to_embedding_index: dict,
                               model: SentenceTransformer,
                               user_cancer_type_raw: str, user_stage_raw: str, user_biomarkers_raw: str,
                               relevance_threshold: float, # Now passed as argument
                               max_results: int = MAX_TRIALS_TO_DISPLAY):
    logging.info("Starting local trial search...")
    if trial_embeddings.size == 0 or not index_to_embedding_index: logging.warning("Trial embeddings/map empty."); return []
    user_biomarkers_list = [b.strip() for b in user_biomarkers_raw.split(',') if b.strip()]
    user_query_text = f"{user_cancer_type_raw} {user_stage_raw} {' '.join(user_biomarkers_list)}".strip()
    logging.info(f"Trial query: '{user_query_text}', Threshold: {relevance_threshold:.2f}")
    if not user_query_text: logging.warning("User trial query empty."); return []
    try: user_embedding = model.encode(user_query_text, convert_to_numpy=True)
    except Exception as e: logging.error(f"Error generating trial query embedding: {e}", exc_info=True); st.warning("Could not generate query embedding."); return []
    potential_results = []
    for index, row in df_trials.iterrows():
        primary_outcome_text = row.get(TRIAL_FILTER_PRIMARY_OUTCOME_COLUMN, 'N/A').lower()
        if TRIAL_FILTER_PRIMARY_OUTCOME_TERM.lower() not in primary_outcome_text: continue
        trial_phases_raw = row.get(TRIAL_FILTER_PHASES_COLUMN, 'N/A')
        if not check_phases(trial_phases_raw): continue
        trial_study_type = row.get(TRIAL_FILTER_STUDY_TYPE_COLUMN, 'N/A').upper()
        if trial_study_type != TRIAL_FILTER_STUDY_TYPE_VALUE.upper(): continue
        if index not in index_to_embedding_index: continue
        embedding_index = index_to_embedding_index[index]
        if embedding_index >= len(trial_embeddings): continue
        try: semantic_sim = cosine_similarity([user_embedding], [trial_embeddings[embedding_index]])[0][0]
        except ValueError as e: logging.error(f"Error calculating trial similarity for index {index}: {e}", exc_info=True); continue
        except Exception as e: logging.error(f"Unexpected error calculating similarity for trial index {index}: {e}", exc_info=True); continue
        if semantic_sim >= relevance_threshold:
            result_dict = {
                'index': index, 'semantic_similarity': semantic_sim,
                'nct_id': row.get('NCT Number', 'N/A'), 'title': row.get('Study Title', 'N/A'),
                'status': row.get('Study Status', 'N/A'), 'conditions': row.get('Conditions', 'N/A'),
                'interventions': row.get('Interventions', 'N/A'), 'phases': trial_phases_raw,
                'brief_summary': row.get('Brief Summary', 'N/A'),
                'primary_outcome': row.get(TRIAL_FILTER_PRIMARY_OUTCOME_COLUMN, 'N/A'),
                'study_type': trial_study_type,
                'url': f"https://clinicaltrials.gov/study/{row.get('NCT Number', '')}" if row.get('NCT Number', 'N/A') != 'N/A' else "#"
            }
            potential_results.append(result_dict)
    phase_order = {'PHASE4': 5, 'PHASE3': 4, 'PHASE2|PHASE3': 3, 'PHASE2': 2, 'PHASE1|PHASE2': 1}
    def get_phase_sort_value(phases_raw):
        if not isinstance(phases_raw, str): return 0
        highest_phase_val = 0
        individual_phases = re.split(r'[|/,\s]+', phases_raw.strip().upper())
        for phase in individual_phases: highest_phase_val = max(highest_phase_val, phase_order.get(phase, 0))
        return highest_phase_val
    potential_results.sort(key=lambda x: (
        sort_key_with_none(x['semantic_similarity'], reverse=True),
        get_phase_sort_value(x.get('phases')),
        0 if 'recruiting' in str(x.get('status', '')).lower() else 1
    ), reverse=True)
    logging.info(f"Found {len(potential_results)} relevant trials passing filters/threshold {relevance_threshold:.2f}.")
    return potential_results[:max_results]


# --- LLM Generation Functions ---
# (generate_llm_response, generate_drug_summary_llm, generate_trial_summary_llm,
#  generate_result_interpretation_llm, generate_final_summary_llm remain unchanged from previous version)

def generate_llm_response(client, model_id, system_prompt, user_prompt, max_tokens=300, temperature=0.2):
    try:
        chat_completion = client.chat.completions.create(messages=[{"role": "system", "content": system_prompt},{"role": "user", "content": user_prompt}], model=model_id, temperature=temperature, max_tokens=max_tokens)
        response = chat_completion.choices[0].message.content
        response = response.strip().replace("```json", "").replace("```", "").strip()
        unwanted_phrases = ["Here is the summary:", "Summary:", "Here's the interpretation:", "Interpretation:", "Okay, here's the explanation:", "Explanation:"]
        for phrase in unwanted_phrases:
             if response.lower().startswith(phrase.lower()): response = response[len(phrase):].strip()
        return response
    except RateLimitError as e: logging.error(f"Groq Rate Limit Error: {e}", exc_info=True); st.warning("AI assistant unavailable (rate limit)."); return "Error: AI assistant rate limit."
    except APIError as e: logging.error(f"Groq API Error: {e}", exc_info=True); st.warning(f"AI assistant API error ({e.status_code})."); return f"Error: AI assistant API error ({e.status_code})."
    except Exception as e: logging.error(f"Unexpected LLM error: {e}", exc_info=True); st.warning("Unexpected error contacting AI assistant."); return "Error: Unexpected AI issue."

def generate_drug_summary_llm(user_inputs, drug_results, model_id):
    logging.info(f"[LLM Call - Drug Summary] For {len(drug_results)} drugs.")
    client = get_llm_client()
    if not drug_results: return f"No matching drug studies found for: Diagnosis '{user_inputs.get('diagnosis', 'N/A')}', Stage '{user_inputs.get('stage', 'N/A')}', Biomarkers '{user_inputs.get('biomarkers', 'N/A')}'."
    drug_list_str = "\n".join([f"- {d.get('Drug Name', 'N/A')} (Sim: {d.get('semantic_similarity', 0):.2f})" for d in drug_results[:5]])
    if len(drug_results) > 5: drug_list_str += f"\n- ...and {len(drug_results) - 5} more."
    system_prompt = "Summarize drug study findings concisely based on relevance. Do NOT give medical advice. Mention number found and basis (similarity, outcome)."
    user_prompt = f"""User Input: Dx="{user_inputs.get('diagnosis', 'N/A')}", Stg="{user_inputs.get('stage', 'N/A')}", Bio="{user_inputs.get('biomarkers', 'N/A')}" Found {len(drug_results)} relevant studies. Top: {drug_list_str} Provide a brief (2-3 sentences) summary. State number found/basis. Mention ranking (relevance, OS/PFS). Advise review/consult doctor."""
    summary = generate_llm_response(client, model_id, system_prompt, user_prompt, max_tokens=250)
    logging.info(f"LLM drug summary: {summary}")
    return summary

def generate_trial_summary_llm(user_inputs, trial_results, model_id):
    logging.info(f"[LLM Call - Trial Summary] For {len(trial_results)} trials.")
    client = get_llm_client()
    if not trial_results: return f"No matching trials found after filters (Outcome: OS, Phase 2+, Interventional) and relevance for: Dx '{user_inputs.get('diagnosis', 'N/A')}', Stg '{user_inputs.get('stage', 'N/A')}', Bio '{user_inputs.get('biomarkers', 'N/A')}'."
    recruiting_count = sum(1 for t in trial_results if 'recruiting' in t.get('status', '').lower())
    trial_list_str = "\n".join([f"- {t.get('nct_id', 'N/A')}: {t.get('title', 'N/A')} (Sts: {t.get('status', 'N/A')}, Ph: {t.get('phases', 'N/A')}, Sim: {t.get('semantic_similarity', 0):.2f})" for t in trial_results[:5]])
    if len(trial_results) > 5: trial_list_str += f"\n- ...and {len(trial_results) - 5} more."
    system_prompt = "Summarize trial findings concisely based on filters/relevance. Do NOT give medical advice. Mention number found, recruiting count, basis (filters + relevance)."
    user_prompt = f"""User Input: Dx="{user_inputs.get('diagnosis', 'N/A')}", Stg="{user_inputs.get('stage', 'N/A')}", Bio="{user_inputs.get('biomarkers', 'N/A')}" Found {len(trial_results)} relevant trials after filters (Outcome: '{TRIAL_FILTER_PRIMARY_OUTCOME_TERM}', Phase: {TRIAL_ACCEPTABLE_PHASES}, Type: '{TRIAL_FILTER_STUDY_TYPE_VALUE}') & sim. threshold. {recruiting_count} recruiting. Top: {trial_list_str} Provide brief (2-3 sentences) summary. State number found/basis. Mention recruiting count. Advise review/consult doctor."""
    summary = generate_llm_response(client, model_id, system_prompt, user_prompt, max_tokens=300)
    logging.info(f"LLM trial summary: {summary}")
    return summary

def generate_result_interpretation_llm(result_type, result_data, user_context, model_id):
    logging.info(f"[LLM Call - Interpretation] For {result_type}: {result_data.get('Drug Name') or result_data.get('nct_id')}")
    client = get_llm_client()
    system_prompt = f"Explain clinical study findings ({result_type}). Interpret OS, PFS, phase, status objectively. CRITICAL: NO MEDICAL ADVICE. State info needs discussion with doctor. Focus ONLY on data provided for this result. Concise (3-5 sentences). Explain OS (Overall Survival), PFS (Progression-Free Survival)."
    user_prompt = f"User Context: Dx: {user_context.get('diagnosis', 'N/A')}, Stg: {user_context.get('stage', 'N/A')}, Bio: {user_context.get('biomarkers', 'N/A')}\n\n"
    if result_type == "drug":
        user_prompt += f"Drug Study:\nDrug: {result_data.get('Drug Name', 'N/A')} | Relevance: {result_data.get('semantic_similarity', 0):.3f}\nSummary: {textwrap.shorten(result_data.get('Brief Study Summary', 'N/A'), width=200)}\n"
        user_prompt += f"OS: Tx=`{result_data.get('Treatment_OS', 'N/A')}`, Ctrl=`{result_data.get('Control_OS', 'N/A')}`, Imprv=`{result_data.get('OS_Improvement (%)', 'N/A')}` (Calc Months: {result_data.get('Calculated_OS_Improvement_Months', 'N/A')})\n"
        user_prompt += f"PFS: Tx=`{result_data.get('Treatment_PFS', 'N/A')}`, Ctrl=`{result_data.get('Control_PFS', 'N/A')}`, Imprv=`{result_data.get('PFS_Improvement (%)', 'N/A')}`\n\n"
        user_prompt += f"Explain these results for {result_data.get('Drug Name', 'N/A')}. Define OS/PFS. Indicate suggested improvement vs control (avoid certainty). Mention relevance score source. Advise doctor discussion."
    elif result_type == "trial":
        user_prompt += f"Trial:\nNCT: {result_data.get('nct_id', 'N/A')} | Relevance: {result_data.get('semantic_similarity', 0):.3f}\nTitle: {result_data.get('title', 'N/A')}\n"
        user_prompt += f"Status: `{result_data.get('status', 'N/A')}` | Phase: `{result_data.get('phases', 'N/A')}`\nCond: {result_data.get('conditions', 'N/A')} | Interv: {result_data.get('interventions', 'N/A')}\n"
        user_prompt += f"Prim Outcome Focus ('{TRIAL_FILTER_PRIMARY_OUTCOME_TERM}'): Yes\nSummary: {textwrap.shorten(result_data.get('brief_summary', 'N/A'), width=200)}\n\n"
        user_prompt += f"Explain trial {result_data.get('nct_id', 'N/A')}. Explain Status (e.g., Recruiting) & Phase (e.g., Phase 3). Mention relevance score source. State trial goal. Advise doctor discussion on eligibility."
    else: return "Error: Invalid result type."
    interpretation = generate_llm_response(client, model_id, system_prompt, user_prompt, max_tokens=400, temperature=0.3)
    logging.info(f"LLM interpretation: {interpretation}")
    return interpretation

def generate_final_summary_llm(user_inputs, drug_results, trial_results, model_id):
    logging.info(f"[LLM Call - Final Summary]")
    client = get_llm_client()
    drug_count = len(drug_results); trial_count = len(trial_results)
    recruiting_trial_count = sum(1 for t in trial_results if 'recruiting' in t.get('status', '').lower())
    drug_summary_part = f"{drug_count} drug studies identified." if drug_count > 0 else "No matching drug studies found."
    trial_summary_part = f"{trial_count} trials identified ({recruiting_trial_count} recruiting)." if trial_count > 0 else "No matching trials found."
    system_prompt = "Provide final summary of findings. Be professional, concise. Strongly emphasize need for medical consultation."
    user_prompt = f"""User Context: Dx: {user_inputs.get("diagnosis", "N/A")}, Stg: {user_inputs.get("stage", "N/A")}, Bio: {user_inputs.get("biomarkers", "N/A")} Findings Summary: Drugs: {drug_summary_part} Trials: {trial_summary_part} Synthesize into brief (3-4 sentences) conclusion. 1. Acknowledge exploration. 2. State counts found. 3. Reiterate this is NOT medical advice. 4. Emphasize CRITICAL need to discuss ALL options with qualified provider."""
    summary = generate_llm_response(client, model_id, system_prompt, user_prompt, max_tokens=350, temperature=0.3)
    logging.info(f"LLM final summary: {summary}")
    # Add final mandatory reminder
    reminder = "\n\n**Reminder:** Discuss these informational findings thoroughly with your healthcare provider before making any decisions."
    return summary + reminder

# --- Streamlit App ---

st.set_page_config(page_title="AI Drug & Trial Match", layout="wide", initial_sidebar_state="expanded")

# --- Load Data and Model (Initialization) ---
data_loaded_successfully = False
try:
    with st.spinner("Initializing resources... This may take a moment on first run."):
        embedding_model = load_sentence_transformer_model()
        df_drugs_processed = load_and_preprocess_drug_data(DRUG_DATA_CSV)
        drug_embeddings_array = get_or_generate_drug_embeddings(df_drugs_processed, embedding_model)
        df_trials_processed = load_and_preprocess_trial_data(TRIAL_DATA_XLSX)
        trial_embeddings_array, trial_index_map = get_or_generate_trial_embeddings(df_trials_processed, embedding_model)

    if drug_embeddings_array.size > 0 and trial_embeddings_array.size > 0 and trial_index_map is not None:
        data_loaded_successfully = True
        logging.info("All resources initialized successfully.")
    else:
        logging.error("Failed to initialize embeddings or index map.")
        st.error("Failed to initialize necessary data embeddings. Cannot proceed.")
except Exception as e:
    logging.critical("App initialization failed.", exc_info=True)

# --- Initialize Session State ---
def initialize_session():
    if 'stage' not in st.session_state:
        logging.info("Initializing new session state.")
        st.session_state.stage = STAGES["INIT"]
        st.session_state.user_inputs = {}
        st.session_state.messages = []
        st.session_state.drug_results = []
        st.session_state.trial_results = []
        st.session_state.consent_given = None
        # Processing flags
        st.session_state.drug_results_processed = False
        st.session_state.trial_results_processed = False
        st.session_state.contact_saved = False
        st.session_state.final_summary_processed = False
        # Configurable thresholds
        st.session_state.drug_threshold = DRUG_RELEVANCE_THRESHOLD_DEFAULT
        st.session_state.trial_threshold = TRIAL_RELEVANCE_THRESHOLD_DEFAULT
        # Default Model
        if 'model_id' not in st.session_state:
            st.session_state.model_id = DEFAULT_MODEL_ID

        # Initial prompts based on successful loading
        if data_loaded_successfully:
             st.session_state.messages.append({"role": "assistant", "content": STAGE_PROMPTS[STAGES["INIT"]], "type": "info"})
             st.session_state.messages.append({"role": "assistant", "content": STAGE_PROMPTS[STAGES["GET_DIAGNOSIS"]]})
             st.session_state.stage = STAGES["GET_DIAGNOSIS"]
        else:
             st.session_state.messages.append({"role": "assistant", "content": "⚠️ Application initialization failed. Cannot proceed.", "type": "error"})
             st.session_state.stage = STAGES["END"]

    # Ensure core keys exist if session persists partially
    keys_defaults = {
        'user_inputs': {}, 'drug_results': [], 'trial_results': [], 'consent_given': None,
        'messages': [], 'model_id': DEFAULT_MODEL_ID, 'drug_threshold': DRUG_RELEVANCE_THRESHOLD_DEFAULT,
        'trial_threshold': TRIAL_RELEVANCE_THRESHOLD_DEFAULT, 'drug_results_processed': False,
        'trial_results_processed': False, 'contact_saved': False, 'final_summary_processed': False
    }
    for key, default_val in keys_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_val

initialize_session()

# --- Sidebar ---
with st.sidebar:
    st.subheader("⚙️ Configuration")
    # LLM Model Selector
    model_display_names = list(AVAILABLE_MODELS.keys())
    current_model_display_name = next((name for name, mid in AVAILABLE_MODELS.items() if mid == st.session_state.model_id), DEFAULT_MODEL_DISPLAY_NAME)
    try: default_index = model_display_names.index(current_model_display_name)
    except ValueError: default_index = 0
    selected_model_display_name = st.selectbox(
        "LLM Model:", options=model_display_names, index=default_index, key="model_select_widget",
        help="AI model for summaries/explanations."
    )
    new_model_id = AVAILABLE_MODELS[selected_model_display_name]
    if new_model_id != st.session_state.model_id:
        st.session_state.model_id = new_model_id; st.toast(f"LLM Model: {selected_model_display_name}", icon="🤖")
        logging.info(f"LLM Model updated to {new_model_id}")

    st.caption(f"Embedding: `{EMBEDDING_MODEL_NAME}`")
    st.divider()
    # Similarity Threshold Sliders
    st.session_state.drug_threshold = st.slider(
        "Drug Relevance Threshold", min_value=0.0, max_value=1.0,
        value=st.session_state.drug_threshold, step=0.05,
        help="Minimum similarity score (0-1) for a drug study to be considered relevant. Higher is stricter."
    )
    st.session_state.trial_threshold = st.slider(
        "Trial Relevance Threshold", min_value=0.0, max_value=1.0,
        value=st.session_state.trial_threshold, step=0.05,
        help="Minimum similarity score (0-1) for a clinical trial to be considered relevant after filtering. Higher is stricter."
    )
    st.divider()
    # Restart Button
    if st.button("🔄 Restart Conversation", key="restart_sidebar"):
        logging.info("Restarting session from sidebar.")
        keys_to_clear = [k for k in st.session_state.keys() if k not in ['model_id', 'model_select_widget']] # Keep LLM choice
        for key in keys_to_clear: del st.session_state[key]
        initialize_session()
        st.rerun()
    st.divider()
    st.markdown("---")
    st.caption("Status Info:")
    current_stage_num = st.session_state.get('stage', STAGES["INIT"])
    st.write(f"Stage: {STAGE_NAMES.get(current_stage_num, 'Unknown')}")
    st.write(f"Initialized: {'✅' if data_loaded_successfully else '❌'}")

# --- Main App Area ---

# Header Section (Not sticky, but at the top)
st.title("🧑‍⚕️ AI Drug & Trial Match")
# st.caption("Explore information from study data. Discuss all findings with your healthcare provider.")

# Chat messages area (renders directly into the main flow)
message_container = st.container() # Use a container just for logical grouping if needed, but not for height/scroll
with message_container:
    for message in st.session_state.get('messages', []):
        avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else USER_AVATAR
        with st.chat_message(name=message["role"], avatar=avatar):
            if message.get("type") == "expander":
                with st.expander(message.get("title", "Details"), expanded=message.get("expanded", False)):
                    st.markdown(message["content"], unsafe_allow_html=True)
            elif message.get("type") == "info": st.info(message["content"])
            elif message.get("type") == "warning": st.warning(message["content"])
            elif message.get("type") == "error": st.error(message["content"])
            else: st.markdown(message["content"], unsafe_allow_html=True)


# --- Stage Advancement & Input/Button Logic --- (Placed after messages)

def advance_stage(next_stage):
    logging.info(f"Advancing stage: {STAGE_NAMES.get(st.session_state.stage)} -> {STAGE_NAMES.get(next_stage)}")
    st.session_state.stage = next_stage
    prompt = STAGE_PROMPTS.get(next_stage)
    is_internal = next_stage in [STAGES["PROCESS_INFO_SHOW_DRUGS"], STAGES["SAVE_CONTACT_SHOW_TRIALS"], STAGES["SHOW_TRIALS_NO_CONSENT"], STAGES["FINAL_SUMMARY"]]
    if prompt and not is_internal and next_stage != STAGES["END"]: # Only add prompts for user-facing stages
        msg = {"role": "assistant", "content": prompt}
        if not st.session_state.messages or st.session_state.messages[-1].get("content") != prompt:
             st.session_state.messages.append(msg)

# Input/Button Area Logic
current_stage = st.session_state.stage

# Consent Buttons
if current_stage == STAGES["ASK_CONSENT"]:
    st.write(STAGE_PROMPTS.get(STAGES["ASK_CONSENT"]))
    cols = st.columns([1, 1, 6])
    with cols[0]:
        if st.button("✔️ Yes", key="consent_yes"):
            st.session_state.consent_given = True; st.session_state.messages.append({"role": "user", "content": "A: Yes"})
            advance_stage(STAGES["GET_NAME"]); st.rerun()
    with cols[1]:
         if st.button("❌ No", key="consent_no"):
            st.session_state.consent_given = False; st.session_state.messages.append({"role": "user", "content": "A: No"})
            context_data = {k: st.session_state.user_inputs.get(k, "N/A") for k in ["diagnosis", "stage", "biomarkers", "prior_treatment", "imaging"]}
            context_data["ConsentGiven"] = False; save_user_data(context_data)
            advance_stage(STAGES["SHOW_TRIALS_NO_CONSENT"]); st.rerun()

# Text Input Stages
elif current_stage not in [STAGES["ASK_CONSENT"], STAGES["END"], STAGES["PROCESS_INFO_SHOW_DRUGS"], STAGES["SAVE_CONTACT_SHOW_TRIALS"], STAGES["SHOW_TRIALS_NO_CONSENT"], STAGES["FINAL_SUMMARY"]]:
    prompt_text = STAGE_PROMPTS.get(current_stage, "Enter response...")
    placeholder = re.search(r"\((e\.g\.,.*?)\)", prompt_text)
    placeholder = placeholder.group(1) if placeholder else "Your answer..."
    user_input = st.chat_input(placeholder=placeholder, key=f"chat_input_{current_stage}")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": f"A: {user_input}"})
        next_stage = None
        if current_stage == STAGES["GET_DIAGNOSIS"]: st.session_state.user_inputs['diagnosis'] = user_input.strip(); next_stage = STAGES["GET_STAGE"]
        elif current_stage == STAGES["GET_STAGE"]: st.session_state.user_inputs['stage'] = user_input.strip(); next_stage = STAGES["GET_BIOMARKERS"]
        elif current_stage == STAGES["GET_BIOMARKERS"]: st.session_state.user_inputs['biomarkers'] = user_input.strip(); next_stage = STAGES["GET_PRIOR_TREATMENT"]
        elif current_stage == STAGES["GET_PRIOR_TREATMENT"]: st.session_state.user_inputs['prior_treatment'] = user_input.strip(); next_stage = STAGES["GET_IMAGING"]
        elif current_stage == STAGES["GET_IMAGING"]: st.session_state.user_inputs['imaging'] = user_input.strip(); next_stage = STAGES["PROCESS_INFO_SHOW_DRUGS"]
        elif current_stage == STAGES["GET_NAME"]: st.session_state.user_inputs['name'] = user_input.strip(); next_stage = STAGES["GET_EMAIL"]
        elif current_stage == STAGES["GET_EMAIL"]:
            if "@" not in user_input or "." not in user_input.split('@')[-1]: st.session_state.messages.append({"role": "assistant", "content": "⚠️ Please enter a valid email.", "type":"warning"})
            else: st.session_state.user_inputs['email'] = user_input.strip(); next_stage = STAGES["GET_PHONE"]
        elif current_stage == STAGES["GET_PHONE"]: st.session_state.user_inputs['phone'] = user_input.strip() if user_input.strip() else "N/A"; next_stage = STAGES["SAVE_CONTACT_SHOW_TRIALS"]
        if next_stage is not None: advance_stage(next_stage); st.rerun()

# Processing Stages
elif current_stage == STAGES["PROCESS_INFO_SHOW_DRUGS"]:
    if not st.session_state.get("drug_results_processed", False):
        with st.spinner("Finding relevant drug studies..."):
            user_inputs = st.session_state.user_inputs
            try:
                # Use threshold from session state
                st.session_state.drug_results = find_relevant_drugs_local(
                    df_drugs=df_drugs_processed, drug_embeddings=drug_embeddings_array, model=embedding_model,
                    user_cancer_type_raw=user_inputs.get("diagnosis", ""), user_stage_raw=user_inputs.get("stage", ""),
                    user_biomarkers_raw=user_inputs.get("biomarkers", ""),
                    relevance_threshold=st.session_state.drug_threshold, # Use state value
                    max_results=MAX_DRUGS_TO_DISPLAY
                )
                with st.spinner("Generating summary & interpretations..."):
                    drug_list_summary = generate_drug_summary_llm(user_inputs, st.session_state.drug_results, st.session_state.model_id)
                    st.session_state.messages.append({"role": "assistant", "content": f"**Drug Study Findings:**\n\n{drug_list_summary}"})
                    if st.session_state.drug_results:
                        st.session_state.messages.append({"role": "assistant", "content": f"Details for top {len(st.session_state.drug_results)} studies:", "type": "info"})
                        for i, drug in enumerate(st.session_state.drug_results):
                            interpretation = generate_result_interpretation_llm("drug", drug, user_inputs, st.session_state.model_id)
                            title = f"#{i+1}: {drug.get('Drug Name', 'N/A')} (Rel: {drug.get('semantic_similarity', 0):.2f})"
                            content = f"**Drug:** {drug.get('Drug Name', 'N/A')}\n**Study Cancer Type:** {drug.get('Cancer Type', 'N/A')}\n"
                            content += f"**Outcomes:** OS Tx:`{drug.get('Treatment_OS', 'N/A')}` vs Ctrl:`{drug.get('Control_OS', 'N/A')}` ({drug.get('OS_Improvement (%)', 'N/A')}); PFS Tx:`{drug.get('Treatment_PFS', 'N/A')}` vs Ctrl:`{drug.get('Control_PFS', 'N/A')}` ({drug.get('PFS_Improvement (%)', 'N/A')})\n\n"
                            google_url = f"https://www.google.com/search?q={drug.get('Drug Name', '').replace(' ', '+')}+{drug.get('Cancer Type', '').replace(' ', '+')}+clinical+study"
                            content += f"🔗 [Search Google]({google_url})\n\n**AI Interpretation:**\n> {interpretation}\n"
                            st.session_state.messages.append({"role": "assistant", "content": content, "type": "expander", "title": title, "expanded": i < 1})
                st.session_state.drug_results_processed = True
                advance_stage(STAGES["ASK_CONSENT"]); st.rerun()
            except Exception as e:
                logging.error("Error during drug processing:", exc_info=True); st.error(f"Drug analysis error: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "Could not complete drug analysis.", "type": "error"})
                st.session_state.drug_results_processed = True; advance_stage(STAGES["ASK_CONSENT"]); st.rerun()

elif current_stage == STAGES["SAVE_CONTACT_SHOW_TRIALS"]:
    if st.session_state.consent_given is True and not st.session_state.get("contact_saved", False):
        contact_data = {k: st.session_state.user_inputs.get(k, "N/A") for k in ["name", "email", "phone", "diagnosis", "stage", "biomarkers", "prior_treatment", "imaging"]}
        contact_data["ConsentGiven"] = True
        if save_user_data(contact_data): st.toast("Contact info recorded.", icon="✅"); st.session_state.messages.append({"role": "assistant", "content": "Contact info recorded.", "type": "info"})
        else: st.toast("Failed to save contact info.", icon="⚠️"); st.session_state.messages.append({"role": "assistant", "content": "⚠️ Issue saving contact info.", "type": "warning"})
        st.session_state.contact_saved = True
    advance_stage(STAGES["SHOW_TRIALS_NO_CONSENT"]); st.rerun()

elif current_stage == STAGES["SHOW_TRIALS_NO_CONSENT"]:
    if not st.session_state.get("trial_results_processed", False):
        with st.spinner("Finding relevant clinical trials..."):
            user_inputs = st.session_state.user_inputs
            try:
                # Use threshold from session state
                st.session_state.trial_results = find_relevant_trials_local(
                    df_trials=df_trials_processed, trial_embeddings=trial_embeddings_array, index_to_embedding_index=trial_index_map, model=embedding_model,
                    user_cancer_type_raw=user_inputs.get("diagnosis", ""), user_stage_raw=user_inputs.get("stage", ""),
                    user_biomarkers_raw=user_inputs.get("biomarkers", ""),
                    relevance_threshold=st.session_state.trial_threshold, # Use state value
                    max_results=MAX_TRIALS_TO_DISPLAY
                )
                with st.spinner("Generating summary & interpretations..."):
                    trial_list_summary = generate_trial_summary_llm(user_inputs, st.session_state.trial_results, st.session_state.model_id)
                    st.session_state.messages.append({"role": "assistant", "content": f"**Clinical Trial Findings:**\n\n{trial_list_summary}"})
                    if st.session_state.trial_results:
                        st.session_state.messages.append({"role": "assistant", "content": f"Details for top {len(st.session_state.trial_results)} trials:", "type": "info"})
                        for i, trial in enumerate(st.session_state.trial_results):
                            interpretation = generate_result_interpretation_llm("trial", trial, user_inputs, st.session_state.model_id)
                            title = f"#{i+1}: {trial.get('nct_id', 'N/A')} - {trial.get('status', 'N/A')} (Rel: {trial.get('semantic_similarity', 0):.2f})"
                            content = f"**NCT ID:** {trial.get('nct_id', 'N/A')} | **Title:** {trial.get('title', 'N/A')}\n"
                            content += f"**Status:** `{trial.get('status', 'N/A')}` | **Phase:** `{trial.get('phases', 'N/A')}`\n"
                            content += f"**Conditions:** {trial.get('conditions', 'N/A')} | **Intervention(s):** {trial.get('interventions', 'N/A')}\n\n"
                            if trial.get('url', '#') != '#': content += f"🔗 [View on ClinicalTrials.gov]({trial.get('url')})\n\n"
                            content += f"**AI Interpretation:**\n> {interpretation}\n"
                            st.session_state.messages.append({"role": "assistant", "content": content, "type": "expander", "title": title, "expanded": i < 1})
                st.session_state.trial_results_processed = True
                advance_stage(STAGES["FINAL_SUMMARY"]); st.rerun()
            except Exception as e:
                logging.error("Error during trial processing:", exc_info=True); st.error(f"Trial analysis error: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "Could not complete trial analysis.", "type": "error"})
                st.session_state.trial_results_processed = True; advance_stage(STAGES["FINAL_SUMMARY"]); st.rerun()

elif current_stage == STAGES["FINAL_SUMMARY"]:
    if not st.session_state.get("final_summary_processed", False):
        with st.spinner("Generating final summary..."):
            try:
                final_summary = generate_final_summary_llm(st.session_state.user_inputs, st.session_state.drug_results, st.session_state.trial_results, st.session_state.model_id)
                st.session_state.messages.append({"role": "assistant", "content": f"**Final Summary:**\n\n{final_summary}"})
            except Exception as e:
                 logging.error("Error generating final summary", exc_info=True); st.error(f"Final summary generation error: {e}")
                 st.session_state.messages.append({"role": "assistant", "content": "Could not generate final summary.", "type": "error"})
        st.session_state.final_summary_processed = True
        advance_stage(STAGES["END"]); st.rerun()

# End Stage - Display final message and restart button
elif current_stage == STAGES["END"]:
    end_prompt_text = STAGE_PROMPTS.get(STAGES["END"])
    # Ensure end message is the last one
    if not st.session_state.messages or st.session_state.messages[-1].get("content") != end_prompt_text:
        st.session_state.messages.append({"role": "assistant", "content": end_prompt_text, "type": "info"})
        st.rerun()
    else:
        # Display restart button only after the end message is shown
        st.info("Use the sidebar or button below to start a new exploration.")
        if st.button("🔄 Start New Exploration", key="restart_main"):
            logging.info("Restarting session from main button.")
            keys_to_clear = [k for k in st.session_state.keys() if k not in ['model_id', 'model_select_widget']]
            for key in keys_to_clear: del st.session_state[key]
            initialize_session(); st.rerun()