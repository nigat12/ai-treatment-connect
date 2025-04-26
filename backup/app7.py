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
import textwrap

# --- Configuration & Setup ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# CRITICAL: Check if Groq API key is loaded BEFORE proceeding
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    # Use st.error for user visibility in the app itself
    st.error("🚨 FATAL ERROR: GROQ_API_KEY not found. Please set it in your .env file or environment variables and restart.")
    logging.critical("GROQ_API_KEY not found. Application cannot proceed.")
    st.stop() # Halt execution

# --- API Endpoints ---
FDA_API_BASE_URL = "https://api.fda.gov/drug/label.json"
CTGOV_API_V2_BASE_URL = "https://clinicaltrials.gov/api/v2/studies" # Endpoint for studies search

# --- File Paths (WARNING: Not Secure for Production PII) ---
# ** SECURITY WARNING: Saving PII to a local text file is INSECURE and NOT SUITABLE
# ** for production environments handling real patient data. Use a secure database
# ** and appropriate encryption/access controls in a real application.
USER_DATA_FILE = "user_contact_data_demo.txt" # For demo purposes ONLY

# --- Constants for UI and Logic ---
ASSISTANT_AVATAR = "🧑‍⚕️"
USER_AVATAR = "👤"
ASSISTANT_NAME = "Assistant"
USER_NAME = "You"
FDA_RESULT_LIMIT = 15 # Increased limit for FDA results per tier
FDA_MIN_RESULTS_FOR_NEXT_TIER = 3 # Try next tier if fewer than this found in current tier
CT_REQUEST_LIMIT = 50 # Request more trials initially from API
CT_DISPLAY_LIMIT = 15 # Show top N trials to the user
CHAT_CONTAINER_HEIGHT = 600 # Adjust height for chat scroll area in pixels

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
    "PROCESS_INFO_SHOW_DRUGS": 6, # New stage for processing and showing drugs
    "ASK_CONSENT": 7,
    "GET_NAME": 8,
    "GET_EMAIL": 9,
    "GET_PHONE": 10,
    "SAVE_CONTACT_SHOW_TRIALS": 11, # New stage to save contact and then show trials
    "SHOW_TRIALS_NO_CONSENT": 12, # Stage to show trials if no consent was given
    "FINAL_SUMMARY": 13, # New stage for the final summary
    "END": 14
}

STAGE_PROMPTS = {
    STAGES["INIT"]: "Hi there! I'm an AI assistant designed to help you explore information about cancer treatments using public data. I work alongside your physician and **do not provide medical advice.** Let's start by gathering some information. Please answer the questions as accurately as possible.\n\nFirst: 👇",
    STAGES["GET_DIAGNOSIS"]: "Q: Kindly share the specific cancer diagnosis? (e.g., Non-small cell lung cancer, Metastatic Breast Cancer)",
    STAGES["GET_STAGE"]: "Q: Any details on the stage, progression, or spread? (e.g., Stage IV, metastatic to bones, locally advanced)",
    STAGES["GET_BIOMARKERS"]: "Q: Any known biomarker details? (e.g., EGFR Exon 19 deletion, HER2 positive, PD-L1 > 50%). Please list them or type 'None'/'Unknown'.",
    STAGES["GET_PRIOR_TREATMENT"]: "Q: What treatments have been received to date? (e.g., Chemotherapy (Carboplatin/Pemetrexed), Surgery, Immunotherapy (Pembrolizumab), Radiation)",
    STAGES["GET_IMAGING"]: "Q: Are there any recent imaging results (CT, PET, MRI) showing changes or current status? (e.g., Recent CT showed stable disease, PET scan showed progression in liver)",
    STAGES["ASK_CONSENT"]: "Q: To potentially share more tailored information or allow for follow-up, may we collect your contact details (Name, Email, Phone)?\n\n*_(For this demo, data is saved to a local text file - **this is not secure for real patient data**)._*",
    STAGES["GET_NAME"]: "Q: Please enter your First and Last Name:",
    STAGES["GET_EMAIL"]: "Q: Please enter your Email Address:",
    STAGES["GET_PHONE"]: "Q: Please enter your Phone Number (optional, press Enter if skipping):",
    STAGES["FINAL_SUMMARY"]: "Generating a final summary based on the gathered information...", # This prompt is immediately followed by the LLM summary
    STAGES["END"]: "You have reached the end of this exploration session. Feel free to restart if you want to explore a different scenario. Remember to discuss this information with your oncologist."
}

# Helper to get stage name for debug display
STAGE_NAMES = {v: k for k, v in STAGES.items()}

# --- Helper Functions ---

def clean_text(text, max_len=None):
    """Basic text cleaning: remove HTML, normalize whitespace, handle None, optional truncation."""
    if text is None: return "N/A"
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text) # Remove HTML tags
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    if max_len and len(text) > max_len:
        # Try to truncate nicely at a word boundary or sentence boundary
        truncated = text[:max_len]
        last_space = truncated.rfind(' ')
        last_sentence_end = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
        if last_sentence_end > last_space and last_sentence_end > max_len * 0.8: # Prioritize sentence end if close to max_len
             truncated = truncated[:last_sentence_end + 1]
        elif last_space > max_len * 0.7: # Otherwise truncate at word boundary if not too short
             truncated = truncated[:last_space]

        if len(truncated) < max_len * 0.5 and len(truncated) < 50: # Avoid overly aggressive truncation
             truncated = text[:max_len] # Fallback to simple character truncation

        text = truncated + "..." if truncated != text else text

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

def get_llm_client(model_id):
    """Initializes and returns the Groq client."""
    try:
        return Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        logging.critical(f"Failed to initialize Groq client: {e}", exc_info=True)
        st.error("FATAL ERROR: Could not initialize AI client. Please check configuration.")
        st.stop() # Halt app if client can't be initialized

# --- Stage Advancement Function ---
def advance_stage(next_stage):
    """Helper to update stage and add assistant prompt to messages."""
    logging.info(f"Advancing stage from {STAGE_NAMES.get(st.session_state.stage, 'Unknown')} to {STAGE_NAMES.get(next_stage, 'Unknown')}")
    st.session_state.stage = next_stage
    prompt = STAGE_PROMPTS.get(next_stage)
    # Add prompt only if it's new and not a processing/end stage trigger
    if prompt and next_stage not in [STAGES["PROCESS_INFO_SHOW_DRUGS"], STAGES["SAVE_CONTACT_SHOW_TRIALS"], STAGES["SHOW_TRIALS_NO_CONSENT"], STAGES["FINAL_SUMMARY"], STAGES["END"]]:
        msg = {"role": "assistant", "content": prompt}
        if next_stage == STAGES["ASK_CONSENT"]: msg["type"] = "buttons" # Indicate this message requires button UI
        st.session_state.messages.append(msg)


# --- LLM Generation Functions ---

def refine_fda_search_strategy_with_llm(diagnosis, stage_info, markers, model_id):
    """Uses LLM to create a prioritized, tiered search strategy for the FDA API."""
    logging.info(f"[LLM Call - Strategy] Refining FDA search strategy for: D='{diagnosis}', S='{stage_info}', M='{markers}', Model={model_id}")

    # Default strategy if LLM fails
    default_strategy = {
        "primary_search": [term.strip('",.') for term in f"{diagnosis} {markers}".split() if len(term) > 2],
        "secondary_search": [term.strip('",.') for term in diagnosis.split() if len(term) > 2] or ["cancer"],
        "tertiary_search": [], # Keep tertiary empty by default
        "fallback_search": ["cancer"]
    }
    if not diagnosis: diagnosis = "cancer" # Ensure diagnosis is not empty for fallback logic

    client = get_llm_client(model_id)

    prompt = f"""
    Analyze the following cancer information provided by a user:
    Diagnosis: "{diagnosis}"
    Stage/Progression: "{stage_info}"
    Biomarkers: "{markers}"

    Create a prioritized, tiered JSON search strategy focusing on the 'indications_and_usage' field for the FDA drug label database. The strategy should list terms relevant to the user's condition, prioritizing specific terms first.

    Output ONLY the JSON object like {{"primary_search": [], "secondary_search": [], "tertiary_search": [], "fallback_search": []}}.

    Rules for generating search terms:
    1.  Identify the main cancer type and any significant subtypes.
    2.  Include specific biomarkers if provided and relevant.
    3.  Add "metastatic" if stage/progression indicates spread (Stage IV, metastatic, advanced).
    4.  Normalize terms (e.g., "nsclc" -> "non-small cell lung cancer").
    5.  Keep terms concise and relevant to drug indications.
    6.  Prioritize terms: primary should be most specific, secondary less so, tertiary even less, and fallback very general.
    7.  Terms within a list will be combined with AND in the search query for that tier.
    8.  Ensure terms are relevant keywords, not full sentences.

    Example 1 (NSCLC Stage IV EGFR positive):
    {{"primary_search": ["non-small cell lung cancer", "EGFR", "metastatic"], "secondary_search": ["lung cancer", "EGFR"], "tertiary_search": ["non-small cell lung cancer", "stage IV"], "fallback_search": ["lung cancer"]}}

    Example 2 (Metastatic Breast Cancer, hormone receptor positive):
    {{"primary_search": ["metastatic breast cancer", "hormone receptor positive"], "secondary_search": ["breast cancer", "metastatic"], "tertiary_search": ["hormone receptor positive cancer"], "fallback_search": ["breast cancer"]}}

    Example 3 (Melanoma, unknown stage/markers):
    {{"primary_search": ["melanoma"], "secondary_search": ["skin cancer"], "tertiary_search": [], "fallback_search": ["cancer"]}}

    Example 4 (Leukemia, Specific Type):
    {{"primary_search": ["acute myeloid leukemia"], "secondary_search": ["leukemia"], "tertiary_search": [], "fallback_search": ["cancer"]}}

    JSON Output:
    """
    strategy = default_strategy # Initialize with default
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert medical terminology assistant designing FDA search strategies. Output ONLY the valid JSON strategy object. Ensure terms are concise and relevant to drug indications."},
                {"role": "user", "content": prompt}
            ],
            model=model_id,
            temperature=0.1,
            max_tokens=400, # Adjusted max tokens
            response_format={"type": "json_object"} # Request JSON object output
        )
        response_content = chat_completion.choices[0].message.content
        logging.info(f"LLM raw strategy response: {response_content}")
        try:
            parsed_json = json.loads(response_content)
            required_keys = ["primary_search", "secondary_search", "tertiary_search", "fallback_search"]
            if isinstance(parsed_json, dict) and all(key in parsed_json for key in required_keys):
                valid_strategy = True; temp_strategy = {}
                for key in required_keys:
                    if isinstance(parsed_json[key], list) and all(isinstance(term, str) for term in parsed_json[key]):
                        # Clean and remove empty terms
                        temp_strategy[key] = [t.strip() for t in parsed_json[key] if t.strip()]
                    else:
                        valid_strategy = False; logging.warning(f"LLM strategy JSON invalid: key '{key}' invalid type or contents. Val: {parsed_json.get(key)}"); temp_strategy[key] = default_strategy.get(key, [])
                strategy = temp_strategy; logging.info(f"LLM FDA search strategy {'fully' if valid_strategy else 'partially'} parsed: {strategy}")
            else:
                logging.warning(f"LLM strategy JSON structure invalid. Using default. Resp: {response_content}"); strategy = default_strategy
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logging.error(f"Error parsing LLM JSON strategy: {e}. Resp: {response_content}", exc_info=True)
            strategy = default_strategy
    except (RateLimitError, APIError) as e:
        logging.error(f"Groq API error (strategy): {e}", exc_info=True)
        st.warning("API error generating FDA search strategy. Using a basic strategy.")
        strategy = default_strategy
    except Exception as e:
        logging.error(f"Unexpected LLM FDA strategy error: {e}", exc_info=True)
        st.warning("Error generating FDA search strategy.")
        strategy = default_strategy

    # Final check to ensure all keys are present and are lists of strings
    for key in ["primary_search", "secondary_search", "tertiary_search", "fallback_search"]:
        if key not in strategy or not isinstance(strategy[key], list):
            strategy[key] = default_strategy.get(key, [])
        strategy[key] = [term for term in strategy[key] if isinstance(term, str) and term.strip()] # Ensure elements are non-empty strings

    logging.info(f"Final refined FDA strategy: {strategy}")
    return strategy


def generate_ctgov_keywords_llm(diagnosis, stage_info, biomarkers, prior_treatment, model_id):
    """Uses LLM to generate relevant keywords and phrases for ClinicalTrials.gov query.term."""
    logging.info(f"[LLM Call - CT Keywords] Generating CT.gov keywords for: D='{diagnosis}', S='{stage_info}', M='{biomarkers}', Model={model_id}")

    client = get_llm_client(model_id)
    prompt = f"""
    Analyze the following cancer information to generate a list of relevant keywords and phrases for searching the ClinicalTrials.gov database using its 'query.term' parameter. This parameter searches across various fields including title, summary, conditions, etc. The API often implicitly ANDs words in this field, and quotes can be used for exact phrases.

    User Information:
    Diagnosis: "{diagnosis}"
    Stage/Progression: "{stage_info}"
    Biomarkers: "{biomarkers}"
    Prior Treatment: "{prior_treatment}"

    Generate a list of keywords and phrases that are most likely to find relevant clinical trials. Include:
    1. The primary cancer type (e.g., "lung cancer", "breast cancer"). Quote multi-word types.
    2. Relevant specific subtypes (e.g., "non-small cell", "adenocarcinoma"). Quote multi-word subtypes.
    3. Significant biomarkers as quoted phrases or single terms (e.g., "EGFR mutation", "HER2 positive", "PD-L1").
    4. Relevant status indicators like "recruiting", "active", "not yet recruiting".
    5. Relevant phase indicators like "phase 2", "phase 3", "phase 4". Quote phases.
    6. Study type indicators like "interventional study", "clinical trial". Quote multi-word types.
    7. "metastatic" or related terms if the stage indicates spread.

    Exclude:
    - Very general terms unless the input is extremely vague ("cancer" as a last resort).
    - Treatment types already received (from Prior Treatment).
    - Full sentences or questions.
    - Terms that are unlikely to appear in trial titles, summaries, or conditions.

    Prioritize keywords directly related to the diagnosis and biomarkers. Include the status, phase, and type keywords to help find relevant trials by those criteria *via keyword matching*, rather than strict filtering.

    Format the output as a space-separated string of keywords and quoted phrases. Example: "non-small cell lung cancer" "EGFR mutation" metastatic recruiting "phase 3" "interventional study". Do NOT include commas.

    Keywords for query.term:
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert medical terminology assistant generating space-separated keywords and quoted phrases for clinical trial search (query.term)."},
                {"role": "user", "content": prompt}
            ],
            model=model_id, temperature=0.3, max_tokens=300
        )
        keywords_str = clean_text(chat_completion.choices[0].message.content)
        # The LLM is asked for space-separated keywords/phrases directly,
        # so we just need to ensure it's not empty or contains unwanted characters.
        # A simple split by whitespace should work, preserving quotes if LLM adds them.
        keywords_list = [term.strip() for term in keywords_str.split() if term.strip() and len(term.strip()) > 1]
        # Ensure basic keywords are present if LLM fails or is too specific
        if not keywords_list:
             fallback_keywords = [diagnosis]
             if biomarkers and biomarkers.lower() not in ['none', 'unknown', 'n/a', '']:
                fallback_keywords.extend(re.split(r'[,\/\s]+', biomarkers))
             fallback_keywords.extend(['recruiting', 'phase 2', 'phase 3', 'phase 4', 'interventional'])
             keywords_list = [f'"{term.strip()}"' if ' ' in term.strip() else term.strip() for term in fallback_keywords if term.strip() and len(term.strip()) > 1] # Quote multi-word fallbacks

        logging.info(f"LLM generated CT.gov keywords: {keywords_list}")
        return keywords_list
    except (RateLimitError, APIError) as e:
        logging.error(f"Groq API error (CT keywords): {e}", exc_info=True)
        st.warning("API error generating ClinicalTrials.gov keywords. Using basic terms.")
        # Fallback to basic terms if LLM fails
        fallback_keywords = [diagnosis]
        if biomarkers and biomarkers.lower() not in ['none', 'unknown', 'n/a', '']:
            fallback_keywords.extend(re.split(r'[,\/\s]+', biomarkers))
        fallback_keywords.extend(['recruiting', 'phase 2', 'phase 3', 'phase 4', 'interventional'])
        # Quote multi-word fallbacks
        return [f'"{term.strip()}"' if ' ' in term.strip() else term.strip() for term in fallback_keywords if term.strip() and len(term.strip()) > 1]
    except Exception as e:
        logging.error(f"Unexpected LLM CT keywords error: {e}", exc_info=True)
        st.warning("Error generating ClinicalTrials.gov keywords. Using basic terms.")
        # Fallback to basic terms
        fallback_keywords = [diagnosis]
        if biomarkers and biomarkers.lower() not in ['none', 'unknown', 'n/a', '']:
            fallback_keywords.extend(re.split(r'[,\/\s]+', biomarkers))
        fallback_keywords.extend(['recruiting', 'phase 2', 'phase 3', 'phase 4', 'interventional'])
        # Quote multi-word fallbacks
        return [f'"{term.strip()}"' if ' ' in term.strip() else term.strip() for term in fallback_keywords if term.strip() and len(term.strip()) > 1]


def generate_drug_summary_llm(diagnosis, markers, fda_drugs, model_id):
    """Generates a concise drug summary from the list of found drugs using LLM."""
    logging.info(f"[LLM Call - Drug Summary] Generating drug summary for {len(fda_drugs)} drugs.")
    if not fda_drugs:
        return f"Based on searches of FDA drug labels related to '{diagnosis}' and markers like '{markers}', no specific drug matches were found based on the search strategy used."

    # Limit drugs sent to LLM prompt to manage token count
    drug_list_str = "\n".join([
        f"- {d.get('brand_name', 'N/A')} ({d.get('generic_name', 'N/A')}): Snippet: '{clean_text(d.get('indication_snippet', 'N/A'), 150)}'"
        for d in fda_drugs[:10]
    ])
    if len(fda_drugs) > 10:
        drug_list_str += f"\n- ...and {len(fda_drugs) - 10} more."


    client = get_llm_client(model_id)
    prompt = f"""
    The user is exploring potential treatments based on their cancer information.
    Diagnosis: "{diagnosis}"
    Biomarkers: "{markers}"
    A search found {len(fda_drugs)} FDA-approved drugs with labels containing relevant indications. Here are some of them:
    {drug_list_str}

    Summarize these findings concisely for the user.
    Start with a sentence stating the search identified potential drugs based on their diagnosis and relevant markers.
    Mention the *number* of drugs found.
    List the brand names (if available) and generic names of the *first few* drugs found (e.g., "including Drug A (Generic B), Drug C (Generic D), etc.").
    Do NOT include any links or external URLs in the summary.
    Keep it to 2-4 sentences.

    Summary:
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI assistant summarizing drug information. Provide a concise summary of the listed drugs based on the user's context. Do not include links. Include the total count."},
                {"role": "user", "content": prompt}
            ],
            model=model_id, temperature=0.2, max_tokens=250
        )
        summary = clean_text(chat_completion.choices[0].message.content)
        # Ensure count is mentioned if LLM missed it
        if str(len(fda_drugs)) not in summary:
             summary = f"Found {len(fda_drugs)} potentially relevant drug(s). " + summary

        logging.info(f"LLM drug summary generated: {summary}")
        return summary
    except (RateLimitError, APIError) as e:
        logging.error(f"Groq API error (drug summary): {e}", exc_info=True)
        return f"An error occurred while generating the drug summary. Found {len(fda_drugs)} potential drug(s)."
    except Exception as e:
        logging.error(f"Unexpected LLM drug summary error: {e}", exc_info=True)
        return f"An error occurred while generating the drug summary. Found {len(fda_drugs)} potential drug(s)."


def generate_trial_summary_llm(diagnosis, markers, stage_info, clinical_trials, model_id):
    """Generates a concise clinical trial summary, including count and status, using LLM."""
    logging.info(f"[LLM Call - Trial Summary] Generating trial summary for {len(clinical_trials)} trials.")

    num_found = len(clinical_trials)
    if num_found == 0:
        return f"Based on keyword searches for '{diagnosis}', stage '{stage_info}', and markers '{markers}', no matching clinical trials were found in the public database."

    # Identify active/recruiting trials for mention in summary from the results
    active_statuses = ['recruiting', 'not yet recruiting', 'enrolling by invitation']
    num_active = len([t for t in clinical_trials if t.get('status', '').lower() in active_statuses])


    # Limit trials sent to LLM prompt to manage token count
    trial_titles_status_str = "\n".join([
        f"- '{clean_text(t.get('title', 'N/A'), 100)}' (Status: {t.get('status', 'N/A')})"
        for t in clinical_trials[:min(num_found, 5)]
    ])
    if num_found > 5:
        trial_titles_status_str += f"\n- ...and {num_found - 5} more."


    client = get_llm_client(model_id)
    prompt = f"""
    The user is exploring potential clinical trials based on their cancer information.
    Diagnosis: "{diagnosis}"
    Stage/Progression: "{stage_info}"
    Biomarkers: "{markers}"
    A keyword search on ClinicalTrials.gov found {num_found} studies. Approximately {num_active} of these are listed with 'Recruiting' or similar active statuses. Here are a few examples from the results:
    {trial_titles_status_str}

    Summarize these findings concisely for the user.
    Start by stating the total number of trials found based on keywords related to their diagnosis and context.
    Mention the approximate number of trials from the results that are currently 'Recruiting' or in similar active statuses.
    Clarify that the search was based on keywords across trial fields, not strict filters.
    Suggest reviewing the detailed list below for more information.
    Do NOT include any links or external URLs in the summary.
    Keep it to 2-4 sentences.

    Summary:
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI assistant summarizing clinical trial information found via keyword search. Provide a concise summary of the listed trials based on the user's context. Include counts and mention recruitment status from the results. Clarify the search method. Do not include links."},
                {"role": "user", "content": prompt}
            ],
            model=model_id, temperature=0.2, max_tokens=300
        )
        summary = clean_text(chat_completion.choices[0].message.content)
        # Ensure counts are mentioned if LLM missed them
        if str(num_found) not in summary or str(num_active) not in summary:
             summary = f"Found {num_found} potential clinical trial(s) with approximately {num_active} currently active/recruiting. " + summary

        logging.info(f"LLM trial summary generated: {summary}")
        return summary
    except (RateLimitError, APIError) as e:
        logging.error(f"Groq API error (trial summary): {e}", exc_info=True)
        return f"An error occurred while generating the clinical trial summary. Found {num_found} potential trial(s) with approximately {num_active} active/recruiting."
    except Exception as e:
        logging.error(f"Unexpected LLM trial summary error: {e}", exc_info=True)
        return f"An error occurred while generating the clinical trial summary. Found {num_found} potential trial(s) with approximately {num_active} active/recruiting."


def generate_final_summary_llm(user_inputs, drug_results, trial_results, model_id):
    """Generates a final comprehensive summary using LLM, combining user context, drugs, and trials."""
    logging.info(f"[LLM Call - Final Summary] Generating final summary.")

    diagnosis = user_inputs.get("diagnosis", "a cancer condition")
    stage_info = user_inputs.get("stage", "stage information unknown")
    biomarkers = user_inputs.get("biomarkers", "unknown biomarkers")
    prior_treatment = user_inputs.get("prior_treatment", "unknown prior treatments")
    imaging = user_inputs.get("imaging", "unknown imaging results")

    drug_count = len(drug_results)
    trial_count = len(trial_results)

    drug_list_summary = "No specific drugs were identified in the FDA label search."
    if drug_count > 0:
        drug_names = [d.get('brand_name', 'N/A') for d in drug_results[:5]] # Limit names sent to LLM
        drug_list_summary = f"The search of FDA drug labels identified {drug_count} potentially relevant drug(s), including: {', '.join(drug_names)}."
        if drug_count > 5: drug_list_summary += " and others."

    trial_list_summary = "No clinical trials were found in the ClinicalTrials.gov search."
    if trial_count > 0:
        active_statuses = ['recruiting', 'not yet recruiting', 'enrolling by invitation']
        num_active = len([t for t in trial_results if t.get('status', '').lower() in active_statuses])
        trial_list_summary = f"The search on ClinicalTrials.gov found {trial_count} potential studies using keyword criteria. Approximately {num_active} of these were listed with 'Recruiting' or similar active statuses in the results."

    client = get_llm_client(model_id)
    prompt = f"""
    Based on the user's provided information (Diagnosis: {diagnosis}, Stage/Progression: {stage_info}, Biomarkers: {biomarkers}):
    - Drugs identified from FDA labels: {drug_list_summary}
    - Clinical trials found from ClinicalTrials.gov (keyword search): {trial_list_summary}

    Synthesize this information into a brief, concluding summary for the user.
    1. Acknowledge the exploration based on their input regarding diagnosis, stage, and biomarkers.
    2. Briefly mention the drug findings count and the clinical trial findings count/active count.
    3. Reiterate strongly that this information is for exploration only and is NOT medical advice.
    4. Emphasize the critical need to discuss these findings and their individual situation with their qualified oncologist or healthcare provider for personalized guidance and decision-making.
    5. Keep it concise, 3-5 sentences. Do not include links.

    Final Summary:
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI assistant providing a final summary of gathered information. Synthesize the provided data concisely and professionally. Include counts and a strong disclaimer about not being medical advice and consulting an oncologist."},
                {"role": "user", "content": prompt}
            ],
            model=model_id, temperature=0.3, max_tokens=400
        )
        summary = clean_text(chat_completion.choices[0].message.content)
        logging.info(f"LLM final summary generated: {summary}")
        return summary
    except (RateLimitError, APIError) as e:
        logging.error(f"Groq API error (final summary): {e}", exc_info=True)
        return "An error occurred while generating the final summary. Please review the drug and trial results above and discuss them with your oncologist. This information is not medical advice."
    except Exception as e:
        logging.error(f"Unexpected LLM final summary error: {e}", exc_info=True)
        return "An error occurred while generating the final summary. Please review the drug and trial results above and discuss them with your oncologist. This information is not medical advice."


# --- API Functions ---

def search_fda_drugs_for_condition_and_markers(search_strategy, limit=FDA_RESULT_LIMIT, min_results_per_tier=FDA_MIN_RESULTS_FOR_NEXT_TIER):
    """Searches FDA drug labels via openFDA API using a tiered search strategy."""
    logging.info(f"[API Call - FDA Tiered] Executing FDA search strategy: {search_strategy}")
    final_results = []
    search_tier_used = "None"
    seen_generic_names = set() # Use generic name for deduplication
    search_tiers = ["primary_search", "secondary_search", "tertiary_search", "fallback_search"]

    for tier_key in search_tiers:
        search_terms = search_strategy.get(tier_key, [])
        if not search_terms:
            logging.info(f"Skipping FDA tier '{tier_key}': no terms provided by strategy.")
            continue

        logging.info(f"Attempting FDA search with tier '{tier_key}': {search_terms}")
        query_parts = []
        for term in search_terms:
            term_cleaned = term.strip()
            if not term_cleaned: continue
            # API quirk: AND is default, phrase searching uses quotes
            if ' ' in term_cleaned: query_parts.append(f'"{term_cleaned}"') # Quote phrases
            else: query_parts.append(term_cleaned)
        if not query_parts:
            logging.warning(f"No valid query parts generated for FDA tier '{tier_key}'. Skipping tier.")
            continue

        # Search within 'indications_and_usage' field using 'AND' between terms
        search_query = f'indications_and_usage:({" AND ".join(query_parts)})'
        logging.info(f"Constructed FDA API query for tier '{tier_key}': {search_query}")

        params = {'search': search_query, 'limit': limit}
        current_tier_new_results = []

        try:
            response = requests.get(FDA_API_BASE_URL, params=params, timeout=20)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            for item in data.get('results', []):
                openfda_data = item.get('openfda', {})
                # Handle potential list format for names and get first element
                brand_name = (openfda_data.get('brand_name') or ['N/A'])[0]
                generic_name = (openfda_data.get('generic_name') or ['N/A'])[0]
                # indications_and_usage is also often a list
                indication_list = item.get('indications_and_usage', ['N/A'])
                indication = indication_list[0] if indication_list else 'N/A'
                indication_snippet = clean_text(indication, max_len=400) # Longer snippet for detail

                # Construct a Google search URL as direct label URLs are unreliable and change frequently
                # Search for "brand_name generic_name FDA prescribing information"
                google_search_query = f"{clean_text(brand_name)} {clean_text(generic_name)} FDA prescribing information"
                label_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(google_search_query)}"

                generic_name_cleaned = clean_text(generic_name)
                # Deduplicate results based on generic name to avoid showing same drug multiple times
                if generic_name_cleaned != "N/A" and generic_name_cleaned not in seen_generic_names:
                     current_tier_new_results.append({
                        "brand_name": clean_text(brand_name),
                        "generic_name": generic_name_cleaned,
                        "indication_snippet": indication_snippet,
                        "url": label_url # This URL goes in the expander, not main text
                     })
                     seen_generic_names.add(generic_name_cleaned)
                     if len(final_results) + len(current_tier_new_results) >= limit:
                         logging.info(f"Reached total limit ({limit}) with tier '{tier_key}'. Stopping.")
                         break # Stop adding results if we hit the overall limit

            final_results.extend(current_tier_new_results)
            logging.info(f"FDA Tier '{tier_key}' search returned {len(current_tier_new_results)} new, unique results. Total unique results: {len(final_results)}")

            # Check if we have enough results OR if we already tried the last tier
            if len(final_results) >= min_results_per_tier or tier_key == search_tiers[-1]:
                search_tier_used = tier_key
                logging.info(f"Met minimum FDA results ({min_results_per_tier} in total) or finished tiers with tier '{tier_key}'. Total unique: {len(final_results)}. Stopping tiered search.")
                break # Stop trying lower priority tiers

        except requests.exceptions.Timeout:
            logging.error(f"FDA API timeout tier '{tier_key}'.")
            st.warning(f"Drug search timed out (tier '{tier_key}'). Results may be incomplete.")
        except requests.exceptions.HTTPError as e:
            logging.error(f"FDA API HTTP error tier '{tier_key}': {e.response.status_code} - {e.response.text}")
            if e.response.status_code == 404: logging.info(f"FDA Tier '{tier_key}' returned no results (404).")
            elif e.response.status_code == 429: st.warning(f"Drug search rate limit hit (tier '{tier_key}'). Please wait a moment before retrying.")
            else: st.warning(f"Drug search failed (tier '{tier_key}', HTTP Error {e.response.status_code}).")
        except requests.exceptions.RequestException as e:
            logging.error(f"FDA API request failed tier '{tier_key}': {e}", exc_info=True)
            st.warning("Drug search network issue.")
        except json.JSONDecodeError as e:
            logging.error(f"FDA API JSON decode error tier '{tier_key}': {e}", exc_info=True)
            st.warning("Invalid response received from drug database.")
        except Exception as e:
            logging.error(f"Unexpected FDA processing error tier '{tier_key}': {e}", exc_info=True)
            st.warning("Unexpected error processing drug information.")

    logging.info(f"FDA Tiered Search finished. Used up to tier '{search_tier_used}'. Final unique results: {len(final_results)}")
    return final_results[:limit] # Return up to the total limit


def search_clinical_trials_with_keywords(diagnosis, stage_info, biomarkers, prior_treatment, model_id, limit=CT_REQUEST_LIMIT):
    """
    Generates keywords using LLM and searches ClinicalTrials.gov V2 API using only query.term.
    """
    logging.info(f"[API Call - CT.gov Keyword] Searching ClinicalTrials.gov for: D='{diagnosis}', S='{stage_info}', M='{biomarkers}'")
    results = []

    # --- Use LLM to generate keywords for query.term ---
    # Call the new LLM function
    # keyword_list = generate_ctgov_keywords_llm(diagnosis, stage_info, biomarkers, prior_treatment, model_id)

    # if not keyword_list:
    #     logging.warning("LLM failed to generate keywords or keywords list is empty. Cannot perform CT.gov search.")
    #     st.warning("Could not generate search terms for ClinicalTrials.gov based on your input.")
    #     return results # Return empty if no keywords generated

    # # Join keywords with spaces for query.term - API handles implicit ANDing of simple terms
    # # LLM is prompted to handle quoting for multi-word phrases
    # query_term_str = " ".join(keyword_list)
    
    query_term_str = diagnosis + " or " + stage_info + " or " + biomarkers

    logging.info(f"Constructed ClinicalTrials.gov query.term: '{query_term_str}' (from LLM keywords)")

    # Fields to request from the API - Use documented V2 fields
    fields_to_request = [
        "NCTId", # IdentificationModule
        "BriefTitle", # Study
        "OverallStatus", # StatusModule
        "Phase", # DesignModule
        "Condition", # ConditionsModule
        "BriefSummary", # DescriptionModule
        "StudyType", # DesignModule
        "InterventionList", # InterventionModule (for names)
        "EligibilityCriteria", # EligibilityModule (for snippet)
        "LocationList", # ContactsLocationsModule (for snippet)
    ]

    params = {
        'query.term': query_term_str, # Keywords from LLM
        # Removed query.filter entirely based on testing issues
        'pageSize': limit, # Request more than displayed to find best matches
        'format': 'json',
        'fields': ",".join(fields_to_request)
        # Sorting by relevance is default for term queries
    }
    logging.info(f"ClinicalTrials.gov Request Params: {params}")

    try:
        response = requests.get(CTGOV_API_V2_BASE_URL, params=params, timeout=30)

        # Handle specific non-200 codes more robustly
        if response.status_code == 400:
            logging.error(f"CT.gov API Error (400 Bad Request) for Query: term='{query_term_str}'")
            try: error_data = response.json(); logging.error(f"API Error Details: {error_data.get('message', error_data)}")
            except json.JSONDecodeError: logging.error("Could not decode error response body for 400 error.")
            st.warning("Clinical trial search failed (Bad Request). Please try again or simplify your input.")
            return [] # Return empty on error
        elif response.status_code == 404:
            logging.info(f"No clinical trials found (404) for Query: term='{query_term_str}'")
            return [] # No results found
        elif response.status_code == 500:
             logging.error(f"CT.gov API Error (500 Internal Server Error) for Query: term='{query_term_str}'")
             try: error_data = response.json(); logging.error(f"API Error Details: {error_data.get('message', error_data)}")
             except json.JSONDecodeError: logging.error("Could not decode error response body for 500 error.")
             st.warning("Clinical trial search failed (Internal Server Error). Please try again or simplify your input.")
             return [] # Return empty on error

        response.raise_for_status() # Raise for other errors (>=400) not specifically handled
        data = response.json()
        processed_results = []

        studies = data.get('studies', [])
        logging.info(f"CT.gov API returned {len(studies)} studies raw.")

        for study in studies:
            # Extract data based on fields requested and V2 structure
            protocol = study.get('protocolSection', {})

            # IdentificationModule
            ident_module = protocol.get('identificationModule', {})
            nct_id = ident_module.get('nctId', 'N/A')

            # StatusModule
            status_module = protocol.get('statusModule', {})
            status = clean_text(status_module.get('overallStatus', 'N/A'))

            # DesignModule
            design_module = protocol.get('designModule', {})
            phases_list = design_module.get('phases', ['N/A'])
            phases = ', '.join(phases_list)
            study_type = clean_text(design_module.get('studyType', 'N/A'))

            # ConditionsModule
            cond_module = protocol.get('conditionsModule', {})
            conditions = clean_text(', '.join(cond_module.get('conditionList', {}).get('condition', ['N/A'])))

            # DescriptionModule
            desc_module = protocol.get('descriptionModule', {})
            brief_summary = clean_text(desc_module.get('briefSummary', 'N/A'), max_len=500) # Longer snippet for summary

            # InterventionModule (InterventionList)
            int_module = protocol.get('interventionModule', {})
            interventions_list = int_module.get('interventionList', [])
            intervention_names = [item.get('interventionName', 'N/A') for item in interventions_list]
            interventions_str = clean_text(', '.join(intervention_names), max_len=300)

            # EligibilityModule (EligibilityCriteria)
            elig_module = protocol.get('eligibilityModule', {})
            eligibility_criteria = clean_text(elig_module.get('eligibilityCriteria', 'N/A'), max_len=400) # Snippet

            # ContactsLocationsModule (LocationList)
            loc_module = protocol.get('contactsLocationsModule', {})
            locations_list = loc_module.get('locationList', [])
            location_summaries = []
            for loc in locations_list:
                 facility = loc.get('location', {}).get('facility', 'N/A')
                 city = loc.get('location', {}).get('city', 'N/A')
                 state = loc.get('location', {}).get('state', 'N/A')
                 country = loc.get('location', {}).get('country', 'N/A')
                 location_summaries.append(f"{facility}, {city}, {state}, {country}")
            locations_snippet = clean_text("; ".join(location_summaries), max_len=300)


            # Basic info (BriefTitle is directly under study in V2 examples, not protocolSection)
            title = clean_text(study.get('briefTitle', 'N/A'))

            url = f"https://clinicaltrials.gov/study/{nct_id}" if nct_id != 'NCTId is missing' else "#"

            processed_results.append({
                 "nct_id": nct_id,
                 "title": title,
                 "status": status,
                 "phase": phases,
                 "study_type": study_type,
                 "conditions": conditions,
                 "summary": brief_summary,
                 "interventions": interventions_str,
                 "eligibility_snippet": eligibility_criteria,
                 "contact_info": "See Study Link", # Contact details are complex in V2, direct link is best
                 "locations_snippet": locations_snippet,
                 "url": url # This URL goes in the expander, not main text
            })
        results = processed_results
        logging.info(f"Successfully processed {len(results)} trials from ClinicalTrials.gov response.")

    except requests.exceptions.Timeout:
        logging.error("CT.gov API request timed out.")
        st.warning("Clinical trial search timed out.")
    except requests.exceptions.RequestException as e:
        logging.error(f"CT.gov API request failed: {e}", exc_info=True)
        st.warning("Could not retrieve trials due to network issue.")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode CT.gov API JSON: {e}", exc_info=True)
        st.warning("Invalid response received from trial database.")
    except Exception as e:
        logging.error(f"Unexpected error processing CT.gov results: {e}", exc_info=True)
        st.warning("Unexpected error processing trials.")

    logging.info(f"ClinicalTrials.gov Search finished. Found {len(results)} results using keyword query: '{query_term_str}'")
    return results


# --- Streamlit App ---

st.set_page_config( page_title="Cancer Treatment Explorer", layout="wide", initial_sidebar_state="expanded" )

# --- Initialize Session State ---
def initialize_session():
    # Check if core session state exists, initialize fully if not
    if 'stage' not in st.session_state or 'messages' not in st.session_state or 'user_inputs' not in st.session_state:
        logging.info("Initializing/resetting full session state.")
        st.session_state.stage = STAGES["INIT"]
        st.session_state.user_inputs = {}
        st.session_state.messages = [] # Start fresh messages
        # Add initial intro message and first question
        st.session_state.messages.append({"role": "assistant", "content": STAGE_PROMPTS[STAGES["INIT"]], "type": "info"})
        st.session_state.messages.append({"role": "assistant", "content": STAGE_PROMPTS[STAGES["GET_DIAGNOSIS"]]})
        st.session_state.stage = STAGES["GET_DIAGNOSIS"] # Set the stage to the first question
        st.session_state.drug_results = []; st.session_state.trial_results = []
        st.session_state.consent_given = None # Use None, True, False
        if 'model_id' not in st.session_state:
             st.session_state.model_id = AVAILABLE_MODELS.get(DEFAULT_MODEL_DISPLAY_NAME, list(AVAILABLE_MODELS.values())[0]) # Fallback if default name invalid
        logging.info(f"Session state initialized to stage {STAGE_NAMES.get(st.session_state.stage, 'Unknown')}, model {st.session_state.model_id}.")
    else:
        # Ensure necessary keys exist even if session state was partially preserved from a previous state
        if 'user_inputs' not in st.session_state: st.session_state.user_inputs = {}
        if 'drug_results' not in st.session_state: st.session_state.drug_results = []
        if 'trial_results' not in st.session_state: st.session_state.trial_results = []
        if 'consent_given' not in st.session_state: st.session_state.consent_given = None
        # Add the current prompt if the last message wasn't it (handles refresh on input stages)
        current_prompt_text = STAGE_PROMPTS.get(st.session_state.stage)
        if current_prompt_text and (not st.session_state.messages or st.session_state.messages[-1].get("content") != current_prompt_text):
             # Avoid adding prompts for internal processing/end stages if they are already there
             if st.session_state.stage not in [STAGES["PROCESS_INFO_SHOW_DRUGS"], STAGES["SAVE_CONTACT_SHOW_TRIALS"], STAGES["SHOW_TRIALS_NO_CONSENT"], STAGES["FINAL_SUMMARY"], STAGES["END"]]:
                 msg = {"role": "assistant", "content": current_prompt_text}
                 if st.session_state.stage == STAGES["ASK_CONSENT"]: msg["type"] = "buttons"
                 st.session_state.messages.append(msg)
                 st.rerun()


initialize_session()

# --- Sidebar ---
with st.sidebar:
    st.subheader("⚙️ Configuration")
    model_display_names = list(AVAILABLE_MODELS.keys())
    current_model_id = st.session_state.get('model_id', AVAILABLE_MODELS.get(DEFAULT_MODEL_DISPLAY_NAME, list(AVAILABLE_MODELS.values())[0]))
    current_model_display_name = next((name for name, mid in AVAILABLE_MODELS.items() if mid == current_model_id), DEFAULT_MODEL_DISPLAY_NAME)
    try: default_index = model_display_names.index(current_model_display_name)
    except ValueError: default_index = 0 # Fallback if current model ID isn't in the list

    selected_model_display_name = st.selectbox( "AI Model:", options=model_display_names, index=default_index, key="model_select_widget", help="Select AI model for query strategy and summary generation." )
    new_model_id = AVAILABLE_MODELS[selected_model_display_name]
    if new_model_id != st.session_state.model_id:
        st.session_state.model_id = new_model_id
        st.toast(f"Model updated to: {selected_model_display_name}", icon="🤖")
        logging.info(f"AI Model updated to {new_model_id}")


    st.divider()
    if st.button("🔄 Restart Exploration", key="restart_sidebar"):
        logging.info("Restarting session from sidebar.")
        # Clear ALL session state keys except essential config like model_id
        keys_to_clear = [key for key in st.session_state.keys() if key not in ['model_id', 'model_select_widget']]
        for key in keys_to_clear:
            del st.session_state[key]
        initialize_session() # Re-initialize state to starting point
        st.rerun()

    st.divider(); st.markdown("---"); st.caption("Debug Info:")
    current_stage_num = st.session_state.get('stage', STAGES["INIT"])
    st.write(f"Current Stage: {STAGE_NAMES.get(current_stage_num, 'Unknown')} ({current_stage_num})")
    st.write("Consent Given:", st.session_state.get('consent_given', 'N/A'))
    # st.json(st.session_state.user_inputs) # Optional: uncomment for debugging inputs
    # st.write("Drugs:", len(st.session_state.drug_results or [])) # Optional debug counts
    # st.write("Trials:", len(st.session_state.trial_results or []))


# --- Main App Area Layout ---

# Fixed Header Area
header_container = st.container()
with header_container:
    st.title("🧑‍⚕️ Cancer Treatment Explorer")
    st.caption("AI-Assisted Public Information Retrieval (Not Medical Advice)")
    # Add main disclaimer clearly
    st.markdown(
        """
        <div style="padding: 10px; border: 1px solid #ff9900; border-radius: 5px; margin-bottom: 10px; background-color: #fff3e0;">
            ⚠️ <strong>Disclaimer:</strong> This application provides information based on publicly available data from the FDA and ClinicalTrials.gov. It is intended for informational exploration only and **does not constitute medical advice.** Consult with a qualified healthcare professional for any health concerns or before making any decisions related to your health or treatment.
        </div>
        """,
        unsafe_allow_html=True
    )


# Scrollable Chat Area
chat_container = st.container(height=CHAT_CONTAINER_HEIGHT) # Scrollable chat area
with chat_container:
    # Display messages inside the scrollable container
    for message in st.session_state.get('messages', []):
        avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else USER_AVATAR
        name = ASSISTANT_NAME if message["role"] == "assistant" else USER_NAME
        with st.chat_message(name=message["role"], avatar=avatar):
            # Use markdown for message content, allowing HTML for expanders/info boxes
            if message.get("type") == "expander":
                with st.expander(message.get("title", "Details"), expanded=False):
                     # Use markdown within the expander content for formatting
                     st.markdown(message["content"], unsafe_allow_html=True)
            elif message.get("type") == "info":
                 # Custom styling for info boxes like the initial disclaimer or end message
                 st.markdown(message["content"], unsafe_allow_html=True)
            else:
                # Regular chat messages
                st.markdown(message["content"], unsafe_allow_html=True)


# --- Input/Button Area --- (Placed after scrollable area)

# Handle consent stage (buttons appear below chat_input implicitly)
if st.session_state.stage == STAGES["ASK_CONSENT"]:
    # Display the prompt above buttons if it's not the last message
    current_prompt_text = STAGE_PROMPTS.get(st.session_state.stage)
    if st.session_state.messages and st.session_state.messages[-1].get("content") != current_prompt_text:
         st.write(current_prompt_text)

    cols = st.columns([1, 1, 6]) # Use columns to place buttons side-by-side
    with cols[0]:
        if st.button("✔️ Yes, Agree", key="consent_yes"):
            logging.info("User consented to contact info.")
            st.session_state.consent_given = True
            # Append user confirmation message
            st.session_state.messages.append({"role": "user", "content": "A: Yes, I agree to share contact information."})
            advance_stage(STAGES["GET_NAME"])
            st.rerun()
    with cols[1]:
         if st.button("❌ No, Decline", key="consent_no"):
            logging.info("User declined contact info.")
            st.session_state.consent_given = False
            # Append user confirmation message
            st.session_state.messages.append({"role": "user", "content": "A: No, I do not wish to share contact information now."})
            # Save user context even without contact details (for record-keeping demo)
            context_data = {
                "ConsentGiven": False,
                "Diagnosis": st.session_state.user_inputs.get("diagnosis", "N/A"),
                "StageProgression": st.session_state.user_inputs.get("stage", "N/A"),
                "Biomarkers": st.session_state.user_inputs.get("biomarkers", "N/A"),
                "PriorTreatment": st.session_state.user_inputs.get("prior_treatment", "N/A"),
                "ImagingResponse": st.session_state.user_inputs.get("imaging", "N/A")
            }
            save_user_data(context_data) # Attempt to save context
            advance_stage(STAGES["SHOW_TRIALS_NO_CONSENT"]) # Skip contact info steps
            st.rerun()

# Handle text input stage
elif st.session_state.stage not in [STAGES["ASK_CONSENT"], STAGES["END"], STAGES["PROCESS_INFO_SHOW_DRUGS"], STAGES["SAVE_CONTACT_SHOW_TRIALS"], STAGES["SHOW_TRIALS_NO_CONSENT"], STAGES["FINAL_SUMMARY"]]:
    current_prompt_text = STAGE_PROMPTS.get(st.session_state.stage, "Enter your response...")
    # Extract example text for placeholder
    placeholder_match = re.search(r"\((e\.g\.,.*?)\)", current_prompt_text)
    placeholder = placeholder_match.group(1) if placeholder_match else "Your answer..."

    user_input = st.chat_input(placeholder=placeholder, key="user_text_input") # Use st.chat_input

    # Process user input if available and not in a processing/final stage
    if user_input: # Only process if input is given
        st.session_state.messages.append({"role": "user", "content": f"A: {user_input}"})
        current_stage = st.session_state.stage
        next_stage = None

        # --- Stage advancement logic based on current stage ---
        if current_stage == STAGES["GET_DIAGNOSIS"]:
            st.session_state.user_inputs['diagnosis'] = user_input.strip()
            next_stage = STAGES["GET_STAGE"]
        elif current_stage == STAGES["GET_STAGE"]:
            st.session_state.user_inputs['stage'] = user_input.strip()
            next_stage = STAGES["GET_BIOMARKERS"]
        elif current_stage == STAGES["GET_BIOMARKERS"]:
            st.session_state.user_inputs['biomarkers'] = user_input.strip()
            next_stage = STAGES["GET_PRIOR_TREATMENT"]
        elif current_stage == STAGES["GET_PRIOR_TREATMENT"]:
            st.session_state.user_inputs['prior_treatment'] = user_input.strip()
            next_stage = STAGES["GET_IMAGING"]
        elif current_stage == STAGES["GET_IMAGING"]:
            st.session_state.user_inputs['imaging'] = user_input.strip()
            # After imaging, we go to the processing stage for drugs
            next_stage = STAGES["PROCESS_INFO_SHOW_DRUGS"]
        elif current_stage == STAGES["GET_NAME"]:
            st.session_state.user_inputs['name'] = user_input.strip()
            next_stage = STAGES["GET_EMAIL"]
        elif current_stage == STAGES["GET_EMAIL"]:
            # Basic email validation
            if "@" not in user_input or "." not in user_input:
                st.session_state.messages.append({"role": "assistant", "content": "⚠️ Please enter a valid email address."})
                # Stay on the current stage to re-ask - no next_stage change
            else:
                st.session_state.user_inputs['email'] = user_input.strip()
                next_stage = STAGES["GET_PHONE"]
        elif current_stage == STAGES["GET_PHONE"]:
             st.session_state.user_inputs['phone'] = user_input.strip() if user_input.strip() else "N/A" # Handle empty input for optional phone
             # After phone, we go to the save contact stage
             next_stage = STAGES["SAVE_CONTACT_SHOW_TRIALS"]

        # Advance stage if next_stage is determined by the logic
        if next_stage is not None:
            advance_stage(next_stage)
            st.rerun()


# --- Internal Processing Stages (Triggered by st.rerun after stage change) ---

if st.session_state.stage == STAGES["PROCESS_INFO_SHOW_DRUGS"]:
    logging.info("Executing stage: PROCESS_INFO_SHOW_DRUGS")
    # Ensure user inputs are available
    user_inputs = st.session_state.user_inputs
    diagnosis = user_inputs.get("diagnosis", "cancer")
    stage_info = user_inputs.get("stage", "")
    markers = user_inputs.get("biomarkers", "")
    model_id = st.session_state.model_id

    # Only run API/LLM calls once per stage transition
    # Check if drug results are already stored OR if the drug summary message is already present
    if not st.session_state.drug_results and not any(m.get("content", "").strip().startswith("**Potential Therapeutics") for m in st.session_state.messages):
        try:
            st.session_state.messages.append({"role": "assistant", "content": "Analyzing your information and searching for relevant FDA-approved drugs..."})
            with st.spinner("Generating advanced FDA search strategy and searching database..."):
                search_strategy = refine_fda_search_strategy_with_llm(diagnosis, stage_info, markers, model_id)
                st.session_state.drug_results = search_fda_drugs_for_condition_and_markers(search_strategy)

            # Generate LLM summary BEFORE showing details
            drug_summary = generate_drug_summary_llm(diagnosis, markers, st.session_state.drug_results, model_id)
            st.session_state.messages.append({"role": "assistant", "content": f"**Potential Therapeutics (Based on FDA Label Search)**\n\n{drug_summary}"})

            # Add expander message with details if results found
            if st.session_state.drug_results:
                num_found = len(st.session_state.drug_results)
                expander_title = f"View {num_found} Found Drug Detail{'s' if num_found != 1 else ''}"
                expander_content = f"**Found {num_found} potentially relevant drug(s):**\n"
                expander_content += f"*_Search based on FDA 'Indications & Usage' for: {diagnosis}, {markers}, {stage_info}_*\n\n"

                for i, drug in enumerate(st.session_state.drug_results):
                    expander_content += f"\n---\n**{i+1}. {drug.get('brand_name', 'N/A')}** ({drug.get('generic_name', 'N/A')})\n\n"
                    expander_content += f"   *   **Indication Snippet:** {drug.get('indication_snippet', 'N/A')}\n"
                    if drug.get('url') and drug.get('url') != '#':
                        # Add link inside expander
                        expander_content += f"   *   **More Info:** [Search Google for Prescribing Information]({drug.get('url')})\n"

                st.session_state.messages.append({"role": "assistant", "content": expander_content.strip(), "type": "expander", "title": expander_title})
            else:
                 st.session_state.messages.append({"role": "assistant", "content": "No specific FDA-approved drugs matching the search criteria were found. Please review your input or consult with your physician."})


        except Exception as e:
            logging.error("Error during PROCESS_INFO_SHOW_DRUGS stage", exc_info=True)
            st.error(f"An error occurred during drug analysis: {e}")
            st.session_state.messages.append({"role": "assistant", "content": "Could not complete the drug analysis due to an error. Please try restarting."})

        # Always advance after processing drugs (either successfully or after error)
        # Check if the next stage message is already present before adding it again
        if not st.session_state.messages or st.session_state.messages[-1].get("content") != STAGE_PROMPTS.get(STAGES["ASK_CONSENT"]):
            advance_stage(STAGES["ASK_CONSENT"])
            st.rerun()


if st.session_state.stage == STAGES["SAVE_CONTACT_SHOW_TRIALS"]:
    logging.info("Executing stage: SAVE_CONTACT_SHOW_TRIALS")
    # This stage is only reached if consent was given and contact info was gathered
    if st.session_state.consent_given:
        # Only save once per stage transition
        if not any(m.get("content", "").strip().startswith("Thank you! Your contact information has been recorded") or m.get("content", "").strip().startswith("⚠️ There was an issue recording your contact information") for m in st.session_state.messages):
            contact_data = {
                "ConsentGiven": True,
                "Name": st.session_state.user_inputs.get("name", "N/A"),
                "Email": st.session_state.user_inputs.get("email", "N/A"),
                "Phone": st.session_state.user_inputs.get("phone", "N/A"),
                "Diagnosis": st.session_state.user_inputs.get("diagnosis", "N/A"),
                "StageProgression": st.session_state.user_inputs.get("stage", "N/A"),
                "Biomarkers": st.session_state.user_inputs.get("biomarkers", "N/A"),
                "PriorTreatment": st.session_state.user_inputs.get("prior_treatment", "N/A"),
                "ImagingResponse": st.session_state.user_inputs.get("imaging", "N/A"),
            }
            if save_user_data(contact_data):
                st.toast("Contact info recorded (demo purposes only).", icon="✅")
                st.session_state.messages.append({"role": "assistant", "content": "Thank you! Your contact information has been recorded (for demo purposes)."})
            else:
                st.toast("Failed to save contact info.", icon="⚠️")
                st.session_state.messages.append({"role": "assistant", "content": "⚠️ There was an issue recording your contact information."})

    # Always proceed to show trials after attempting save (or if consent wasn't given via this path)
    advance_stage(STAGES["SHOW_TRIALS_NO_CONSENT"])
    st.rerun()


if st.session_state.stage == STAGES["SHOW_TRIALS_NO_CONSENT"]: # This stage is reached either after declining consent or after saving contact info
    logging.info("Executing stage: SHOW_TRIALS_NO_CONSENT (Show Trials)")
    # Ensure user inputs are available
    user_inputs = st.session_state.user_inputs
    diagnosis = user_inputs.get("diagnosis", "cancer")
    stage_info = user_inputs.get("stage", "")
    markers = user_inputs.get("biomarkers", "")
    prior_treatment = user_inputs.get("prior_treatment", "")
    model_id = st.session_state.model_id

    # Only run API/LLM calls once per stage transition
    # Check if trial results are already stored OR if the trial summary message is already present
    if not st.session_state.trial_results and not any(m.get("content", "").strip().startswith("**Potential Clinical Trials") for m in st.session_state.messages):
        try:
            st.session_state.messages.append({"role": "assistant", "content": "Searching ClinicalTrials.gov for relevant studies using AI-generated keywords..."})
            with st.spinner("Generating search terms and searching ClinicalTrials.gov database..."):
                 # Use the new function that calls LLM for keywords and then searches
                 st.session_state.trial_results = search_clinical_trials_with_keywords(diagnosis, stage_info, markers, prior_treatment, model_id, limit=CT_REQUEST_LIMIT)

            # Generate LLM summary BEFORE showing details
            trial_summary = generate_trial_summary_llm(diagnosis, markers, stage_info, st.session_state.trial_results, model_id)
            st.session_state.messages.append({"role": "assistant", "content": f"**Potential Clinical Trials (Keyword Search)**\n\n{trial_summary}"})

            # Add expander message with details if results found
            if st.session_state.trial_results:
                num_found = len(st.session_state.trial_results)
                num_to_display = min(num_found, CT_DISPLAY_LIMIT)
                expander_title = f"View Top {num_to_display} (of {num_found}) Clinical Trial Detail{'s' if num_to_display != 1 else ''}"
                expander_content = f"**Found {num_found} potentially relevant trial(s), showing top {num_to_display}:**\n\n"
                # Updated clarification based on using only query.term
                expander_content += f"*_Search based on AI-generated keywords derived from your input (e.g., diagnosis, markers, stage, 'recruiting', 'phase 2/3/4', 'interventional'). Results are sorted by relevance by the API._*\n"
                expander_content += f"*_Always verify details, eligibility criteria, and locations on ClinicalTrials.gov via the link._*\n"


                for i, trial in enumerate(st.session_state.trial_results[:num_to_display]): # Limit displayed results in expander
                    expander_content += f"\n---\n**{i+1}. {trial.get('nct_id', 'N/A')}: {trial.get('title', 'N/A')}**\n\n"
                    # Using Markdown list format for details
                    expander_content += f"-   **Status:** {trial.get('status','N/A')}\n"
                    expander_content += f"-   **Phase:** {trial.get('phase','N/A')}\n"
                    expander_content += f"-   **Type:** {trial.get('study_type','N/A')}\n"
                    expander_content += f"-   **Conditions:** {trial.get('conditions','N/A')}\n"
                    expander_content += f"-   **Interventions:** {trial.get('interventions','N/A')}\n"
                    # Add eligibility, locations snippets if available
                    if trial.get('eligibility_snippet') and trial['eligibility_snippet'] != 'N/A':
                         expander_content += f"-   **Eligibility Snippet:** {trial.get('eligibility_snippet')}\n"
                    if trial.get('locations_snippet') and trial['locations_snippet'] != 'N/A':
                         expander_content += f"-   **Locations Snippet:** {trial.get('locations_snippet')}\n"

                    if trial.get('url') and trial.get('url') != '#':
                         # Add link inside expander
                         expander_content += f"-   **Link:** [View Full Details on ClinicalTrials.gov]({trial.get('url')})\n"
                    # Add summary last
                    if trial.get('summary') and trial['summary'] != 'N/A':
                         expander_content += f"-   **Summary:** {trial.get('summary')}\n"


                st.session_state.messages.append({"role": "assistant", "content": expander_content.strip(), "type": "expander", "title": expander_title})
            else:
                 st.session_state.messages.append({"role": "assistant", "content": "No clinical trials matching the keyword search criteria were found. Please review your input, try broader terms, or consult with your physician."})

        except Exception as e:
            logging.error("Error during SHOW_TRIALS_NO_CONSENT stage", exc_info=True)
            st.error(f"An error occurred during clinical trial analysis: {e}")
            st.session_state.messages.append({"role": "assistant", "content": "Could not complete the clinical trial analysis due to an error. Please try restarting."})

        # Always advance after showing trials (either successfully or after error)
        # Check if the next stage message is already present before adding it again
        if not st.session_state.messages or st.session_state.messages[-1].get("content") != STAGE_PROMPTS.get(STAGES["FINAL_SUMMARY"]):
             advance_stage(STAGES["FINAL_SUMMARY"])
             st.rerun()


if st.session_state.stage == STAGES["FINAL_SUMMARY"]:
    logging.info("Executing stage: FINAL_SUMMARY")
    # Only generate final summary once per stage transition
    # Check if the final summary message is already present (distinguished by starting with "**Final Summary**")
    final_summary_already_generated = any(
        m["role"] == "assistant" and m["content"].strip().startswith("**Final Summary**")
        for m in st.session_state.messages[-2:] # Check last couple of messages
    )

    # Add "Generating..." message if it hasn't been added yet for this transition
    generating_message_text = STAGE_PROMPTS.get(STAGES["FINAL_SUMMARY"])
    generating_message_present = (st.session_state.messages and st.session_state.messages[-1].get("content") == generating_message_text)

    if not generating_message_present and not final_summary_already_generated:
        st.session_state.messages.append({"role": "assistant", "content": generating_message_text})
        st.rerun() # Rerun to display the generating message
        # After this rerun, the script will re-execute and enter this stage again,
        # and `generating_message_present` will be True.

    # Now, if the generating message is present BUT the final summary isn't yet,
    # generate and display the final summary.
    if generating_message_present and not final_summary_already_generated:
         try:
             final_summary = generate_final_summary_llm(
                 st.session_state.user_inputs,
                 st.session_state.drug_results,
                 st.session_state.trial_results,
                 st.session_state.model_id
             )
             # Append the final summary message after generation
             st.session_state.messages.append({"role": "assistant", "content": f"**Final Summary**\n\n{final_summary}"})

         except Exception as e:
              logging.error("Error generating final summary", exc_info=True)
              st.error(f"An error occurred while generating the final summary: {e}")
              st.session_state.messages.append({"role": "assistant", "content": "Could not generate the final summary due to an error. Please review the details above and consult your physician."})

         # Always advance to END stage after attempting final summary generation
         advance_stage(STAGES["END"])
         st.rerun() # Rerun to trigger the END stage logic for displaying the final END message and button
    # If the final summary IS already generated, and we are in this stage,
    # the next logical step is the END stage, which will be handled by the subsequent check/block.


if st.session_state.stage == STAGES["END"]:
    logging.info("Reached END stage.")
    # Display the end prompt if it's not already the last message
    end_prompt_text = STAGE_PROMPTS.get(STAGES["END"])
    # Check last 2 messages in case final summary was just added before it
    end_message_present = any(m.get('content') == end_prompt_text for m in st.session_state.messages[-2:])

    if not end_message_present:
        # Add it as an info box for clear separation
         st.session_state.messages.append({"role": "assistant", "content": end_prompt_text, "type":"info"})
         st.rerun() # Rerun to display the added end message

    # Offer restart button below the chat area
    # This button will only be rendered when stage is END
    st.info("Session ended. Use the sidebar or button below to start a new exploration.")
    if st.button("🔄 Start New Exploration", key="restart_main_button"):
        logging.info("Restarting session from main button.")
        # Clear ALL session state keys except essential config like model_id
        keys_to_clear = [key for key in st.session_state.keys() if key not in ['model_id', 'model_select_widget']]
        for key in keys_to_clear:
            del st.session_state[key]
        initialize_session() # Re-initialize state to starting point
        st.rerun()