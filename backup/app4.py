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
CTGOV_API_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

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
# Define stages/states instead of just questions for more complex flow control
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
    STAGES["INIT"]: "Hi there! I'm an AI assistant designed to help you explore information about cancer treatments using public data. I work alongside your physician and **do not provide medical advice.** Let's start by gathering some information. Please answer the questions as accurately as possible.\n\nFirst: 👇", # Added arrow to guide user
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

# Map stage numbers back to names for debugging display
STAGE_NAMES = {v: k for k, v in STAGES.items()}

# --- Helper Functions ---

def clean_text(text):
    """Basic text cleaning: remove HTML, normalize whitespace."""
    if not text: return "N/A"
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text) # Remove HTML tags
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    return text

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

# --- API & LLM Functions (Robust Implementations Needed) ---
# ============================================================
# IMPORTANT: Replace the dummy logic below with your actual,
# robust API calling and LLM interaction functions, including
# proper error handling (timeouts, retries, status codes),
# data parsing, and potentially more sophisticated querying.
# ============================================================

def search_fda_drugs_for_condition_and_markers(diagnosis, markers, limit=15):
    """
    Searches FDA drug labels via openFDA API.
    Placeholder - Requires full implementation with error handling.
    """
    logging.info(f"[API Call - Placeholder] Searching FDA drugs for: D='{diagnosis}', M='{markers}'")
    # *** Replace with your full FDA search implementation ***
    time.sleep(1) # Simulate network delay
    results = []
    if diagnosis and "lung cancer" in diagnosis.lower():
        if markers and "egfr" in markers.lower():
             results.append({
                 "brand_name": "Tagrisso", "generic_name": "Osimertinib",
                 "indication_snippet": "...treatment of patients with metastatic non-small cell lung cancer (NSCLC) whose tumors have epidermal growth factor receptor (EGFR) exon 19 deletions or exon 21 L858R mutations...",
                 "url": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2020/208065s014lbl.pdf" # Example URL
                })
        results.append({
            "brand_name": "Keytruda", "generic_name": "Pembrolizumab",
            "indication_snippet": "...in combination with pemetrexed and platinum chemotherapy, for the first-line treatment of patients with metastatic nonsquamous non-small cell lung cancer (NSCLC), with no EGFR or ALK genomic tumor aberrations...",
             "url": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2021/125514s098lbl.pdf"
        })
    logging.info(f"FDA Placeholder Search returned {len(results)} results.")
    return results

def search_filtered_clinical_trials(diagnosis, stage_info, markers, prior_treatment, limit=20):
    """
    Searches ClinicalTrials.gov V2 API with filters.
    Placeholder - Requires full implementation with error handling.
    """
    logging.info(f"[API Call - Placeholder] Searching ClinicalTrials.gov for: D='{diagnosis}', S='{stage_info}', M='{markers}', Prior='{prior_treatment}'")
    # *** Replace with your full ClinicalTrials.gov V2 API search implementation ***
    time.sleep(2) # Simulate network delay
    results = []
    if diagnosis and "lung cancer" in diagnosis.lower():
        if markers and "egfr" in markers.lower():
             results.append({
                 "nct_id": "NCT0XXXXXX1",
                 "title": "A Study of Novel Agent X in Patients With EGFR-Mutated NSCLC Progressing on Osimertinib",
                 "status": "RECRUITING", "phase": "PHASE 2", "study_type": "INTERVENTIONAL",
                 "conditions": "EGFR Positive Non-Small Cell Lung Cancer",
                 "summary": "This study evaluates the efficacy and safety of Novel Agent X in patients with advanced NSCLC harboring EGFR mutations...",
                 "contact_info": "Trial Contact: 1-800-555-TRIAL / email@example.com", # Example contact
                 "locations_snippet": "Multiple sites across the US and Europe",
                 "url": f"https://clinicaltrials.gov/study/NCT0XXXXXX1" # Example URL
                 })
    results.append({
        "nct_id": "NCT0YYYYYY2",
        "title": "Immunotherapy Combination Study for Advanced NSCLC After Chemotherapy",
        "status": "RECRUITING", "phase": "PHASE 3", "study_type": "INTERVENTIONAL",
        "conditions": "Non-Small Cell Lung Cancer Stage IV",
        "summary": "Comparing standard of care vs. a new immunotherapy combination in patients who have received prior platinum-based chemotherapy.",
        "contact_info": "Trial Contact: research.center@example.org",
        "locations_snippet": "USA, Canada, Australia",
        "url": f"https://clinicaltrials.gov/study/NCT0YYYYYY2"
    })
    logging.info(f"ClinicalTrials.gov Placeholder Search returned {len(results)} results.")
    return results

def generate_drug_summary_llm(diagnosis, markers, fda_drugs, model_id):
    """
    Generates the CAUTIOUS drug summary using LLM.
    Placeholder - Requires full implementation with Groq API call.
    """
    logging.info(f"[LLM Call - Placeholder] Generating drug summary for: D='{diagnosis}', M='{markers}', Model={model_id}")
    # *** Replace with your full cautious drug summary LLM call implementation ***
    time.sleep(1.5) # Simulate LLM processing
    if not GROQ_API_KEY: return "**Note:** Groq API key not configured. Cannot generate AI summary. Showing raw findings."

    if fda_drugs:
        drug_list = [f"{d['brand_name']} ({d['generic_name']})" for d in fda_drugs]
        summary = (f"Based on automated searches of FDA drug labels for terms related to '{diagnosis}' and markers like '{markers}', several drugs were found, including: {', '.join(drug_list)}. "
                   f"These labels contain keywords potentially relevant to your situation (see details below). "
                   f"**Crucially, this automated search does not confirm suitability or effectiveness for you.** Drug selection depends on many factors not captured here. "
                   f"Please **discuss these specific drugs and your full medical history with your oncologist** to understand if they are appropriate options.")
    else:
        summary = (f"An automated search of FDA drug label indication texts did not find specific matches for approved drugs based *solely* on the terms '{diagnosis}' and markers '{markers}'. "
                   f"This does *not* mean there are no relevant treatments. Standard treatments or drugs approved for broader indications might apply. "
                   f"**It is essential to discuss all treatment options with your oncologist.**")
    logging.info("LLM Placeholder Drug Summary generated.")
    return summary

def generate_trial_summary_llm(diagnosis, markers, stage_info, prior_treatment, clinical_trials, model_id):
    """
    Generates the clinical trial summary using LLM.
    Placeholder - Requires full implementation with Groq API call.
    """
    logging.info(f"[LLM Call - Placeholder] Generating trial summary for: D='{diagnosis}', M='{markers}', S='{stage_info}', Prior='{prior_treatment}', Model={model_id}")
    # *** Replace with your full trial summary LLM call implementation ***
    time.sleep(2) # Simulate LLM processing
    if not GROQ_API_KEY: return "**Note:** Groq API key not configured. Cannot generate AI summary. Showing raw findings."

    if clinical_trials:
        trial_titles = [f"'{t['title'][:60]}...'" for t in clinical_trials[:2]] # Show first few titles concisely
        summary = (f"Searches on ClinicalTrials.gov for actively recruiting Phase 2 or 3 interventional studies related to '{diagnosis}' (considering stage '{stage_info}', markers '{markers}', prior treatments like '{prior_treatment}') yielded potential matches, such as: {', '.join(trial_titles)}. "
                   f"These trials are investigating newer approaches (details below). "
                   f"**Eligibility for clinical trials is very specific and depends on detailed medical criteria not fully assessed here.** "
                   f"Please **review the details of these or other trials carefully with your oncologist** to determine if any are suitable and available for you.")
    else:
        summary = (f"Based on the specific criteria provided (diagnosis '{diagnosis}', stage '{stage_info}', markers '{markers}', prior treatment '{prior_treatment}') and filters (Recruiting, Phase 2+, Interventional), no matching clinical trials were found in this automated search of ClinicalTrials.gov. "
                   f"Trial availability changes frequently, and different search terms or broader criteria might yield results. "
                   f"**Discuss clinical trial options thoroughly with your oncologist, as they have the best understanding of relevant studies.**")
    logging.info("LLM Placeholder Trial Summary generated.")
    return summary

# --- Streamlit App Layout ---

# Page Config (do this first)
st.set_page_config(
    page_title="Cancer Treatment Explorer",
    layout="wide",
    initial_sidebar_state="expanded" # Keep sidebar visible always
)

# --- Initialize Session State FIRST ---
# Ensure initialization happens before accessing session state for display
def initialize_session():
    # Check if the 'stage' key itself is missing, indicating first run or reset
    if 'stage' not in st.session_state:
        st.session_state.stage = STAGES["INIT"]
        st.session_state.user_inputs = {}
        # Add the *very first* introductory message
        st.session_state.messages = [{"role": "assistant", "content": STAGE_PROMPTS[STAGES["INIT"]]}]
        # Then add the *first actual question* right after the intro
        st.session_state.messages.append({"role": "assistant", "content": STAGE_PROMPTS[STAGES["GET_DIAGNOSIS"]]})
        st.session_state.stage = STAGES["GET_DIAGNOSIS"] # Set stage to expect answer for the first question

        # Initialize other keys
        st.session_state.drug_results = None
        st.session_state.trial_results = None
        st.session_state.consent_given = None
        if 'model_id' not in st.session_state: # Initialize model only if not already set
             st.session_state.model_id = AVAILABLE_MODELS[DEFAULT_MODEL_DISPLAY_NAME]
        logging.info("Session state initialized for a new conversation.")

initialize_session() # Run initialization logic


# --- Sidebar ---
# Now that session state is guaranteed to be initialized, draw the sidebar
with st.sidebar:
    st.subheader("⚙️ Configuration")
    # Model selection
    model_display_names = list(AVAILABLE_MODELS.keys())
    # Get current model safely, defaulting if needed (though init should prevent this)
    current_model_id = st.session_state.get('model_id', AVAILABLE_MODELS[DEFAULT_MODEL_DISPLAY_NAME])
    current_model_display_name = next((name for name, mid in AVAILABLE_MODELS.items() if mid == current_model_id), DEFAULT_MODEL_DISPLAY_NAME)
    try:
        default_index = model_display_names.index(current_model_display_name)
    except ValueError:
        default_index = 0 # Fallback

    selected_model_display_name = st.selectbox(
        "AI Model:",
        options=model_display_names, index=default_index, key="model_select_widget",
        help="Select the AI model used for generating summaries."
    )
    new_model_id = AVAILABLE_MODELS[selected_model_display_name]
    if new_model_id != st.session_state.model_id:
        st.session_state.model_id = new_model_id
        st.success(f"Model updated to: {selected_model_display_name}")

    st.divider()
    if st.button("🔄 Restart Conversation", key="restart_sidebar"):
        # Clear relevant session state keys, trigger re-initialization
        keys_to_clear = ['stage', 'user_inputs', 'messages', 'drug_results', 'trial_results', 'consent_given']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        # Don't clear model_id
        st.rerun()

    st.divider()
    st.markdown("---")
    st.caption("Debug Info:")
    # Access stage safely, using the name mapping
    current_stage_num = st.session_state.get('stage', STAGES["INIT"]) # Default to INIT if somehow missing
    st.write(f"Current Stage: {STAGE_NAMES.get(current_stage_num, 'Unknown')} ({current_stage_num})")
    st.write("Consent Given:", st.session_state.get('consent_given', 'N/A'))


# --- Main App Area ---
st.title("🧑‍⚕️ Cancer Treatment Explorer")
st.caption("AI-Assisted Information Retrieval (Not Medical Advice)")


# --- Main Chat Area ---
# Reduce height to try and fit everything in viewport
chat_container = st.container(height=450)

with chat_container:
    # Display all messages *except* the initial intro message if it's not the only one
    # This avoids repeating the intro after the first question is shown.
    # Or simply display all, the flow is logical anyway. Let's display all for simplicity.
    for message in st.session_state.get('messages', []): # Use .get for safety
        avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else USER_AVATAR
        name = ASSISTANT_NAME if message["role"] == "assistant" else USER_NAME
        with st.chat_message(name=message["role"], avatar=avatar):
            if message.get("type") == "expander":
                with st.expander(message.get("title", "Details"), expanded=False):
                    st.markdown(message["content"], unsafe_allow_html=True)
            elif message.get("type") == "buttons":
                 st.markdown(message["content"]) # Button text/question
                 # Actual buttons rendered below input logic
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


# Handle consent buttons specifically (placed below chat_input area logically)
if st.session_state.stage == STAGES["ASK_CONSENT"]:
    # Use columns for button layout
    cols = st.columns(8)
    with cols[0]:
        if st.button("✔️ Yes, Agree", key="consent_yes"):
            st.session_state.consent_given = True
            st.session_state.user_inputs['consent'] = True
            st.session_state.messages.append({"role": "user", "content": "A: Yes, I agree to share contact information."})
            advance_stage(STAGES["GET_NAME"])
            st.rerun()
    with cols[1]:
         if st.button("❌ No, Decline", key="consent_no"):
            st.session_state.consent_given = False
            st.session_state.user_inputs['consent'] = False
            st.session_state.messages.append({"role": "user", "content": "A: No, I do not wish to share contact information now."})
            context_data = {
                "Timestamp": datetime.now().isoformat(), "ConsentGiven": False,
                "Diagnosis": st.session_state.user_inputs.get("diagnosis"),
                "StageProgression": st.session_state.user_inputs.get("stage"),
                "Biomarkers": st.session_state.user_inputs.get("biomarkers"),
                "PriorTreatment": st.session_state.user_inputs.get("prior_treatment"),
                "ImagingResponse": st.session_state.user_inputs.get("imaging"),
            }
            save_user_data(context_data)
            st.session_state.stage = STAGES["SHOW_TRIALS_NO_CONSENT"]
            st.rerun()


# Use st.chat_input for user text input, only if not in a button/processing/end stage
elif st.session_state.stage not in [
    STAGES["ASK_CONSENT"], STAGES["END"], STAGES["PROCESS_INFO_SHOW_DRUGS"],
    STAGES["SAVE_CONTACT_SHOW_TRIALS"], STAGES["SHOW_TRIALS_NO_CONSENT"]
    ]:
    # Determine placeholder text based on the current stage's *expected* prompt
    current_prompt = STAGE_PROMPTS.get(st.session_state.stage, "Enter response...")
    placeholder_match = re.search(r"\((e\.g\.,.*?)\)", current_prompt)
    placeholder = placeholder_match.group(1) if placeholder_match else "Your answer..."

    if user_input := st.chat_input(placeholder):
        st.session_state.messages.append({"role": "user", "content": f"A: {user_input}"})

        current_stage = st.session_state.stage # Capture stage before potential change
        next_stage = None # Flag to determine if we should advance

        if current_stage == STAGES["GET_DIAGNOSIS"]:
            st.session_state.user_inputs['diagnosis'] = user_input
            next_stage = STAGES["GET_STAGE"]
        elif current_stage == STAGES["GET_STAGE"]:
            st.session_state.user_inputs['stage'] = user_input
            next_stage = STAGES["GET_BIOMARKERS"]
        elif current_stage == STAGES["GET_BIOMARKERS"]:
            st.session_state.user_inputs['biomarkers'] = user_input
            next_stage = STAGES["GET_PRIOR_TREATMENT"]
        elif current_stage == STAGES["GET_PRIOR_TREATMENT"]:
            st.session_state.user_inputs['prior_treatment'] = user_input
            next_stage = STAGES["GET_IMAGING"]
        elif current_stage == STAGES["GET_IMAGING"]:
            st.session_state.user_inputs['imaging'] = user_input
            st.session_state.stage = STAGES["PROCESS_INFO_SHOW_DRUGS"] # Direct state change for processing
        elif current_stage == STAGES["GET_NAME"]:
            st.session_state.user_inputs['name'] = user_input
            next_stage = STAGES["GET_EMAIL"]
        elif current_stage == STAGES["GET_EMAIL"]:
            if "@" not in user_input or "." not in user_input:
                st.session_state.messages.append({"role": "assistant", "content": "⚠️ Please enter a valid email address."})
                # Do not set next_stage, stay on current stage
            else:
                st.session_state.user_inputs['email'] = user_input
                next_stage = STAGES["GET_PHONE"]
        elif current_stage == STAGES["GET_PHONE"]:
            st.session_state.user_inputs['phone'] = user_input if user_input else "N/A"
            st.session_state.stage = STAGES["SAVE_CONTACT_SHOW_TRIALS"] # Direct state change

        # Advance stage if next_stage was set
        if next_stage is not None:
            advance_stage(next_stage)

        st.rerun()


# --- Handle Internal Processing Stages ---
# (These stages run without requiring direct user input via chat_input)

if st.session_state.stage == STAGES["PROCESS_INFO_SHOW_DRUGS"]:
    logging.info("Entering stage: PROCESS_INFO_SHOW_DRUGS")
    with st.spinner("Analyzing drug information... Please wait."):
        diagnosis = st.session_state.user_inputs.get("diagnosis", "")
        markers = st.session_state.user_inputs.get("biomarkers", "")
        model_id = st.session_state.model_id
        st.session_state.drug_results = search_fda_drugs_for_condition_and_markers(diagnosis, markers)
        drug_summary = generate_drug_summary_llm(diagnosis, markers, st.session_state.drug_results, model_id)

    st.session_state.messages.append({
        "role": "assistant",
        "content": f"**Potential Therapeutics (Based on FDA Label Search)**\n\n{drug_summary}"
    })
    if st.session_state.drug_results:
        expander_content = "**Found Drug Details (Label Information):**\n\n"
        for i, drug in enumerate(st.session_state.drug_results):
            expander_content += f"**{i+1}. {drug.get('brand_name', 'N/A')} ({drug.get('generic_name', 'N/A')})**\n"
            expander_content += f"*   *Indication Snippet:* {clean_text(drug.get('indication_snippet', 'N/A'))}\n"
            if drug.get('url'):
                expander_content += f"*   *Source Label (Example):* [Link]({drug.get('url')})\n"
            expander_content += "---\n"
        st.session_state.messages.append({
            "role": "assistant", "content": expander_content.strip('---\n'),
            "type": "expander", "title": f"View {len(st.session_state.drug_results)} Found Drug Details"
        })
    elif st.session_state.drug_results == []:
        st.session_state.messages.append({"role": "assistant", "content": "_Note: No specific drug labels strongly matched the search terms in the indication field._"})

    advance_stage(STAGES["ASK_CONSENT"])
    st.rerun()


if st.session_state.stage == STAGES["SAVE_CONTACT_SHOW_TRIALS"]:
    logging.info("Entering stage: SAVE_CONTACT_SHOW_TRIALS")
    contact_data = {
        "Timestamp": datetime.now().isoformat(), "ConsentGiven": st.session_state.consent_given,
        "Name": st.session_state.user_inputs.get("name"), "Email": st.session_state.user_inputs.get("email"),
        "Phone": st.session_state.user_inputs.get("phone"), "Diagnosis": st.session_state.user_inputs.get("diagnosis"),
        "StageProgression": st.session_state.user_inputs.get("stage"), "Biomarkers": st.session_state.user_inputs.get("biomarkers"),
        "PriorTreatment": st.session_state.user_inputs.get("prior_treatment"), "ImagingResponse": st.session_state.user_inputs.get("imaging"),
    }
    if save_user_data(contact_data):
        st.toast("Contact info recorded (demo purposes only).", icon="✅")
    else:
        st.toast("Failed to save contact info.", icon="⚠️")

    st.session_state.stage = STAGES["SHOW_TRIALS_NO_CONSENT"] # Use the same next step logic
    st.rerun()


if st.session_state.stage == STAGES["SHOW_TRIALS_NO_CONSENT"]:
    logging.info("Entering stage: SHOW_TRIALS_NO_CONSENT (or after consent given)")
    with st.spinner("Analyzing clinical trial information... Please wait."):
        diagnosis = st.session_state.user_inputs.get("diagnosis", "")
        stage_info = st.session_state.user_inputs.get("stage", "")
        markers = st.session_state.user_inputs.get("biomarkers", "")
        prior_treatment = st.session_state.user_inputs.get("prior_treatment", "")
        model_id = st.session_state.model_id
        st.session_state.trial_results = search_filtered_clinical_trials(diagnosis, stage_info, markers, prior_treatment)
        trial_summary = generate_trial_summary_llm(diagnosis, markers, stage_info, prior_treatment, st.session_state.trial_results, model_id)

    st.session_state.messages.append({
        "role": "assistant",
        "content": f"**Potential Clinical Trials (Based on Filtered Search)**\n\n{trial_summary}"
    })
    if st.session_state.trial_results:
        expander_content = "**Found Clinical Trial Details (Recruiting, Phase 2+, Interventional):**\n\n"
        expander_content += f"*_Search based on: {diagnosis}, {stage_info}, {markers}, {prior_treatment}. Filters applied._*\n\n"
        for i, trial in enumerate(st.session_state.trial_results):
            expander_content += f"**{i+1}. {trial.get('nct_id', 'N/A')}: {trial.get('title', 'N/A')}**\n"
            expander_content += f"*   **Phase:** {trial.get('phase','N/A')} | **Status:** {trial.get('status','N/A')}\n"
            expander_content += f"*   **Conditions:** {clean_text(trial.get('conditions','N/A'))}\n"
            expander_content += f"*   **Summary Snippet:** {clean_text(trial.get('summary','N/A'))}\n"
            if trial.get('url'):
                expander_content += f"*   **Link:** [View on ClinicalTrials.gov]({trial.get('url')})\n"
            expander_content += "---\n"
        st.session_state.messages.append({
            "role": "assistant", "content": expander_content.strip('---\n'),
            "type": "expander", "title": f"View {len(st.session_state.trial_results)} Found Clinical Trial Details"
        })
    elif st.session_state.trial_results == []:
        st.session_state.messages.append({"role": "assistant", "content": f"_Note: No recruiting Phase 2+ interventional trials strongly matched the specific search terms and filters._"})

    advance_stage(STAGES["FINAL_SUMMARY"])
    st.rerun()


# Handle End Stage
if st.session_state.stage == STAGES["END"]:
     # Final message should have been added by advance_stage when reaching FINAL_SUMMARY
     # Add the end prompt here if necessary or just show the restart button
     if not st.session_state.messages or STAGES["END"] not in [STAGE_PROMPTS.get(m['content'], None) for m in st.session_state.messages]:
         end_prompt = STAGE_PROMPTS.get(STAGES["END"])
         if end_prompt: st.session_state.messages.append({"role": "assistant", "content": end_prompt})

     st.info("Session ended. Use the sidebar or button below to start a new exploration.")
     if st.button("🔄 Start New Exploration", key="restart_main"):
        keys_to_clear = ['stage', 'user_inputs', 'messages', 'drug_results', 'trial_results', 'consent_given']
        for key in keys_to_clear:
            if key in st.session_state: del st.session_state[key]
        st.rerun()

# --- Persistent Disclaimer Footer ---
# st.divider() # Keep divider? Maybe not if hiding disclaimer.
# --- Disclaimer Hidden as Requested ---
# st.markdown(
#     """
#     <div style="background-color: #1E1E1E; border-left: 6px solid #DC3545; padding: 15px; border-radius: 8px; margin-top: 25px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
#     <p style="font-weight: bold; color: #DC3545; font-size: 1.1em; margin-bottom: 10px;">⚠️ IMPORTANT DISCLAIMER:</p>
#     <p style="color: #EAEAEA; line-height: 1.6;">This tool is for informational purposes ONLY and is NOT a substitute for professional medical advice, diagnosis, or treatment. Information is sourced from public databases (FDA, ClinicalTrials.gov) and summarized by AI, which may contain inaccuracies or omissions. Drug effectiveness and trial eligibility are highly individual. <strong style="color: #FFC107;">ALWAYS consult your oncologist or qualified healthcare provider</strong> to discuss your specific situation and any information found here before making ANY health decisions.</p>
#     </div>
#     """,
#     unsafe_allow_html=True
# )