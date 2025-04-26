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

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
USER_DATA_FILE = "user_data.txt" # File to store user info (use with caution!)

# --- Available Groq Models ---
AVAILABLE_MODELS = {
    "Llama 3 8B": "llama3-8b-8192",
    "Mixtral 8x7B": "mixtral-8x7b-32768",
    "Gemma 7B": "gemma-7b-it",
}
DEFAULT_MODEL_DISPLAY_NAME = "Llama 3 8B"

# --- API Endpoints ---
FDA_API_BASE_URL = "https://api.fda.gov/drug/label.json"
CTGOV_API_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

# --- Chatbot Questions & Stages ---
# Added placeholders for clarity
QUESTIONS = [
    {"id": "intro", "text": "Hi there, I'm designed to use AI to help you explore information about cancer treatments alongside your physician. I am not a medical professional and this is not medical advice."},
    {"id": "diagnosis", "text": "Kindly share your diagnosis? (e.g., Non-small cell lung cancer, Invasive ductal carcinoma breast cancer). Share as much detail as you are comfortable with.", "type": "text_area", "placeholder": "Enter diagnosis details..."},
    {"id": "stage_progression", "text": "Any details on stage (e.g., Stage IV, metastatic), progression, and spread to lymph nodes or other locations?", "type": "text_area", "placeholder": "Enter stage, progression, spread details..."},
    {"id": "biomarkers", "text": "Any biomarker details? (e.g., ALK positive, EGFR Exon 19 deletion, KRAS G12C, ER positive, HER2 negative, BRCA mutation). Please list known markers.", "type": "text_area", "placeholder": "List known biomarkers, separated by commas if multiple..."},
    {"id": "prior_treatment", "text": "What treatments have you had to date? (e.g., Chemotherapy (Carboplatin/Pemetrexed), Surgery, Radiation, Immunotherapy (Pembrolizumab)).", "type": "text_area", "placeholder": "List prior treatments..."},
    {"id": "imaging_response", "text": "Have you had any imaging (CT, PET, MRI) or other studies showing changes in tumor size or progression recently?", "type": "text_area", "placeholder": "Describe recent imaging results or changes..."},
    # --- Processing & Consent Stages ---
    {"id": "show_drugs", "text": "Processing drug information..."}, # Internal processing trigger
    {"id": "ask_consent", "text": "Can we potentially share more information about therapeutics that may be relevant to your condition over email or phone?"}, # Explicit question
    {"id": "get_name", "text": "Please enter your First and Last Name:", "type": "text_input", "placeholder": "First Last"},
    {"id": "get_email", "text": "Please enter your Email Address:", "type": "text_input", "placeholder": "your.email@example.com"},
    {"id": "get_phone", "text": "Please enter your Phone Number (optional):", "type": "text_input", "placeholder": "e.g., 555-123-4567"},
    {"id": "show_trials", "text": "Processing clinical trial information..."}, # Internal processing trigger
    {"id": "end", "text": "Thank you. Remember to discuss all information with your healthcare provider."} # Final message
]

# --- Helper Functions --- (clean_text, save_user_data, API calls, LLM calls)
# Assume clean_text, save_user_data, API search functions, and LLM functions
# are defined here exactly as in the previous corrected version.
# (For brevity, they are omitted here, but they ARE needed for the full script)
# Make sure they include the necessary logging and error handling.

# --- Placeholder Functions (Replace with actual implementations from previous answer) ---
def clean_text(text):
    if not text: return "N/A"
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def save_user_data(data):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(USER_DATA_FILE, "a", encoding="utf-8") as f:
            f.write(f"--- Entry Start: {timestamp} ---\n")
            # Filter out sensitive items if not consented or needed
            items_to_save = {k: v for k, v in data.items() if k not in ['drug_results', 'trial_results']}
            for key, value in items_to_save.items():
                 f.write(f"{key}: {value}\n")
            f.write("--- Entry End ---\n\n")
        logging.info("User data saved successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to save user data: {e}")
        st.error("Could not save contact information due to a file system error.")
        return False

def search_fda_drugs_for_condition_and_markers(diagnosis, markers, limit=15):
    # *** Placeholder - Use the full function from the previous response ***
    logging.info(f"Simulating FDA search for: {diagnosis} / {markers}")
    # Example dummy result structure
    if "lung" in diagnosis.lower():
        return [{"brand_name": "Tagrisso", "generic_name": "Osimertinib", "indication_snippet": "Indicated for EGFR mutation positive non-small cell lung cancer..."}]
    return []

def search_filtered_clinical_trials(diagnosis, stage_info, markers, limit=20):
    # *** Placeholder - Use the full function from the previous response ***
    logging.info(f"Simulating Trial search for: {diagnosis} / {stage_info} / {markers}")
    # Example dummy result structure
    if "lung" in diagnosis.lower():
        return [{"nct_id": "NCT12345678", "title": "Trial of New Drug X in NSCLC", "status": "RECRUITING", "phase": "PHASE3", "study_type": "INTERVENTIONAL", "conditions": "NSCLC", "summary": "A study evaluating Drug X...", "contact": "Trial Contact", "locations_snippet": "City Hospital", "url": "#"}]
    return []

def generate_drug_summary_llm(diagnosis, markers, fda_drugs, model_id):
     # *** Placeholder - Use the full function from the previous response ***
     logging.info("Simulating cautious drug summary generation.")
     if fda_drugs:
         return f"Based on label searches for '{diagnosis}' and '{markers}', drugs like {fda_drugs[0]['generic_name']} were found. Discuss potential relevance and actual trial data with your doctor."
     return f"No specific drug labels matching '{diagnosis}' and '{markers}' were found in this search."

def generate_trial_summary_llm(diagnosis, markers, stage_info, clinical_trials, model_id):
     # *** Placeholder - Use the full function from the previous response ***
     logging.info("Simulating trial summary generation.")
     if clinical_trials:
         return f"Recruiting Phase 2+ interventional trials possibly related to '{diagnosis}' were found, like '{clinical_trials[0]['title']}'. Review details and discuss eligibility with your doctor."
     return f"No recruiting Phase 2+ interventional trials matching the search terms for '{diagnosis}' were found."
# --- End Placeholder Functions ---


# --- Streamlit App ---
st.set_page_config(page_title="Cancer Treatment Explorer", layout="wide") # Use wide layout
st.title("🔬 Cancer Treatment Explorer (AI-Assisted Info)")

# --- Sidebar for Model Selection ---
st.sidebar.subheader("AI Model Selection")
model_display_names = list(AVAILABLE_MODELS.keys())
# Find index of current or default model
current_model_id = st.session_state.get('model_id', AVAILABLE_MODELS[DEFAULT_MODEL_DISPLAY_NAME])
current_model_display_name = next((name for name, mid in AVAILABLE_MODELS.items() if mid == current_model_id), DEFAULT_MODEL_DISPLAY_NAME)
default_index = model_display_names.index(current_model_display_name)

selected_model_display_name = st.sidebar.selectbox(
    "Choose AI Model for Summaries:",
    options=model_display_names,
    index=default_index,
    key="model_select_widget"
)
# Update model_id in session state if changed
new_model_id = AVAILABLE_MODELS[selected_model_display_name]
if 'model_id' not in st.session_state or new_model_id != st.session_state.model_id:
    st.session_state.model_id = new_model_id
    # No automatic rerun needed unless model change should clear state

# --- Main Chat Area ---
chat_container = st.container() # Use a container for chat messages

# Initialize session state
if 'stage' not in st.session_state:
    st.session_state.stage = 1 # Start at the first question stage
    st.session_state.user_inputs = {}
    # Add only the intro message initially
    st.session_state.messages = [{"role": "assistant", "content": QUESTIONS[0]["text"]}]
    st.session_state.drug_results = None
    st.session_state.trial_results = None
    st.session_state.consent_given = None
    # model_id already handled above or set here if not in sidebar
    if 'model_id' not in st.session_state:
        st.session_state.model_id = AVAILABLE_MODELS[DEFAULT_MODEL_DISPLAY_NAME]

# Display chat history within the container
with chat_container:
    for i, msg in enumerate(st.session_state.messages):
        st.chat_message(msg["role"], avatar="🧑‍⚕️" if msg["role"] == "assistant" else "👤").write(msg["content"])


# --- Interaction Logic ---
current_stage_index = st.session_state.stage
current_question_data = QUESTIONS[current_stage_index] if current_stage_index < len(QUESTIONS) else None

# --- Handle Stages Requiring User Input ---
if current_question_data and current_question_data.get("type") in ["text_input", "text_area"]:
    question_id = current_question_data["id"]
    question_text = current_question_data["text"]
    placeholder = current_question_data.get("placeholder", "Enter your response...")
    input_type = current_question_data["type"]
    input_key = f"input_{question_id}"

    # Display the current question clearly before the input box
    st.chat_message("assistant", avatar="🧑‍⚕️").write(question_text)

    # Create a container for the input form elements
    with st.form(key=f"form_{question_id}", clear_on_submit=True):
        user_response = None
        if input_type == "text_area":
            user_response = st.text_area("Your response:", key=input_key, placeholder=placeholder, height=100, label_visibility="collapsed")
        else:
            user_response = st.text_input("Your response:", key=input_key, placeholder=placeholder, label_visibility="collapsed")

        submitted = st.form_submit_button("Submit Response")

        if submitted:
            if user_response or question_id == "get_phone": # Allow empty optional phone
                response_to_log = user_response if user_response else ("N/A" if question_id == "get_phone" else "")

                # Store response
                st.session_state.user_inputs[question_id] = response_to_log
                st.session_state.messages.append({"role": "user", "content": response_to_log}) # Log user response

                # Move to next stage
                st.session_state.stage += 1
                next_stage_index = st.session_state.stage

                # Special handling after collecting phone number
                if question_id == "get_phone":
                    # --- Save Data ---
                    contact_data = {
                        "Timestamp": datetime.now().isoformat(),
                        "Name": st.session_state.user_inputs.get("get_name"),
                        "Email": st.session_state.user_inputs.get("get_email"),
                        "Phone": st.session_state.user_inputs.get("get_phone"),
                        "ConsentGiven": st.session_state.consent_given,
                        "Diagnosis": st.session_state.user_inputs.get("diagnosis"),
                        "StageProgression": st.session_state.user_inputs.get("stage_progression"),
                        "Biomarkers": st.session_state.user_inputs.get("biomarkers"),
                        "PriorTreatment": st.session_state.user_inputs.get("prior_treatment"),
                        "ImagingResponse": st.session_state.user_inputs.get("imaging_response"),
                    }
                    if save_user_data(contact_data):
                         st.toast("Contact information recorded.", icon="✅") # Use toast for brief confirmation
                    else:
                         st.error("Failed to record contact information.")
                    # --- Move to Trials ---
                    st.session_state.stage = next((i for i, q in enumerate(QUESTIONS) if q["id"] == "show_trials"), len(QUESTIONS) -1)
                    # Don't add the "processing" message here, let the next block handle it
                # else:
                #     # If it's a regular question, prepare for the next one (handled by next loop iteration)
                #     pass # No need to add next question text here, loop will handle

                st.rerun() # Rerun to display next stage/question
            else:
                st.warning("Please provide the requested information.")

# --- Handle Consent Stage ---
elif current_question_data and current_question_data["id"] == "ask_consent":
    st.chat_message("assistant", avatar="🧑‍⚕️").write(current_question_data["text"])
    col1, col2, col3 = st.columns([1,1,5]) # Layout buttons
    with col1:
        if st.button("✔️ Yes", key="consent_yes", help="Agree to share contact info"):
            st.session_state.consent_given = True
            st.session_state.messages.append({"role": "user", "content": "Yes, I agree to share contact information."})
            st.session_state.stage += 1 # Move to get_name stage
            st.rerun()
    with col2:
       if st.button("❌ No", key="consent_no", help="Decline sharing contact info"):
           st.session_state.consent_given = False
           st.session_state.messages.append({"role": "user", "content": "No, I do not wish to share contact information now."})
           # Skip contact info stages, move directly to show_trials
           st.session_state.stage = next((i for i, q in enumerate(QUESTIONS) if q["id"] == "show_trials"), len(QUESTIONS) -1)
           st.rerun()

# --- Handle Processing Stages (Drugs & Trials) ---
elif current_question_data and current_question_data["id"] == "show_drugs":
    st.chat_message("assistant", avatar="🧑‍⚕️").info("Finding potentially relevant drugs based on your input...")
    with st.spinner("Searching FDA database and generating summary..."):
        diagnosis = st.session_state.user_inputs.get("diagnosis", "N/A")
        markers = st.session_state.user_inputs.get("biomarkers", "N/A")
        st.session_state.drug_results = search_fda_drugs_for_condition_and_markers(diagnosis, markers)
        drug_summary = generate_drug_summary_llm(diagnosis, markers, st.session_state.drug_results, st.session_state.model_id)

    # Add summary to messages
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"**Potentially Relevant Therapeutics to Discuss**\n\n{drug_summary}"
    })

    # Add expander content to messages (will be displayed by loop on rerun)
    if st.session_state.drug_results:
        expander_content = "**Found Drug Details (Based on Label Search):**\n\n"
        for i, drug in enumerate(st.session_state.drug_results):
            expander_content += f"**{i+1}. {drug['brand_name']} ({drug['generic_name']})**\n"
            expander_content += f"*Indication Snippet:* {drug.get('indication_snippet', 'N/A')}\n"
            if i < len(st.session_state.drug_results) - 1: expander_content += "---\n" # Use markdown separator
        st.session_state.messages.append({"role": "assistant", "content": expander_content, "type": "expander", "title": "See Found Drug Details"})

    elif st.session_state.drug_results == []:
         st.session_state.messages.append({"role": "assistant", "content": "No specific drug labels matched the search terms in the indication field."})

    # Move to next stage (ask consent)
    st.session_state.stage += 1
    st.rerun()

elif current_question_data and current_question_data["id"] == "show_trials":
    st.chat_message("assistant", avatar="🧑‍⚕️").info("Finding potentially relevant clinical trials based on your input and filters...")
    with st.spinner("Searching ClinicalTrials.gov and generating summary..."):
        diagnosis = st.session_state.user_inputs.get("diagnosis", "N/A")
        stage_info = st.session_state.user_inputs.get("stage_progression", "N/A")
        markers = st.session_state.user_inputs.get("biomarkers", "N/A")
        st.session_state.trial_results = search_filtered_clinical_trials(diagnosis, stage_info, markers)
        trial_summary = generate_trial_summary_llm(diagnosis, markers, stage_info, st.session_state.trial_results, st.session_state.model_id)

    # Add summary to messages
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"**Clinical Trial Information (Filtered Search)**\n\n{trial_summary}"
    })

     # Add expander content to messages
    if st.session_state.trial_results:
        expander_content = "**Found Clinical Trial Details (Recruiting, Phase 2+, Interventional):**\n\n"
        expander_content += '*Searched using diagnosis, markers, and "overall survival" keyword. Filters applied.*\n\n'
        for i, trial in enumerate(st.session_state.trial_results):
            expander_content += f"**{i+1}. {trial['nct_id']}: {trial['title']}**\n"
            expander_content += f"*Phase:* {trial.get('phase','N/A')} | *Status:* {trial.get('status','N/A')} | *Type:* {trial.get('study_type','N/A')}\n"
            expander_content += f"*Conditions:* {trial.get('conditions','N/A')}\n"
            expander_content += f"*Summary:* {trial.get('summary','N/A')}\n"
            expander_content += f"*Link:* [{trial.get('url','#')}]({trial.get('url','#')})\n"
            if i < len(st.session_state.trial_results) - 1: expander_content += "---\n"
        st.session_state.messages.append({"role": "assistant", "content": expander_content, "type": "expander", "title": "See Found Clinical Trial Details"})

    elif st.session_state.trial_results == []:
         st.session_state.messages.append({"role": "assistant", "content": f"No recruiting Phase 2+ interventional trials matching the search terms for '{diagnosis}' were found."})

    # Move to final stage
    st.session_state.stage += 1
    # Add final message immediately
    if st.session_state.stage < len(QUESTIONS):
         st.session_state.messages.append({"role": "assistant", "content": QUESTIONS[st.session_state.stage]["text"]})
    st.rerun()


# --- Handle End Stage ---
elif current_question_data and current_question_data["id"] == "end":
    # Final message is already displayed by the message loop
    if st.button("Start Over", key="restart"):
         # Clear state variables to restart
         st.session_state.stage = 1
         st.session_state.user_inputs = {}
         st.session_state.messages = [{"role": "assistant", "content": QUESTIONS[0]["text"]}] # Reset messages
         st.session_state.drug_results = None
         st.session_state.trial_results = None
         st.session_state.consent_given = None
         # Model selection persists unless reset explicitly
         st.rerun()

# --- Persistent Disclaimer ---
# Use markdown with some HTML for background color for emphasis
st.markdown("---")
st.markdown(
    """
    <div style="background-color: #262730; border-left: 6px solid #ff4b4b; padding: 10px; border-radius: 5px; margin-top: 20px;">
    <p style="font-weight: bold; color: #ff4b4b;">IMPORTANT DISCLAIMER:</p>
    <p style="color: #fafafa;">This tool is for informational purposes ONLY and is NOT a substitute for professional medical advice, diagnosis, or treatment. Information is from public databases and AI summaries, which may have inaccuracies. Drug effectiveness and trial eligibility are complex and individual. <strong>ALWAYS consult your oncologist or qualified healthcare provider</strong> to discuss your specific situation and any information found here before making ANY health decisions.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Debug Info (Optional) ---
with st.sidebar:
    st.subheader("Debug Info")
    st.write("Current Stage:", st.session_state.get('stage', 'N/A'))
    st.write("Consent Given:", st.session_state.get('consent_given', 'N/A'))
    st.write("User Inputs:", st.session_state.get('user_inputs', {}))