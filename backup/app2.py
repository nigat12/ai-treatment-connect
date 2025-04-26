import streamlit as st
import requests
import os
from groq import Groq, RateLimitError, APIError
from dotenv import load_dotenv
import json
import re
import urllib.parse # For URL encoding parameters
import logging

# --- Basic Logging Setup ---
# This helps in debugging if run locally or on platforms capturing stdout/stderr
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- Available Groq Models ---
AVAILABLE_MODELS = {
    "Llama 3 8B": "llama3-8b-8192",
    "Mixtral 8x7B": "mixtral-8x7b-32768",
    "Gemma 7B": "gemma-7b-it",
}
DEFAULT_MODEL_DISPLAY_NAME = "Llama 3 8B"

# --- API Endpoints ---
FDA_API_BASE_URL = "https://api.fda.gov/drug/label.json"
CTGOV_API_BASE_URL = "https://clinicaltrials.gov/api/v2/studies" # V2 API

# --- Helper Functions ---

def clean_text(text):
    """Basic text cleaning: remove HTML, normalize whitespace."""
    if not text:
        return "N/A"
    text = str(text) # Ensure it's a string
    # Remove HTML tags using regex (basic)
    text = re.sub(r'<[^>]+>', '', text)
    # Replace multiple whitespace characters (including newlines, tabs) with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    # Limit length for display sanity if needed (optional)
    # MAX_SNIPPET_LEN = 500
    # if len(text) > MAX_SNIPPET_LEN:
    #     text = text[:MAX_SNIPPET_LEN] + "..."
    return text

# --- FDA Drug Search Function ---
def search_fda_drugs_for_condition(condition, limit=15):
    """Searches FDA drug labels for indications matching the condition."""
    logging.info(f"Searching FDA for condition: {condition}")
    # Search within indications_and_usage field.
    search_query = f'indications_and_usage:"{condition}"'
    params = {
        'search': search_query,
        'limit': limit
    }
    headers = {'User-Agent': 'CancerTrialFinder/1.0'}

    try:
        response = requests.get(FDA_API_BASE_URL, params=params, headers=headers, timeout=20) # Increased timeout
        response.raise_for_status() # Check for 4xx/5xx errors
        data = response.json()

        drugs = []
        if data.get('results'):
            seen_generics = set() # Simple deduplication
            logging.info(f"Found {len(data['results'])} potential drug labels from FDA.")
            for result in data['results']:
                openfda_data = result.get('openfda', {})
                # Handle list or string for names
                brand_names_raw = openfda_data.get('brand_name', ['N/A'])
                generic_names_raw = openfda_data.get('generic_name', ['N/A'])
                brand_names = brand_names_raw if isinstance(brand_names_raw, list) else [brand_names_raw]
                generic_names = generic_names_raw if isinstance(generic_names_raw, list) else [generic_names_raw]

                indication_raw = result.get('indications_and_usage', ['N/A'])
                indication = clean_text(indication_raw[0] if isinstance(indication_raw, list) and indication_raw else indication_raw)

                # Basic deduplication based on the primary generic name
                generic_key = generic_names[0].lower() if generic_names[0] != 'N/A' else None
                if generic_key and generic_key in seen_generics:
                    continue
                if generic_key:
                    seen_generics.add(generic_key)

                # Limit indication length for context/display
                indication_snippet = indication[:500] + "..." if len(indication) > 500 else indication

                drugs.append({
                    "brand_name": clean_text(', '.join(brand_names)),
                    "generic_name": clean_text(', '.join(generic_names)),
                    "indication_snippet": indication_snippet,
                })
            logging.info(f"Returning {len(drugs)} unique drugs after deduplication.")
            return drugs
        else:
            logging.info("No drug results found from FDA.")
            return [] # Return empty list if no results

    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error fetching FDA drug data: {e}")
        st.error(f"Error fetching FDA data: {e.response.status_code}. The FDA server may be busy or the query invalid.")
        return None # Indicate error
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error fetching FDA drug data: {e}")
        st.error(f"Network error connecting to FDA API: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding error for FDA response: {e}")
        st.error("Error processing FDA API response.")
        return None
    except Exception as e:
        logging.error(f"Unexpected error in FDA search: {e}", exc_info=True)
        st.error("An unexpected error occurred while searching for drugs.")
        return None


# --- ClinicalTrials.gov Search Function (V2 API) ---
def search_clinical_trials(condition, limit=15):
    """Searches ClinicalTrials.gov V2 API for active, recruiting trials."""
    logging.info(f"Searching ClinicalTrials.gov for condition: {condition}")
    params = {
        'query.term': condition,
        'filter.overallStatus': "RECRUITING",
        'pageSize': limit,
        'fields': "NCTId,BriefTitle,OverallStatus,Condition,BriefSummary,CentralContactName,CentralContactPhone,LocationFacility"
    }
    headers = {'User-Agent': 'CancerTrialFinder/1.0', 'Accept': 'application/json'}

    try:
        query_string = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
        full_url = f"{CTGOV_API_BASE_URL}?{query_string}"
        # logging.debug(f"ClinicalTrials.gov request URL: {full_url}") # Use debug level

        response = requests.get(full_url, headers=headers, timeout=30) # Generous timeout
        response.raise_for_status() # Check for 4xx/5xx errors
        data = response.json()

        trials = []
        if data.get('studies'):
            logging.info(f"Found {len(data['studies'])} potential trials from ClinicalTrials.gov.")
            for study_wrapper in data['studies']:
                 # --- Start: Parsing Logic (V2 Structure) ---
                 study = study_wrapper.get('protocolSection', {})
                 if not study: continue

                 ident_module = study.get('identificationModule', {})
                 status_module = study.get('statusModule', {})
                 desc_module = study.get('descriptionModule', {})
                 cond_module = study.get('conditionsModule', {})
                 contacts_module = study.get('contactsLocationsModule', {})
                 central_contacts = contacts_module.get('centralContacts', [])
                 locations = contacts_module.get('locations', [])

                 # Extract and format contact info
                 contact_info = "Contact details not listed"
                 if central_contacts:
                     cc = central_contacts[0]
                     cc_name = cc.get('name', '')
                     cc_phone = cc.get('phone', '')
                     cc_email = cc.get('email', '')
                     contact_parts = [part for part in [cc_name, cc_phone, cc_email] if part]
                     if contact_parts:
                        contact_info = f"{cc_name} ({', '.join(filter(None, [cc_phone, cc_email]))})" if cc_name and (cc_phone or cc_email) else ', '.join(contact_parts)

                 # Extract location snippet
                 location_names = [loc.get('facility') for loc in locations[:3] if loc.get('facility')] # Get first 3 valid facility names
                 location_str = ", ".join(location_names) if location_names else ""
                 if not location_str and locations:
                     location_str = f"{len(locations)} locations (details in link)"
                 elif not locations:
                     location_str = "See study record for locations"

                 trial_nct_id = ident_module.get('nctId', '')
                 trial_url = f"https://clinicaltrials.gov/study/{trial_nct_id}" if trial_nct_id else "#"

                 trials.append({
                     "nct_id": trial_nct_id if trial_nct_id else 'N/A',
                     "title": clean_text(ident_module.get('briefTitle', 'N/A')),
                     "status": status_module.get('overallStatus', 'N/A'),
                     "conditions": clean_text(', '.join(cond_module.get('conditions', ['N/A']))),
                     "summary": clean_text(desc_module.get('briefSummary', 'N/A')[:500] + "..."), # Limit summary
                     "contact": clean_text(contact_info),
                     "locations_snippet": location_str,
                     "url": trial_url
                 })
            logging.info(f"Returning {len(trials)} parsed trials.")
            return trials
        else:
            logging.info("No trial results found from ClinicalTrials.gov.")
            return [] # Return empty list if no results

    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error fetching ClinicalTrials.gov data: {e}")
        st.error(f"Error fetching Clinical Trials data: Status {e.response.status_code}.")
        try:
            error_details = e.response.json()
            st.error(f"API Error Message: {error_details.get('message', e.response.text[:500])}") # Try to show specific message
        except json.JSONDecodeError:
             st.error(f"Response content (non-JSON): {e.response.text[:500]}...")
        return None # Indicate error
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error fetching ClinicalTrials.gov data: {e}")
        st.error(f"Network error connecting to ClinicalTrials.gov API: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding error for ClinicalTrials.gov response: {e}")
        st.error("Error processing ClinicalTrials.gov API response.")
        # logging.debug(f"Response text causing decode error: {response.text[:1000]}") # Keep for debugging
        return None
    except Exception as e:
        logging.error(f"Unexpected error in ClinicalTrials.gov search: {e}", exc_info=True)
        st.error("An unexpected error occurred while searching for clinical trials.")
        return None

# --- LLM Summary Function ---
def generate_patient_summary_llm(cancer_type, fda_drugs, clinical_trials, model_id):
    """Generates a patient-friendly summary using Groq API."""
    if not GROQ_API_KEY:
        st.error("Groq API key not found. Please set the GROQ_API_KEY environment variable.")
        return "Summary could not be generated due to missing API key."
    if fda_drugs is None and clinical_trials is None:
        return "Could not generate summary because initial data fetching failed."
    if not fda_drugs and not clinical_trials:
         return "No specific drug labels or active clinical trials were found matching your query based on the available data."

    client = Groq(api_key=GROQ_API_KEY)
    logging.info(f"Generating summary using Groq model: {model_id}")

    # --- Prepare Context (Limited) ---
    drug_context = "Potential Medications Found (Based on Label Indications):\n"
    if fda_drugs:
        for drug in fda_drugs[:5]: # Limit context size
            drug_context += f"- {drug['brand_name']} ({drug['generic_name']}): Indicated for conditions including... '{drug['indication_snippet']}'\n"
        if len(fda_drugs) > 5: drug_context += "- ... (more drugs found, see details below)\n"
    else:
        drug_context += "- No drug labels found directly matching the indication query in the FDA database.\n"

    trial_context = "\nActive & Recruiting Clinical Trials Found (from ClinicalTrials.gov):\n"
    if clinical_trials:
        for trial in clinical_trials[:5]: # Limit context size
            trial_context += f"- {trial['nct_id']}: {trial['title']}\n   Summary Snippet: {trial['summary']}\n"
        if len(clinical_trials) > 5: trial_context += "- ... (more trials found, see details below)\n"
    else:
        trial_context += f"- No active, recruiting trials found matching '{cancer_type}' on ClinicalTrials.gov at this time.\n"

    # --- Construct Prompt ---
    prompt = f"""
    You are an AI assistant helping patients understand potential cancer treatment avenues based on public data about '{cancer_type}'.
    Your response should be based *strictly* on the information provided below.

    **Instructions for Summary:**
    1.  **Introduction:** Briefly state that the following is a summary based on FDA drug labels and active clinical trials found for '{cancer_type}'.
    2.  **Potential Drugs:** Mention the drugs found (if any). Use cautious language. **DO NOT state or imply they improve overall survival or are suitable for the patient.** Phrase like: "Some potentially relevant drugs found based on their approved uses include [Drug A, Drug B]. Drugs approved for conditions like '{cancer_type}' are sometimes studied for specific outcomes, but effectiveness depends on many factors." If no drugs were found, clearly state that.
    3.  **Clinical Trials:** Mention the recruiting clinical trials found (if any). Phrase like: "There are actively recruiting clinical trials related to '{cancer_type}' listed on ClinicalTrials.gov, such as [Trial Title A, Trial Title B]." Briefly mention what they might be investigating if discernible from the title/summary snippet. If none were found, clearly state that.
    4.  **Actionable Advice:** Conclude by strongly advising the patient to discuss these findings with their oncologist or healthcare provider, as this information is general and not personalized medical advice.
    5.  **Tone & Format:** Use simple, clear language. Use bullet points or short paragraphs. Keep the total summary concise (around 150-250 words).
    6.  **Constraint:** Do *not* add any information, speculation, or medical opinions beyond interpreting the provided data snippets as instructed.

    **Provided Data Snippets:**
    {drug_context}
    {trial_context}

    **Generate the patient-friendly summary now:**
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You summarize potential cancer drug and clinical trial information for patients simply, cautiously, and based *only* on provided data snippets. You emphasize that this is not medical advice and discussion with a doctor is essential."},
                {"role": "user", "content": prompt}
            ],
            model=model_id,
            temperature=0.2, # Lower temperature for more factual, less creative summary
            max_tokens=500,
            top_p=0.9,
        )
        summary = chat_completion.choices[0].message.content
        logging.info("Successfully generated summary from Groq.")
        return summary

    except RateLimitError:
        logging.warning(f"Groq API rate limit exceeded for model '{model_id}'.")
        st.error(f"AI summary generation failed due to rate limits. Please try again later.")
        return "Summary generation failed due to temporary rate limits."
    except APIError as e:
        logging.error(f"Groq API error with model '{model_id}': {e}")
        st.error(f"AI summary generation failed due to an API error: {e.status_code}")
        return f"Summary generation failed due to an API error."
    except Exception as e:
        logging.error(f"Unexpected error during summary generation: {e}", exc_info=True)
        st.error("An unexpected error occurred during summary generation.")
        return "Summary generation failed unexpectedly."


# --- Streamlit UI ---
st.set_page_config(page_title="Cancer Trial & Drug Finder", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for disclaimer emphasis
st.markdown("""
<style>
.stAlert > div > div > p[data-testid="stMarkdownContainer"] {
    font-weight: bold;
}
.disclaimer-error .stAlert p {
     font-size: 1.1em !important;
     font-weight: bold !important;
     color: #8B0000 !important; /* Dark Red */
}
</style>
""", unsafe_allow_html=True)


st.title("🔬 Cancer Clinical Trial & Drug Finder (Experimental)")
st.caption("Find potential drugs (based on FDA labels) and recruiting trials (from ClinicalTrials.gov).")
st.markdown("---")


# --- Input Area ---
st.subheader("1. Describe the Condition")
cancer_type = st.text_input(
    "Cancer Type or Disease Area:",
    key="cancer_input",
    placeholder="e.g., metastatic lung cancer, glioblastoma, HER2+ breast cancer",
    help="Enter the condition you want to search for. Be specific for better results."
)

st.subheader("2. Contact Information (Optional Simulation)")
email_address = st.text_input(
    "Email Address:",
    key="email_input",
    placeholder="your.email@example.com",
    help="This simulates capturing contact info. Your email is NOT stored or used."
)

st.subheader("3. Acknowledge Disclaimer")
consent = st.checkbox(
    "I understand this tool provides general information from public databases and AI summaries. It is NOT medical advice. I MUST discuss any findings with my doctor or healthcare provider before making any decisions.",
    key="consent_check",
    value=False # Default to unchecked
)

# --- Model Selection ---
st.subheader("4. Select AI Model for Summary")
model_display_names = list(AVAILABLE_MODELS.keys())
default_index = model_display_names.index(DEFAULT_MODEL_DISPLAY_NAME) if DEFAULT_MODEL_DISPLAY_NAME in model_display_names else 0
selected_model_display_name = st.selectbox(
    "Choose AI Model:",
    options=model_display_names,
    index=default_index,
    key="model_select"
)
selected_model_id = AVAILABLE_MODELS[selected_model_display_name]

# --- Button and Processing Logic ---
st.markdown("---") # Visual separator
search_pressed = st.button("Search for Information", key="search_btn", type="primary", use_container_width=True)

if search_pressed:
    validation_passed = True
    if not cancer_type:
        st.warning("⚠️ Please enter a cancer type or disease area to search.")
        validation_passed = False
    if not consent:
        st.warning("⚠️ Please read and check the acknowledgment box to proceed.")
        validation_passed = False

    if validation_passed:
        st.info(f"🚀 Searching for information related to: **{cancer_type}**")
        if email_address:
            st.success(f"✉️ Email captured (simulation): **{email_address}** (Not stored/used).")

        drugs_found = None
        trials_found = None
        api_error_occurred = False

        # --- Perform Searches ---
        with st.spinner("Searching FDA drug database... (may take a few seconds)"):
            drugs_found = search_fda_drugs_for_condition(cancer_type)
            if drugs_found is None: api_error_occurred = True

        # Only proceed if the first API call didn't fail critically
        if not api_error_occurred:
            with st.spinner("Searching ClinicalTrials.gov for recruiting trials... (may take a few seconds)"):
                trials_found = search_clinical_trials(cancer_type)
                if trials_found is None: api_error_occurred = True

        # --- Display Results and Summary ---
        if api_error_occurred:
            st.error("❌ One or more API searches failed. Cannot proceed with summary or full results.")
        else:
            # Generate and Display Summary
            st.subheader(f"📊 AI-Generated Summary (using {selected_model_display_name})")
            with st.spinner(f"Generating summary with {selected_model_display_name}..."):
                llm_summary = generate_patient_summary_llm(cancer_type, drugs_found, trials_found, selected_model_id)
            st.markdown(llm_summary) # Display summary from LLM

            st.markdown("---")

            # --- Display Detailed Findings ---
            st.subheader("📋 Detailed Findings")

            # Drugs Section
            with st.expander("**Potential Drugs (Based on Label Indication Search)**", expanded=False):
                if drugs_found:
                    st.caption("Note: Inclusion here is based on matching text in the drug's FDA label. It does NOT guarantee effectiveness or suitability for an individual. Discuss with your doctor.")
                    for i, drug in enumerate(drugs_found):
                        st.markdown(f"**{i+1}. {drug['brand_name']} ({drug['generic_name']})**")
                        st.markdown(f"    *Indication Snippet:* {drug['indication_snippet']}")
                        if i < len(drugs_found) - 1: st.markdown("---") # Separator between drugs
                elif drugs_found == []: # Explicitly check for empty list (successful search, no results)
                     st.info("No specific drugs found matching the indication query in the FDA label database during this search.")
                # else: handled by api_error_occurred check

            # Trials Section
            with st.expander("**Active & Recruiting Clinical Trials (from ClinicalTrials.gov)**", expanded=False):
                 if trials_found:
                    st.caption("Details sourced from ClinicalTrials.gov. Status: Recruiting.")
                    for i, trial in enumerate(trials_found):
                        st.markdown(f"**{i+1}. {trial['nct_id']}: {trial['title']}**")
                        st.markdown(f"    *Status:* {trial['status']}")
                        st.markdown(f"    *Conditions:* {trial['conditions']}")
                        st.markdown(f"    *Summary:* {trial['summary']}")
                        st.markdown(f"    *Contact:* {trial['contact']}")
                        st.markdown(f"    *Locations (Sample):* {trial['locations_snippet']}")
                        st.markdown(f"    *Link:* [{trial['url']}]({trial['url']})")
                        if i < len(trials_found) - 1: st.markdown("---") # Separator between trials
                 elif trials_found == []: # Explicitly check for empty list
                     st.info(f"No active, recruiting trials found matching '{cancer_type}' on ClinicalTrials.gov during this search.")
                 # else: handled by api_error_occurred check

        # --- FINAL IMPORTANT DISCLAIMER ---
        st.markdown("---")
        # Use custom class for emphasis
        st.markdown('<div class="disclaimer-error">', unsafe_allow_html=True)
        st.error(
            "** VERY IMPORTANT DISCLAIMER:** This tool is for informational purposes ONLY and is NOT a substitute for professional medical advice, diagnosis, or treatment. Information is sourced from public databases (FDA, ClinicalTrials.gov) and summarized by AI, which may contain inaccuracies or omissions. Drug effectiveness and trial eligibility are highly individual. **ALWAYS consult your oncologist or qualified healthcare provider** to discuss your specific situation and any information found here before making any health decisions. Do not delay seeking professional advice based on this tool's output."
        )
        st.markdown('</div>', unsafe_allow_html=True)


# Add some padding/space at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)