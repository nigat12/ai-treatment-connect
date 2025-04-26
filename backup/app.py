import streamlit as st
import requests
import os
from groq import Groq, RateLimitError, APIError
from dotenv import load_dotenv
import json
import re

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file

# --- Available Groq Models (Check Groq documentation for current availability) ---
AVAILABLE_MODELS = {
    "Llama 3 8B": "llama3-8b-8192",
    "Llama 3 70B": "llama3-70b-8192",
    "DeepSeek R1 Distill Llama 70B": "deepseek-r1-distill-llama-70b",
    "Gemma 2 Instruct": "gemma2-9b-it",
    "Mistral Saba 24B": "mistral-saba-24b",
}
DEFAULT_MODEL_DISPLAY_NAME = "Llama 3 8B"

# --- FDA API Functions ---
FDA_API_BASE_URL = "https://api.fda.gov/drug/label.json"

def fetch_fda_data(medication_name):
    """Fetches medication data from the openFDA API."""
    # Search for brand name or generic name - limit to 1 result for simplicity
    # Using 'openfda.brand_name' or 'openfda.generic_name' seems more reliable
    search_query = f'(openfda.brand_name:"{medication_name}" OR openfda.generic_name:"{medication_name}")'
    params = {
        'search': search_query,
        'limit': 1
    }
    headers = {'User-Agent': 'SafeMedAdvisor/1.0'} # Good practice to identify your app

    try:
        response = requests.get(FDA_API_BASE_URL, params=params, headers=headers, timeout=15) # Added timeout
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()

        if data.get('results') and len(data['results']) > 0:
            return data['results'][0] # Return the first matching result
        else:
            # Try searching within specific fields if generic search fails
            # This might be less reliable but worth a try
            search_query_alt = f'indications_and_usage:"{medication_name}" OR description:"{medication_name}"'
            params['search'] = search_query_alt
            response_alt = requests.get(FDA_API_BASE_URL, params=params, headers=headers, timeout=15)
            response_alt.raise_for_status()
            data_alt = response_alt.json()
            if data_alt.get('results') and len(data_alt['results']) > 0:
                 st.warning(f"Could not find exact match for '{medication_name}' by name. Found related information.")
                 return data_alt['results'][0]
            else:
                 return None # No results found

    except requests.exceptions.RequestException as e:
        st.error(f"Network or API error fetching FDA data: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Error decoding FDA API response.")
        return None

def clean_text(text):
    """Basic text cleaning: remove HTML tags and excessive whitespace."""
    if not text:
        return "Not available"
    # Simple regex to remove HTML tags, could be improved with libraries like BeautifulSoup if needed
    text = re.sub(r'<.*?>', '', text)
    # Replace multiple whitespace characters with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_relevant_info(fda_data):
    """Extracts and cleans key information from the FDA data."""
    if not fda_data:
        return None

    # Helper to safely get data (often fields are lists with one item)
    def get_field(data, key, default="Not available"):
        field_data = data.get(key)
        if isinstance(field_data, list) and len(field_data) > 0:
            # Join list items if multiple entries exist (e.g., multiple brand names)
            return ', '.join(map(clean_text, map(str, field_data)))
        elif isinstance(field_data, str):
             return clean_text(field_data)
        return default

    # Use the helper for openfda fields as well
    openfda_data = fda_data.get("openfda", {})

    info = {
        "brand_name": get_field(openfda_data, "brand_name"),
        "generic_name": get_field(openfda_data, "generic_name"),
        "manufacturer": get_field(openfda_data, "manufacturer_name"),
        "effective_time": fda_data.get("effective_time", "Not available"), # Usually label update date
        "indications_and_usage": get_field(fda_data, "indications_and_usage"),
        "dosage_and_administration": get_field(fda_data, "dosage_and_administration"),
        "contraindications": get_field(fda_data, "contraindications"),
        "warnings_and_cautions": get_field(fda_data, "warnings_and_cautions", get_field(fda_data, "warnings")), # Fallback to 'warnings'
        "adverse_reactions": get_field(fda_data, "adverse_reactions"),
        "drug_interactions": get_field(fda_data, "drug_interactions"),
    }
    # Format the effective_time date if available
    if info["effective_time"] != "Not available" and len(info["effective_time"]) >= 8:
         # Add basic validation
         if info["effective_time"].isdigit():
             info["effective_time"] = f"{info['effective_time'][0:4]}-{info['effective_time'][4:6]}-{info['effective_time'][6:8]}"
         else: # Keep original if not in expected YYYYMMDD format
            info["effective_time"] = fda_data.get("effective_time", "Not available")
    else:
        info["effective_time"] = "Not available"

    return info

# --- LLM Integration (Groq) ---
# Modified to accept the model name
def get_llm_summary(extracted_info, medication_name, model_id):
    """Generates a patient-friendly summary using Groq API with the specified model."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("Groq API key not found. Please set the GROQ_API_KEY environment variable.")
        return "Summary could not be generated due to missing API key."
    if not extracted_info:
        return "Summary cannot be generated as no FDA data was found."
    if not model_id:
         st.error("No LLM model selected for summary generation.")
         return "Summary generation failed: No model selected."

    client = Groq(api_key=api_key)

    # Prepare the context for the LLM (limit length to avoid exceeding token limits)
    context = f"""
    Medication Information:
    - Brand Name(s): {extracted_info.get('brand_name', 'N/A')}
    - Generic Name(s): {extracted_info.get('generic_name', 'N/A')}
    - Manufacturer: {extracted_info.get('manufacturer', 'N/A')}
    - Label Effective Date: {extracted_info.get('effective_time', 'N/A')}
    - Indications and Usage (What it's for): {extracted_info.get('indications_and_usage', 'Not available')[:1000]}...
    - Warnings and Cautions: {extracted_info.get('warnings_and_cautions', 'Not available')[:1500]}...
    - Adverse Reactions (Side Effects): {extracted_info.get('adverse_reactions', 'Not available')[:1500]}...
    - Contraindications (When not to use): {extracted_info.get('contraindications', 'Not available')[:1000]}...
    - Drug Interactions: {extracted_info.get('drug_interactions', 'Not available')[:1000]}...
    """

    prompt = f"""
    You are SafeMed Advisor, an AI assistant explaining medication information simply.
    Based *only* on the FDA data provided below for the medication '{medication_name}', generate a concise summary for a patient with no medical background.

    Focus on:
    1. What is this medication generally used for?
    2. What are some common or important side effects mentioned?
    3. Are there any major warnings, precautions, or situations where it shouldn't be used (contraindications)?
    4. Briefly mention any significant drug interactions if listed.
    5. Mention if key information (like usage or side effects) seems 'Not available' in the provided data.

    Keep the language very simple and direct. Use bullet points or short paragraphs.
    Start the summary directly. Do *not* add any information not present in the text below.
    Do *not* include a disclaimer saying this is not medical advice (it will be added separately).
    Limit the summary to about 100-150 words.

    FDA Data:
    {context}
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant that summarizes FDA medication data for patients in simple terms."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model_id, # Use the selected model ID
            temperature=0.3,
            max_tokens=300,
            top_p=0.9,
        )
        summary = chat_completion.choices[0].message.content
        # Optionally add which model generated the summary
        # summary += f"\n\n*(Summary generated using {model_id})*"
        return summary

    except RateLimitError:
        st.error(f"Groq API rate limit exceeded for model '{model_id}'. Please try again later or select a different model.")
        return "Summary generation failed due to rate limits."
    except APIError as e:
        st.error(f"Groq API error with model '{model_id}': {e}")
        return f"Summary generation failed due to an API error: {e.status_code}"
    except Exception as e:
        st.error(f"An unexpected error occurred during summary generation with model '{model_id}': {e}")
        return "Summary generation failed due to an unexpected error."


# --- Streamlit UI ---
st.set_page_config(page_title="SafeMed Advisor", layout="wide")
st.title("💊 SafeMed Advisor")
st.caption("Check medication information based on FDA data (Not a substitute for professional medical advice)")

# --- Input Area ---
col1, col2 = st.columns([3, 2]) # Adjust column ratios as needed

with col1:
    medication_name = st.text_input("Enter Medication Name (e.g., Metformin, Lipitor):", key="med_input")

with col2:
    # Get list of display names for the dropdown
    model_display_names = list(AVAILABLE_MODELS.keys())
    # Find the index of the default model for the dropdown
    default_index = model_display_names.index(DEFAULT_MODEL_DISPLAY_NAME) if DEFAULT_MODEL_DISPLAY_NAME in model_display_names else 0
    # Create the dropdown
    selected_model_display_name = st.selectbox(
        "Select AI Model for Summary:",
        options=model_display_names,
        index=default_index, # Set the default selection
        key="model_select"
    )
    # Get the actual model ID corresponding to the selected display name
    selected_model_id = AVAILABLE_MODELS[selected_model_display_name]


# --- Button and Processing Logic ---
if st.button("Get Medication Info", key="get_info_btn"):
    if not medication_name:
        st.warning("Please enter a medication name.")
    else:
        with st.spinner(f"Searching FDA database for '{medication_name}'..."):
            fda_raw_data = fetch_fda_data(medication_name)

        if fda_raw_data:
            st.success(f"Found information potentially related to '{medication_name}'.")

            with st.spinner("Extracting and cleaning data..."):
                extracted_info = extract_relevant_info(fda_raw_data)

            if extracted_info:
                # --- Display Extracted Data ---
                st.subheader("📋 Extracted FDA Information")
                st.markdown(f"**Brand Name(s):** {extracted_info['brand_name']}")
                st.markdown(f"**Generic Name(s):** {extracted_info['generic_name']}")
                st.markdown(f"**Manufacturer:** {extracted_info['manufacturer']}")
                st.markdown(f"**Label Effective Date:** {extracted_info['effective_time']}")

                with st.expander("What it's used for (Indications and Usage)"):
                    st.markdown(extracted_info['indications_and_usage'])
                with st.expander("Warnings and Cautions"):
                     st.markdown(extracted_info['warnings_and_cautions'])
                with st.expander("Side Effects (Adverse Reactions)"):
                     st.markdown(extracted_info['adverse_reactions'])
                with st.expander("When NOT to use (Contraindications)"):
                     st.markdown(extracted_info['contraindications'])
                with st.expander("Dosage and Administration"):
                    st.markdown(extracted_info['dosage_and_administration'])
                with st.expander("Drug Interactions"):
                    st.markdown(extracted_info['drug_interactions'])

                st.divider() # Visual separator

                # --- Generate and Display Summary ---
                st.subheader(f"🤖 AI Summary (using {selected_model_display_name})")
                # Use the selected model ID from the dropdown
                with st.spinner(f"Generating summary with {selected_model_display_name}..."):
                    llm_summary = get_llm_summary(extracted_info, medication_name, selected_model_id)

                st.markdown(llm_summary)

                # --- IMPORTANT DISCLAIMER ---
                st.divider()
                st.warning(
                    "**Disclaimer:** This tool provides information based on publicly available FDA data and AI summarization. "
                    "It is **not** a substitute for professional medical advice, diagnosis, or treatment. "
                    "Always consult your doctor or pharmacist regarding your specific health conditions and medications. "
                    "Do not disregard professional medical advice or delay seeking it because of something you have read here."
                )

            else:
                st.error("Could not extract relevant information from the fetched FDA data. The label format might be unexpected.")
        else:
            st.error(f"Could not find information for '{medication_name}' in the FDA database. Please check the spelling or try a different name (e.g., brand name vs. generic name).")

# Add some padding/space at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)