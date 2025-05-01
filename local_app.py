import streamlit as st
import pandas as pd
import numpy as np
import torch
import re
import math
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
CSV_FILENAME = 'drug_data.csv'
# Use a broader set of columns for initial text combination, cleaning will focus later
TEXT_COLUMNS_FOR_EMBEDDING = ['Cancer Type', 'Brief Study Summary', 'Drug Name'] # Consider including Drug Name for better matching if relevant
OUTCOME_COLUMNS = ['Treatment_OS', 'Control_OS', 'OS_Improvement (%)', 'Treatment_PFS', 'Control_PFS', 'PFS_Improvement (%)']

# Model Names
ST_MODEL_NAME = 'all-MiniLM-L6-v2' # General purpose Sentence Transformer
HF_MODEL_NAME = 'dmis-lab/biobert-v1.1' # Health-specific Hugging Face model

# --- HELPER FUNCTIONS ---

# Basic text cleaning
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s-]', '', text) # Remove special characters, except hyphens
        text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
        return text
    return ''

# Parse time strings (e.g., '18 months', '2.1 years') into months
def parse_time_to_months(time_str):
    if isinstance(time_str, (int, float)) and not math.isnan(time_str):
        return float(time_str) # Assume it's already in months if numeric
    if not isinstance(time_str, str):
        return None
    time_str = time_str.strip().lower()
    if time_str in ['n/a', 'not applicable', 'not reported', 'not reached', 'nr']:
        return None
    match = re.match(r'(\d+(\.\d+)?)\s*(month|year)s?', time_str)
    if match:
        value = float(match.group(1))
        unit = match.group(3)
        return value * 12 if unit == 'year' else value
    # Handle simple numbers as strings
    num_match = re.match(r'^(\d+(\.\d+)?)$', time_str)
    if num_match:
        return float(num_match.group(1))
    return None

# Parse percentage strings (e.g., '41.8%', 'NSS') into numerical percentage
def parse_improvement_percentage(perc_str):
    if isinstance(perc_str, (int, float)) and not math.isnan(perc_str):
         return float(perc_str) # Assume it's already a percentage float
    if not isinstance(perc_str, str):
        return None
    perc_str = perc_str.strip().lower()
    if perc_str in ['n/a', 'not applicable', 'not reported', 'not statistically significant', 'nss']:
        return None
    match = re.match(r'(\d+(\.\d+)?)\s*%', perc_str)
    if match:
        return float(match.group(1))
    num_match = re.match(r'^(\d+(\.\d+)?)$', perc_str)
    if num_match:
        return float(num_match.group(1))
    return None

# Custom sorting key to handle None values (push them to the end when sorting descending)
def sort_key_with_none(value, reverse=True):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        # For reverse=True (descending), None should be treated as smallest
        return float('-inf') if reverse else float('inf')
    return value

# --- MODEL LOADING (Cached) ---

@st.cache_resource
def load_sentence_transformer(model_name=ST_MODEL_NAME):
    print(f"Loading Sentence Transformer model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        print("Sentence Transformer model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading Sentence Transformer model '{model_name}': {e}")
        return None

@st.cache_resource
def load_huggingface_model(model_name=HF_MODEL_NAME):
    print(f"Loading Hugging Face model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval() # Set model to evaluation mode
        print("Hugging Face model and tokenizer loaded successfully.")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading Hugging Face model '{model_name}': {e}")
        return None, None

# --- DATA LOADING AND PREPROCESSING (Cached) ---

@st.cache_data # Cache the result of this function
def load_and_preprocess_data(filename):
    print(f"Loading and preprocessing data from {filename}...")
    try:
        df = pd.read_csv(filename)
        # Combine text columns for embedding input
        df['combined_text_for_embedding'] = df[TEXT_COLUMNS_FOR_EMBEDDING].fillna('').agg(' '.join, axis=1)
        df['combined_text_cleaned_for_embedding'] = df['combined_text_for_embedding'].apply(clean_text)

        # Parse Outcome Metrics numerically
        df['Treatment_OS_Months_Parsed'] = df['Treatment_OS'].apply(parse_time_to_months)
        df['Control_OS_Months_Parsed'] = df['Control_OS'].apply(parse_time_to_months)
        df['Calculated_OS_Improvement_Months'] = df.apply(
            lambda row: row['Treatment_OS_Months_Parsed'] - row['Control_OS_Months_Parsed']
            if pd.notna(row['Treatment_OS_Months_Parsed']) and pd.notna(row['Control_OS_Months_Parsed']) else None,
            axis=1
        )
        df['OS_Improvement_Percentage_Parsed'] = df['OS_Improvement (%)'].apply(parse_improvement_percentage)

        # Parse PFS (optional, but good for sorting/display)
        df['Treatment_PFS_Months_Parsed'] = df['Treatment_PFS'].apply(parse_time_to_months)
        df['Control_PFS_Months_Parsed'] = df['Control_PFS'].apply(parse_time_to_months)
        df['Calculated_PFS_Improvement_Months'] = df.apply(
            lambda row: row['Treatment_PFS_Months_Parsed'] - row['Control_PFS_Months_Parsed']
            if pd.notna(row['Treatment_PFS_Months_Parsed']) and pd.notna(row['Control_PFS_Months_Parsed']) else None,
            axis=1
        )
        df['PFS_Improvement_Percentage_Parsed'] = df['PFS_Improvement (%)'].apply(parse_improvement_percentage)


        print("Data loaded and preprocessed successfully.")
        return df.copy() # Return a copy to avoid modifying the cached object directly

    except FileNotFoundError:
        st.error(f"Error: {filename} not found. Please place the CSV file in the same directory as the script.")
        return None
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return None


# --- EMBEDDING GENERATION (Cached) ---

# Function for Sentence Transformer embedding
@st.cache_data # Cache embeddings
def generate_st_embeddings(_model, texts): # Use underscore for model arg to ensure cache invalidation if model changes conceptually
    print("Generating Sentence Transformer embeddings...")
    if _model is None or not texts:
        return None
    try:
        embeddings = _model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        print("Sentence Transformer embeddings generated.")
        return embeddings
    except Exception as e:
        st.error(f"Error generating Sentence Transformer embeddings: {e}")
        return None

# Function for Hugging Face embedding (Mean Pooling)
def get_hf_mean_pooling_embedding(text, _tokenizer, _model, device='cpu'): # Underscore args for cache safety if needed
    if _tokenizer is None or _model is None:
        return None
    # Ensure model is on the correct device
    _model.to(device)
    inputs = _tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _model(**inputs)
    last_hidden = outputs.last_hidden_state
    mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
    sum_embeddings = torch.sum(last_hidden * mask, dim=1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = (sum_embeddings / sum_mask).cpu().numpy()
    return mean_pooled[0] # Return the single embedding vector

@st.cache_data # Cache embeddings
def generate_hf_embeddings(_tokenizer, _model, texts, device='cpu'):
    print("Generating Hugging Face embeddings...")
    if _tokenizer is None or _model is None or not texts:
        return None
    try:
        embeddings = np.array([get_hf_mean_pooling_embedding(text, _tokenizer, _model, device) for text in texts])
        print("Hugging Face embeddings generated.")
        return embeddings
    except Exception as e:
        st.error(f"Error generating Hugging Face embeddings: {e}")
        return None


# --- CORE SEARCH FUNCTION ---

def find_relevant_drugs(df: pd.DataFrame, drug_embeddings: np.ndarray,
                        st_model: SentenceTransformer, hf_tokenizer, hf_model, # Pass models for user query embedding
                        selected_model_name: str,
                        user_cancer_type_raw: str, user_stage_raw: str, user_biomarkers_raw: str,
                        relevance_threshold: float, top_n: int, device: str):

    # Clean user inputs
    user_cancer_type_cleaned = clean_text(user_cancer_type_raw)
    user_stage_cleaned = clean_text(user_stage_raw)
    user_biomarkers_cleaned_list = [clean_text(b.strip()) for b in user_biomarkers_raw.split(',') if b.strip()]

    # Create the full user query string for embedding
    user_query_text = f"{user_cancer_type_cleaned} {user_stage_cleaned} {' '.join(user_biomarkers_cleaned_list)}"
    user_query_text = user_query_text.strip() # Ensure it's not just whitespace

    if not user_query_text:
         st.warning("Please enter some patient information (e.g., Cancer Type).")
         return []

    # Generate user query embedding based on the selected model
    user_embedding = None
    try:
        if selected_model_name == ST_MODEL_NAME and st_model:
            user_embedding = st_model.encode(user_query_text, convert_to_numpy=True)
        elif selected_model_name == HF_MODEL_NAME and hf_tokenizer and hf_model:
            user_embedding = get_hf_mean_pooling_embedding(user_query_text, hf_tokenizer, hf_model, device)
        else:
             st.error(f"Selected model '{selected_model_name}' is not available or loaded.")
             return []

        if user_embedding is None:
            st.error("Failed to generate embedding for the user query.")
            return []

    except Exception as e:
        st.error(f"Error generating user query embedding using {selected_model_name}: {e}")
        return []

    # Calculate Cosine Similarity
    # Ensure user_embedding is 2D for cosine_similarity
    if user_embedding.ndim == 1:
        user_embedding = user_embedding.reshape(1, -1)

    try:
        similarities = cosine_similarity(user_embedding, drug_embeddings)[0] # Get the similarity scores for the single user query
    except Exception as e:
        st.error(f"Error calculating cosine similarity: {e}")
        return []


    # --- Filtering and Ranking ---
    potential_results = []
    for index, row in df.iterrows():
        semantic_sim = similarities[index]

        # Filter by Relevance Threshold
        if semantic_sim >= relevance_threshold:
            potential_results.append({
                'index': index,
                'semantic_similarity': semantic_sim,
                # Include parsed numeric values for sorting
                'calculated_os_improvement_months': row['Calculated_OS_Improvement_Months'],
                'os_improvement_percentage_parsed': row['OS_Improvement_Percentage_Parsed'],
                'calculated_pfs_improvement_months': row['Calculated_PFS_Improvement_Months'], # Added PFS for potential sorting/display
                'pfs_improvement_percentage_parsed': row['PFS_Improvement_Percentage_Parsed'],
                # Include original data needed for display
                'Drug Name': row['Drug Name'],
                'Cancer Type': row['Cancer Type'],
                'Brief Study Summary': row['Brief Study Summary'],
                'Treatment_OS': row['Treatment_OS'],
                'Control_OS': row['Control_OS'],
                'OS_Improvement (%)': row['OS_Improvement (%)'],
                'Treatment_PFS': row['Treatment_PFS'],
                'Control_PFS': row['Control_PFS'],
                'PFS_Improvement (%)': row['PFS_Improvement (%)'],
            })

    # Advanced Sorting (apply this logic regardless of embedding model)
    potential_results.sort(key=lambda x: (
        x['semantic_similarity'],                              # 1: Semantic Similarity (desc)
        sort_key_with_none(x['calculated_os_improvement_months'], reverse=True), # 2: Calc OS diff months (desc, None last)
        sort_key_with_none(x['os_improvement_percentage_parsed'], reverse=True), # 3: OS % value (desc, None last)
        sort_key_with_none(x['calculated_pfs_improvement_months'], reverse=True),# 4: Calc PFS diff months (desc, None last) - Tie Breaker
        sort_key_with_none(x['pfs_improvement_percentage_parsed'], reverse=True) # 5: PFS % value (desc, None last) - Tie Breaker
    ), reverse=True) # Primary sort key (similarity) needs descending, hence reverse=True

    # Return top N results
    return potential_results[:top_n]


# --- STREAMLIT UI ---

st.set_page_config(layout="wide", page_title="Clinical Trial Drug Matcher")
st.title("💊 Clinical Trial Drug Matcher")
st.markdown("Enter patient details to find potentially relevant drugs based on clinical trial data.")

# --- Load Models and Data ---
# Determine device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
st.sidebar.caption(f"Using device: {DEVICE.upper()}")

# Load models using caching
st_model = load_sentence_transformer()
hf_tokenizer, hf_model = load_huggingface_model()

# Load and preprocess data using caching
df_processed = load_and_preprocess_data(CSV_FILENAME)

# --- Generate Embeddings (only if data loaded successfully) ---
drug_embeddings_st = None
drug_embeddings_hf = None
if df_processed is not None:
    texts_to_embed = df_processed['combined_text_cleaned_for_embedding'].tolist()
    # Generate ST embeddings
    if st_model:
        drug_embeddings_st = generate_st_embeddings(st_model, texts_to_embed)
    # Generate HF embeddings
    if hf_tokenizer and hf_model:
        drug_embeddings_hf = generate_hf_embeddings(hf_tokenizer, hf_model, texts_to_embed, DEVICE)

# --- Sidebar Inputs ---
st.sidebar.header("Patient Profile")
user_cancer_type = st.sidebar.text_input("Cancer Type:", "e.g., Non-Small Cell Lung Cancer")
user_stage = st.sidebar.text_input("Stage:", "e.g., Stage IV, metastatic")
user_biomarkers = st.sidebar.text_input("Biomarkers (comma-separated):", "e.g., EGFR mutation, PD-L1 positive")

st.sidebar.header("Search Configuration")
# Model Selection
available_models = []
if st_model and drug_embeddings_st is not None:
    available_models.append(ST_MODEL_NAME)
if hf_model and drug_embeddings_hf is not None:
    available_models.append(HF_MODEL_NAME)

if not available_models:
    st.sidebar.error("No embedding models could be loaded. Cannot perform search.")
    selected_model = None
else:
    # Default to BioBERT if available, otherwise the first available one
    default_model_index = available_models.index(HF_MODEL_NAME) if HF_MODEL_NAME in available_models else 0
    selected_model = st.sidebar.selectbox(
        "Select Embedding Model:",
        options=available_models,
        index=default_model_index,
        help="Choose the model used for understanding text meaning. BioBERT is specialized for biomedical text."
    )

# Other Configs
relevance_threshold = st.sidebar.slider(
    "Relevance Score Threshold:",
    min_value=0.0, max_value=1.0, value=0.5, step=0.05,
    help="Minimum semantic similarity score for a drug to be considered relevant."
)
top_n_results = st.sidebar.number_input(
    "Max Number of Results:",
    min_value=1, max_value=50, value=10, step=1,
    help="Maximum number of relevant drugs to display."
)

# --- Search Execution ---
search_button = st.sidebar.button("🔍 Find Relevant Drugs")

st.markdown("---") # Separator

if search_button and selected_model and df_processed is not None:
    st.subheader(f"Search Results (using {selected_model})")

    # Select the correct pre-computed embeddings
    if selected_model == ST_MODEL_NAME:
        drug_embeddings_to_use = drug_embeddings_st
    elif selected_model == HF_MODEL_NAME:
        drug_embeddings_to_use = drug_embeddings_hf
    else: # Should not happen if UI logic is correct
        st.error("Invalid model selected.")
        drug_embeddings_to_use = None

    if drug_embeddings_to_use is not None:
        with st.spinner("Searching..."):
            results = find_relevant_drugs(
                df=df_processed,
                drug_embeddings=drug_embeddings_to_use,
                st_model=st_model, # Pass loaded models for query embedding
                hf_tokenizer=hf_tokenizer,
                hf_model=hf_model,
                selected_model_name=selected_model,
                user_cancer_type_raw=user_cancer_type,
                user_stage_raw=user_stage,
                user_biomarkers_raw=user_biomarkers,
                relevance_threshold=relevance_threshold,
                top_n=top_n_results,
                device=DEVICE
            )

        if not results:
            st.info(f"No relevant drugs found matching the criteria (Threshold: {relevance_threshold:.2f}). Try adjusting the threshold or patient profile.")
        else:
            st.success(f"Found {len(results)} potentially relevant drugs:")
            for i, result in enumerate(results):
                with st.expander(f"**{i+1}. {result['Drug Name']}** (Relevance Score: {result['semantic_similarity']:.3f})"):
                    st.markdown(f"**Cancer Type (Trial):** {result['Cancer Type']}")
                    st.markdown(f"**Trial Summary:** {result['Brief Study Summary']}")

                    # Explanation Section (like script 1)
                    explanation_parts = []
                    explanation_parts.append(f"**Relevance Score (Semantic Sim):** {result['semantic_similarity']:.4f}")

                    # OS Info
                    os_imp_perc_orig = result.get('OS_Improvement (%)', 'N/A')
                    os_imp_months_calc = result.get('calculated_os_improvement_months')
                    os_imp_str = ""
                    if os_imp_months_calc is not None:
                        os_imp_str += f"{os_imp_months_calc:.2f} months difference " \
                                      f"(Treatment: {result.get('Treatment_OS', 'N/A')}, " \
                                      f"Control: {result.get('Control_OS', 'N/A')})"
                    elif os_imp_perc_orig not in ['N/A', None, '']:
                         os_imp_str += f"{os_imp_perc_orig}"
                    if os_imp_str:
                         explanation_parts.append(f"**OS Improvement:** {os_imp_str}")
                    else:
                         explanation_parts.append(f"**OS Data:** Treatment: {result.get('Treatment_OS', 'N/A')}, Control: {result.get('Control_OS', 'N/A')}")


                    # PFS Info
                    pfs_imp_perc_orig = result.get('PFS_Improvement (%)', 'N/A')
                    pfs_imp_months_calc = result.get('calculated_pfs_improvement_months')
                    pfs_imp_str = ""
                    if pfs_imp_months_calc is not None:
                         pfs_imp_str += f"{pfs_imp_months_calc:.2f} months difference " \
                                        f"(Treatment: {result.get('Treatment_PFS', 'N/A')}, " \
                                        f"Control: {result.get('Control_PFS', 'N/A')})"
                    elif pfs_imp_perc_orig not in ['N/A', None, '']:
                         pfs_imp_str += f"{pfs_imp_perc_orig}"
                    if pfs_imp_str:
                         explanation_parts.append(f"**PFS Improvement:** {pfs_imp_str}")
                    else:
                         explanation_parts.append(f"**PFS Data:** Treatment: {result.get('Treatment_PFS', 'N/A')}, Control: {result.get('Control_PFS', 'N/A')}")

                    # Display Explanation Bullet Points
                    st.markdown("**Key Metrics:**")
                    for part in explanation_parts:
                         st.markdown(f"- {part}")

                    # Optionally display raw row data for debugging/verification
                    # st.dataframe(df_processed.iloc[result['index']][['Drug Name'] + TEXT_COLUMNS_FOR_EMBEDDING + OUTCOME_COLUMNS])

    elif df_processed is not None:
        # This case means embeddings failed to generate for the selected model
        st.error(f"Could not generate embeddings using the selected model ({selected_model}). Cannot perform search.")

elif df_processed is None:
    st.error("Data could not be loaded. Please check the CSV file and script configuration.")
else:
    st.info("Enter patient details and click 'Find Relevant Drugs' in the sidebar.")