import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample data (replace with your Excel file path)
data = {
    'NCT Number': ['NCT05394337', 'NCT01203722', 'NCT04521413', 'NCT06150417', 'NCT04673175'],
    'Study Title': [
        'Neoadjuvant PD-1 Plus TIGIT Blockade in Patients With Cisplatin-Ineligible Operable High-Risk Urothelial Carcinoma',
        'Reduced Intensity, Partially HLA Mismatched BMT to Treat Hematologic Malignancies',
        'Safety and Efficacy Study of CFI-402411 in Subjects With Advanced Solid Malignancies',
        'MDRT in Prostate Cancer Treated With Long-term Androgen Deprivation Therapy in the STAMPEDE Trial (METANOVA)',
        'Ceftolozane-Tazobactam for Directed Treatment of Pseudomonas Aeruginosa Bacteremia and Pneumonia in Patients With Hematological Malignancies and Hematopoietic Stem Cell Transplantation'
    ]
}
df = pd.DataFrame(data)

# Uncomment the line below and replace 'path_to_file.xlsx' with your actual Excel file path
# df = pd.read_excel('path_to_file.xlsx')

# Function to extract condition from study title
def extract_condition(title):
    split_phrases = ["in Patients With", "for", "to Treat", "in Subjects With"]
    for phrase in split_phrases:
        if phrase in title:
            parts = title.split(phrase)
            return parts[-1].strip()
    return title  # Fallback to full title if no split phrase is found

# Extract conditions from study titles
conditions = df['Study Title'].apply(extract_condition).tolist()

# Load BioBERT model
print("Loading BioBERT model...")
model = SentenceTransformer('dmis-lab/biobert-base-cased-v1.1')

# Precompute embeddings for conditions
print("Computing embeddings for study conditions...")
condition_embeddings = model.encode(conditions)

# Function to find relevant studies based on user input
def find_relevant_studies(user_input, top_n=3):
    # Compute embedding for user input
    user_embedding = model.encode([user_input])[0]
    # Calculate cosine similarities
    similarities = cosine_similarity([user_embedding], condition_embeddings)[0]
    # Get indices of top matches
    top_indices = np.argsort(similarities)[::-1][:top_n]
    # Collect results
    results = []
    for idx in top_indices:
        study_title = df['Study Title'].iloc[idx]
        similarity = similarities[idx]
        results.append((study_title, similarity))
    return results

# Test the program with example queries
def test_program():
    test_queries = [
        "urothelial carcinoma",
        "prostate cancer",
        "hematologic malignancies",
        "stage IV bladder cancer"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("Top matching studies:")
        results = find_relevant_studies(query)
        for title, sim in results:
            print(f"  - {title} (Similarity: {sim:.4f})")

def test_program(find_relevant_studies):
    """
    Test the study matching program with various scenarios.
    Args:
        find_relevant_studies: Function that takes a query and returns [(title, similarity_score), ...]
    """
    test_queries = [
        "Non-small cell lung cancer type",  # Specific cancer type
        "urothelial carcinoma",             # Matches sample data NCT05394337
        "prostate cancer",                  # Matches sample data NCT06150417
        "hematologic malignancies",         # Matches sample data NCT01203722, NCT04673175
        "solid tumors",                     # Matches sample data NCT04521413
        "bladder cancer",                   # Related to urothelial carcinoma
        "lung cancer",                      # General cancer type
        "stage IV urothelial carcinoma",    # Stage-specific
        "PD-1 blockade",                    # Biomarker/treatment
        "advanced lung cancer with EGFR mutation",  # Multi-element query
        "metastatic breast cancer",         # Stage and cancer type
        "relapsed lymphoma",                # Progression state
        "leukemia",                         # No match in sample data
        "cancer",                           # Very general query
        "lung cancr",                       # Misspelling
        "pulmonary carcinoma"               # Synonym for lung cancer
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("Top matching studies:")
        try:
            results = find_relevant_studies(query)
            if results:
                for title, sim in results:
                    print(f"  - {title} (Similarity: {sim:.4f})")
            else:
                print("    No matches found.")
        except Exception as e:
            print(f"    Error processing query: {e}")

# Example usage (uncomment and replace with your actual function)
# from your_module import find_relevant_studies
# test_program(find_relevant_studies)
# Run the test
if __name__ == "__main__":
    test_program()