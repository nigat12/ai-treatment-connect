import os
from dotenv import load_dotenv
from groq import Groq, RateLimitError, APIError

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


def get_llm_client():
    """Initializes and returns the Groq client."""
    try:
        return Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print("error: ", e)
        # logging.critical(f"Failed to initialize Groq client: {e}", exc_info=True)
        # st.error("FATAL ERROR: Could not initialize AI client. Please check configuration.")
        # st.stop() # Halt app if client can't be initialized



def generate_ctgov_keywords_llm(diagnosis, stage_info, biomarkers):
    """Uses LLM to generate a clean JSON array of keywords and phrases for ClinicalTrials.gov query.term."""
    
    client = get_llm_client()
    prompt = f"""
    Take the following patient information and build clinicaltrials api query.titles search parameter 
    return very important keywords connected by term 'OR' 
    User Information:
    Diagnosis: "{diagnosis}"
    Stage/Progression: "{stage_info}"
    Biomarkers: "{biomarkers}"

    here is information for you 
    
    1. The primary cancer type (e.g., "lung cancer", "breast cancer"). Quote multi-word types.
    2. Relevant specific subtypes (e.g., "non-small cell", "adenocarcinoma"). 
    3. Significant biomarkers as quoted phrases or single terms (e.g., "EGFR mutation", "HER2 positive", "PD-L1").
    4. "metastatic" or related terms if the stage indicates spread.

    Exclude from the array:
    - Full sentences or questions.
    - Any introductory or explanatory text (like "Here is the list...", "Note that...", etc.).
    - Any conversational elements.

    Output ONLY the SEARCH PARAMETER VALUE
    AVOID STARTING OR ENDING SENTENCES 
    DONT COMMENT ON YOUR RESPONSE
    DONT EXPLAIN YOUR ANSWER 
    
    !IMPORTANT ONLY RESPOND THE SEARCH VALUES

    Example output: "Breast Cancer OR STAGE III"
    
    Output:
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are clinicalstrials api search parameter writer."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192", temperature=0.3, max_tokens=300,
            
        )
        response_content = chat_completion.choices[0].message.content
        # logging.info(f"LLM raw CT.gov keywords response: {response_content}")

        return response_content
    except Exception as e:
        print("Exception2: ", e)

print(generate_ctgov_keywords_llm("Non-small cell lung adenocarcinoma", "Stage IV, metastatic to liver and brain", "EGFR Exon 19 deletion positive"))