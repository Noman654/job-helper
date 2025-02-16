from fastapi import FastAPI
import re
import json
import logging
from typing import List

# For document processing
import docx2txt
import PyPDF2
from pathlib import Path

# For Gemini integration
import google.generativeai as genai
import time

from models import LLMMode, Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Gemini client
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = "AIzaSyDD9_mQ2i_J9EG4Zgyl0IYZMmweO-RjubA"


if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found. LLM features will be limited.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Constants
DEFAULT_MODEL =  "gemini-2.0-flash"
PREMIUM_MODEL = "gemini-2.0-pro-exp-02-05"
MAX_TOKENS = 30000  # Gemini's context window is larger than GPT's
TOKEN_BUFFER = 2000  # Buffer to leave room for response


app = FastAPI(
    title="Advanced Resume Ranking System with Gemini",
    description="API for extracting criteria from job descriptions and scoring resumes with Gemini integration",
    version="2.0.0"
)

# Store the job scores for later retrieval
job_scores_store = {}

# Function to get default settings
def get_default_settings() -> Settings:
    return Settings()

# Simple token estimator for Gemini
def num_tokens_from_string(string: str, model_name: str = DEFAULT_MODEL) -> int:
    """Returns an estimated number of tokens in a text string."""
    # Gemini doesn't have a specific tokenizer exposed, so we'll use a rough estimate
    # Average English word is ~4 characters, and tokens are ~4 characters on average
    return len(string) // 4

def get_llm_model(settings: Settings):
    """Return the appropriate model based on settings."""
    if settings.llm_mode == LLMMode.DISABLED:
        return None
    elif settings.llm_mode == LLMMode.PREMIUM:
        return PREMIUM_MODEL
    else:
        return DEFAULT_MODEL

async def call_llm_api(prompt: str, model_name: str = DEFAULT_MODEL, max_tokens: int = 1000):
    """Call the Gemini API with retry logic and error handling."""
    if not GEMINI_API_KEY:
        logger.error("Cannot call LLM: API key not set")
        return None
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            # Configure the model
            generation_config = {
                "temperature": 0.2,  # Lower temperature for more consistent results
                "max_output_tokens": max_tokens,
                "top_p": 0.95,
                "top_k": 0,
            }
            
            # Get the model
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                system_instruction="You are a helpful assistant specializing in HR, recruitment, and resume analysis."
            )
            
            # Generate content
            response = model.generate_content(prompt)
            
            json_text = re.sub(r"```json|```", "", response.text.strip()).strip()
            return json_text
            # return response.text.strip()
            
        except Exception as e:
            logger.warning(f"LLM API call failed (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"All LLM API call attempts failed: {str(e)}")
                return None

def extract_text_from_file(file_path, filename):
    """Extract text from PDF or DOCX file."""
    text = ""
    
    try:
        if filename.lower().endswith('.pdf'):
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        
        elif filename.lower().endswith('.docx'):
            text = docx2txt.process(file_path)
        
        else:
            raise ValueError("Unsupported file format. Please upload PDF or DOCX.")
    
    except Exception as e:
        logger.error(f"Error extracting text from {filename}: {str(e)}")
        raise ValueError(f"Could not process file {filename}: {str(e)}")
    
    # Clean up text - remove extra whitespace, normalize line breaks
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

async def extract_criteria_with_llm(text: str, settings: Settings):
    """Use LLM to extract job criteria."""
    model = get_llm_model(settings)
    if not model:
        return extract_criteria_rule_based(text)
    
    # Check if text is too long and truncate if necessary
    token_count = num_tokens_from_string(text, model)
    if token_count > MAX_TOKENS - TOKEN_BUFFER:
        logger.warning(f"Text too long ({token_count} tokens), truncating to fit token limit")
        # Simple truncation - in production you might want to use more sophisticated methods
        text = text[:4 * (MAX_TOKENS - TOKEN_BUFFER)]  # Rough conversion from tokens to chars
    
    prompt = f"""
    You are an expert HR recruiter. Analyze the following job description and extract the key requirements and criteria that would be used to evaluate candidates. 
    
    Format your response as a JSON list of strings, with each string representing one criterion. Focus on:
    - Required skills, technologies, and tools
    - Years of experience required
    - Education requirements
    - Certifications
    - Soft skills and personal qualities
    - Any other important qualifications
    
    Job Description:
    {text}
    
    Return ONLY a valid JSON list of strings, no other text.
    """
    
    response = await call_llm_api(prompt, model)
    
    if not response:
        logger.warning("LLM criteria extraction failed, falling back to rule-based extraction")
        return extract_criteria_rule_based(text)
    
    try:
        criteria_list = json.loads(response)
        if isinstance(criteria_list, list) and all(isinstance(item, str) for item in criteria_list):
            return criteria_list
        else:
            logger.warning("LLM returned invalid format for criteria, falling back to rule-based extraction")
            return extract_criteria_rule_based(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM response as JSON, falling back to rule-based extraction")
        return extract_criteria_rule_based(text)

def extract_criteria_rule_based(text):
    """Extract key criteria from job description text using rule-based approach."""
    criteria = []
    
    # Look for common patterns in job descriptions
    patterns = [
        # Experience
        r'(\d+\+?\s+years?.*?experience.*?(?:in|with).*?)[\.\n]',
        # Certification
        r'(certification.*?in.*?|.*?certificate.*?)[\.\n]',
        # Education
        r'((?:bachelor|master|phd|doctorate|degree).*?in.*?)[\.\n]',
        # Skills
        r'((?:proficiency|proficient|experience|skilled).*?in.*?)[\.\n]',
        # Requirements
        r'((?:must|should).*?have.*?)[\.\n]',
        # Knowledge
        r'(knowledge of.*?)[\.\n]',
        # Familiar with
        r'(familiar with.*?)[\.\n]',
        # Understanding of
        r'(understanding of.*?)[\.\n]'
    ]
    
    # Extract matches
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        criteria.extend(match.strip() for match in matches)
    
    # Clean up
    criteria = list(set(criteria))
    criteria = [c for c in criteria if len(c.split()) > 2]  # Filter out too short criteria
    
    return criteria

async def score_resume_with_llm(resume_text: str, criteria: List[str], settings: Settings):
    """Score a resume against criteria using LLM."""
    model = get_llm_model(settings)
    if not model:
        return score_resume_keyword_match(resume_text, criteria)
    
    # Prepare criteria list as a numbered list for the prompt
    criteria_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(criteria)])
    
    # Create a prompt with truncated resume if needed
    token_count = num_tokens_from_string(resume_text, model)
    if token_count > MAX_TOKENS - TOKEN_BUFFER:
        logger.warning(f"Resume too long ({token_count} tokens), truncating for LLM analysis")
        resume_text = resume_text[:4 * (MAX_TOKENS - TOKEN_BUFFER)]  # Rough conversion from tokens to chars
    
    prompt = f"""
    You are an expert HR recruiter. Evaluate the following resume against these job criteria. 
    Score each criterion on a scale of 0-5, where:
    0 = Not mentioned at all
    1 = Briefly mentioned without details
    2 = Some relevant experience/skills
    3 = Good match but not complete
    4 = Strong match
    5 = Perfect match

    Job Criteria:
    {criteria_text}
    
    Resume:
    {resume_text}
    
    Return ONLY a valid JSON object where keys are the criterion numbers (1, 2, 3, etc.) and values are the scores (0-5).
    Example: {{"1": 4, "2": 2, "3": 5}}
    """
    
    response = await call_llm_api(prompt, model)
    
    if not response:
        logger.warning("LLM scoring failed, falling back to keyword matching")
        return score_resume_keyword_match(resume_text, criteria)
    
    try:
        scores_dict = json.loads(response)
        
        # Convert to the expected format with criterion text as keys
        result = {}
        for i, criterion in enumerate(criteria):
            key = str(i + 1)
            if key in scores_dict:
                # Use first 3 words of criterion as the column name
                column_name = ' '.join(criterion.split()[:3]).title()
                result[column_name] = int(scores_dict[key])
            else:
                # Fallback if criterion is missing in response
                column_name = ' '.join(criterion.split()[:3]).title()
                result[column_name] = 0
        
        return result
    
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Failed to parse LLM scoring response: {str(e)}, falling back to keyword matching")
        return score_resume_keyword_match(resume_text, criteria)

def score_resume_keyword_match(resume_text: str, criteria: List[str]):
    """Score a resume against criteria using keyword matching."""
    scores = {}
    
    for criterion in criteria:
        # Convert criterion to a column name (first 3 words)
        column_name = ' '.join(criterion.split()[:3]).title()
        if len(column_name) > 20:  # Truncate if too long
            column_name = column_name[:20]
        
        # Score based on keyword presence and frequency
        score = 0
        if criterion.lower() in resume_text.lower():
            # Count occurrences
            count = resume_text.lower().count(criterion.lower())
            
            # Basic scoring logic
            if count > 3:
                score = 5
            elif count > 2:
                score = 4
            elif count > 1:
                score = 3
            else:
                score = 2
        
        scores[column_name] = score
    
    return scores

async def extract_candidate_name_with_llm(resume_text: str, filename: str, settings: Settings):
    """Extract candidate name using LLM when available, fallback to rule-based."""
    model = get_llm_model(settings)
    if not model:
        return extract_candidate_name_rule_based(filename, resume_text)
    
    # Take just the first 1000 characters - name is typically at the top
    resume_start = resume_text[:1000]
    
    prompt = f"""
    Extract the full name of the candidate from this resume extract. Return ONLY the name, nothing else.
    
    Resume extract:
    {resume_start}
    """
    
    response = await call_llm_api(prompt, model, max_tokens=100)
    
    if response and len(response.split()) >= 2:
        return response.strip()
    else:
        return extract_candidate_name_rule_based(filename, resume_text)

def extract_candidate_name_rule_based(filename, resume_text):
    """Extract candidate name from filename or resume text using rules."""
    # First try to extract from filename by removing extension and special chars
    name = Path(filename).stem.replace('_', ' ').replace('-', ' ').title()
    
    # If the name looks reasonable (2+ words), return it
    if len(name.split()) >= 2:
        return name
    
    # Otherwise, try to find a name pattern in the first few lines of the resume
    first_lines = resume_text.split('\n')[:10]
    for line in first_lines:
        # Look for a line with 2-3 words that might be a name
        words = line.strip().split()
        if 2 <= len(words) <= 3 and all(w.isalpha() for w in words):
            return line.strip()
    
    # If all else fails, return the filename-based name
    return name
