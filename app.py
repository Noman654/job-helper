from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import os
import tempfile
import shutil
from typing import List
import logging
import pandas as pd
# For Gemini integration

from models import LLMMode, ExtractionMode, ScoringMode, Settings, CriteriaResponse

from utills import (
    extract_text_from_file,
    extract_candidate_name_rule_based,
    extract_candidate_name_with_llm,
    score_resume_keyword_match,
    score_resume_with_llm,
    extract_criteria_rule_based,
    extract_criteria_with_llm
)

import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Advanced Resume Ranking System with Gemini",
    description="API for extracting criteria from job descriptions and scoring resumes with Gemini integration",
    version="2.0.0"
)



# FIX: Properly define query parameters for settings
@app.post("/extract-criteria", 
          summary="Extract ranking criteria from job description",
          description="Accepts a job description file (PDF or DOCX) and extracts key ranking criteria",
          response_model=CriteriaResponse)
async def extract_criteria(
    file: UploadFile = File(...),
    llm_mode: LLMMode = Query(LLMMode.BASIC, description="Level of LLM usage"),
    extraction_mode: ExtractionMode = Query(ExtractionMode.LLM_ASSISTED, description="Method for extracting criteria"),
):
    """
    Extract ranking criteria from a job description file.
    
    Parameters:
    - file: PDF or DOCX file containing the job description
    - llm_mode: Level of LLM usage (disabled, basic, premium)
    - extraction_mode: Method for extracting criteria (rule_based, llm_assisted)
    
    Returns:
    - JSON with list of extracted criteria
    """
    # Create settings object from query parameters
    settings = Settings(
        llm_mode=llm_mode,
        extraction_mode=extraction_mode
    )
    
    # Create temp file to store the uploaded file
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name
    
    try:
        # Extract text based on file type
        text = extract_text_from_file(temp_path, file.filename)
        
        # Extract criteria based on selected mode
        if settings.extraction_mode == ExtractionMode.LLM_ASSISTED and settings.llm_mode != LLMMode.DISABLED:
            criteria = await extract_criteria_with_llm(text, settings)
        else:
            criteria = extract_criteria_rule_based(text)
        
        return {"criteria": criteria}
    
    except Exception as e:
        logger.error(f"Error in extract_criteria: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    
    finally:
        # Clean up the temp file
        os.unlink(temp_path)


# FIX: Properly define query parameters for settings
@app.post("/score-resumes",
          summary="Score resumes against extracted criteria",
          description="Scores multiple resumes based on provided criteria and returns Excel sheet with results")
async def score_resumes(
    # criteria: List[str] = Form(..., description="List of criteria strings to match against resumes"),
    criteria: List[str] = Query(..., description="List of criteria strings to match against resumes"),
    files: List[UploadFile] = File(...),
    llm_mode: LLMMode = Form(LLMMode.BASIC, description="Level of LLM usage"),
    extraction_mode: ExtractionMode = Form(ExtractionMode.LLM_ASSISTED, description="Method for extracting criteria"),
    scoring_mode: ScoringMode = Form(ScoringMode.SEMANTIC_MATCH, description="Method for scoring resumes")
):
    """
    Score multiple resumes against provided criteria.
    
    Parameters:
    - criteria: List of criteria strings to match against resumes
    - files: List of resume files (PDF or DOCX)
    - llm_mode: Level of LLM usage (disabled, basic, premium)
    - extraction_mode: Method for extracting criteria (rule_based, llm_assisted)
    - scoring_mode: Method for scoring resumes (keyword_match, semantic_match, full_analysis)
    
    Returns:
    - Excel file with candidate scores
    """
    # Create settings object from form parameters
    settings = Settings(
        llm_mode=llm_mode,
        extraction_mode=extraction_mode,
        scoring_mode=scoring_mode
    )
    
    # Create temp directory for storing files
    temp_dir = tempfile.mkdtemp()
    
    try:
        results = []
        
        for file in files:
            # Save the uploaded file
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            try:
                # Extract text from resume
                resume_text = extract_text_from_file(file_path, file.filename)
                
                # Get candidate name
                if settings.llm_mode != LLMMode.DISABLED:
                    candidate_name = await extract_candidate_name_with_llm(resume_text, file.filename, settings)
                else:
                    candidate_name = extract_candidate_name_rule_based(file.filename, resume_text)
                
                # Score resume against criteria
                if settings.scoring_mode in [ScoringMode.SEMANTIC_MATCH, ScoringMode.FULL_ANALYSIS] and settings.llm_mode != LLMMode.DISABLED:
                    scores = await score_resume_with_llm(resume_text, criteria, settings)
                else:
                    scores = score_resume_keyword_match(resume_text, criteria)
                
                # Add to results
                result = {
                    "Candidate Name": candidate_name,
                    **scores,
                    "Total Score": sum(scores.values())
                }
                results.append(result)
            
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                # Add error entry
                results.append({
                    "Candidate Name": f"ERROR: {file.filename}",
                    "Error": str(e),
                    "Total Score": 0
                })
        
        # Sort results by total score (descending)
        results.sort(key=lambda x: x.get("Total Score", 0), reverse=True)
        
        # Create Excel file
        df = pd.DataFrame(results)
        
        # Reorder columns to put Candidate Name first, Total Score last
        columns = ["Candidate Name"] + [c for c in df.columns if c not in ["Candidate Name", "Total Score", "Error"]]
        if "Error" in df.columns:
            columns.append("Error")
        columns.append("Total Score")
        df = df[columns]
        

        dir = './temp'
        # excel_file_path = os.path.join(dir, "resume_scores.xlsx")
        excel_file_path = os.path.join(dir, "resume_scores.csv")
        os.makedirs(dir, exist_ok=True)
        # df.to_excel(excel_file_path, index=False)
        df.to_csv(excel_file_path, index=False)
        
        
        # Return the Excel file
        return FileResponse(
            excel_file_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename="resume_scores.csv"
        )
    
    except Exception as e:
        logger.error(f"Error in score_resumes: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    
    finally:
        # Clean up the temp directory
        shutil.rmtree(temp_dir)



@app.get("/health",
         summary="Health check endpoint",
         description="Verifies the API is running and returns configuration information")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
    - Status information about the API
    """
    has_gemini_key = bool(GEMINI_API_KEY)
    
    return {
        "status": "healthy",
        "version": app.version,
        "llm_available": has_gemini_key,
        "llm_model": DEFAULT_MODEL,
        "premium_model": PREMIUM_MODEL,
        "jobs_in_progress": len(job_scores_store),
        "timestamp": time.time()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)