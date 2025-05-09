
from enum import Enum
from typing import List
from pydantic import BaseModel, Field



# DEFAULT_MODEL =  "gemini-2.0-flash", PREMIUM_MODEL = "gemini-2.0-pro-exp-02-05"
class LLMMode(str, Enum):
    DISABLED = "disabled"
    BASIC = "gemini-2.0-flash"
    PREMIUM = "gemini-2.0-pro-exp-02-05"
    
class ExtractionMode(str, Enum):
    RULE_BASED = "rule_based"
    LLM_ASSISTED = "llm_assisted"
    
class ScoringMode(str, Enum):
    KEYWORD_MATCH = "keyword_match"
    SEMANTIC_MATCH = "semantic_match"
    FULL_ANALYSIS = "full_analysis"

class Settings(BaseModel):
    llm_mode: LLMMode = Field(default=LLMMode.BASIC, description="Level of LLM usage")
    extraction_mode: ExtractionMode = Field(default=ExtractionMode.LLM_ASSISTED, description="Method for extracting criteria")
    scoring_mode: ScoringMode = Field(default=ScoringMode.SEMANTIC_MATCH, description="Method for scoring resumes")
    
class CriteriaResponse(BaseModel):
    criteria: List[str]
    