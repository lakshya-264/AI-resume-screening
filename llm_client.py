import os
from openai import OpenAI
from pydantic import BaseModel
from typing import List

# Define the expected strict JSON output schema using Pydantic
class CandidateEvaluation(BaseModel):
    candidate_name: str
    score: int
    strengths: List[str]
    gaps: List[str]
    recommendation: str

def evaluate_resume(jd_text: str, resume_text: str) -> CandidateEvaluation:
    """Calls OpenAI API to evaluate the resume against the JD."""
    # Ensure OPENAI_API_KEY is available in the environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
    client = OpenAI(api_key=api_key)
    
    system_prompt = """You are an expert technical recruiter and AI hiring assistant. Your task is to evaluate a candidate's resume against a provided Job Description (JD).

### SCORING SYSTEM
Calculate a final score out of 100 based on:
1. Skills Match (50%): Core skills alignment.
2. Experience Relevance (30%): Relevant past experience and domain knowledge.
3. Tools/Tech Stack (20%): Knowledge of specific software/tools required.

### INSTRUCTIONS
1. Analyze the JD and extract mandatory requirements.
2. Analyze the Candidate's Resume.
3. Score the candidate out of 100 using the weights above.
4. Identify top 3 strengths and top 3 gaps (missing skills/experience).
5. Give a recommendation strictly as ONE of these three options: "Strong Fit", "Moderate Fit", or "Not Fit". ("Strong Fit" > 80, "Moderate Fit" 60-80, "Not Fit" < 60).
"""
    
    user_prompt = f"### JOB DESCRIPTION:\n{jd_text}\n\n### CANDIDATE RESUME:\n{resume_text}"
    
    # We use gpt-4o-mini with structured outputs for robust JSON generation
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=CandidateEvaluation,
    )
    
    return response.choices[0].message.parsed
