import os
import pandas as pd
from dotenv import load_dotenv

from extractor import extract_text_from_file
from llm_client import evaluate_resume

# Load environment variables (e.g., OPENAI_API_KEY from .env)
load_dotenv()

def main():
    # 1. Configuration
    jd_path = "job_description.txt"
    resumes_dir = "resumes"
    output_file = "output_rankings.csv"
    
    # Check if directories/files exist
    if not os.path.exists(jd_path):
        print(f"Error: Could not find Job Description at '{jd_path}'. Please create it.")
        return
        
    if not os.path.exists(resumes_dir):
        print(f"Creating '{resumes_dir}' directory. Please place PDF/TXT resumes in it and run again.")
        os.makedirs(resumes_dir)
        return

    # 2. Load Job Description
    print("Loading Job Description...")
    jd_text = extract_text_from_file(jd_path)
    if not jd_text.strip():
        print(f"Error: Job Description file '{jd_path}' is empty.")
        return
    
    # 3. Process Resumes
    evaluations = []
    resume_files = [f for f in os.listdir(resumes_dir) if f.endswith(('.pdf', '.txt'))]
    
    if not resume_files:
        print(f"No resumes found in '{resumes_dir}'. Please add some and run again.")
        return
        
    print(f"Found {len(resume_files)} resumes. Starting evaluation...")
    
    for filename in resume_files:
        print(f"  -> Processing: {filename}")
        file_path = os.path.join(resumes_dir, filename)
        
        resume_text = extract_text_from_file(file_path)
        if not resume_text.strip():
            print(f"     Warning: Could not extract text from {filename}. Skipping.")
            continue
            
        try:
            # Call OpenAI API
            eval_result = evaluate_resume(jd_text, resume_text)
            
            # Convert Result to Dict
            eval_dict = eval_result.model_dump()
            eval_dict["filename"] = filename
            
            # Format list of strings into a single readable string for the CSV
            eval_dict["strengths"] = "\n".join(f"- {s}" for s in eval_dict["strengths"])
            eval_dict["gaps"] = "\n".join(f"- {g}" for g in eval_dict["gaps"])
            
            evaluations.append(eval_dict)
            print(f"     Done. Score: {eval_dict['score']} | Recommendation: {eval_dict['recommendation']}")
        except Exception as e:
            print(f"     Error evaluating {filename}: {e}")
            
    # 4. Save Rankings
    if evaluations:
        print("\nCompiling rankings and saving to CSV...")
        df = pd.DataFrame(evaluations)
        
        # Sort by score descending
        df = df.sort_values(by="score", ascending=False)
        
        # Reorder columns for better readability
        cols = ["candidate_name", "score", "recommendation", "strengths", "gaps", "filename"]
        df = df[cols]
        
        df.to_csv(output_file, index=False)
        print(f"\nSuccess! Rankings saved to {output_file}")
        
if __name__ == "__main__":
    main()
