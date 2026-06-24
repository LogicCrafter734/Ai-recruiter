import json
import os
import sys
import gzip
import pandas as pd
import numpy as np
from embedding import get_embedding, model
from signals_scorer import calculate_behavioral_multiplier

OFFICIAL_JD_TEXT = """
Job Description: Senior AI Engineer — Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid — flexible cadence) | Open to relocation candidates from Tier-1 Indian cities
Employment Type: Full-time
Experience Required: 5–9 years

Deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning.
Scrappy product-engineering attitude — willing to ship a working ranker in a week even if the underlying ML is suboptimal.
The right answer to this JD is not find candidates whose skills section contains the most AI keywords. That's a trap we've explicitly built into the dataset.
The right answer involves reasoning about the gap between what the JD says and what the JD means. A candidate may not use the words RAG or Pinecone in their profile, but if their career history shows they built a recommendation system at a product company, they're a fit.
Your ranking system should also weigh behavioral signals — a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available. Down-weight them appropriately.
Located in or willing to relocate to Noida or Pune.
"""

def get_job_description():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    docx_path = os.path.join(base_dir, "job_description.docx")
    md_path = os.path.join(base_dir, "job_description.md")

    if os.path.exists(docx_path):
        import docx
        doc = docx.Document(docx_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    elif os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(OFFICIAL_JD_TEXT.strip())
        return OFFICIAL_JD_TEXT.strip()

def main():
    print("🚀 Initializing Vectorized Fast Production Pipeline...")
    jd_text = get_job_description()
    jd_vector = get_embedding(jd_text[:1200])
    norm_jd = np.linalg.norm(jd_vector)

    full_data_gz = "candidates.jsonl.gz"
    full_data_raw = "candidates.jsonl"
    sample_data = "sample_candidates.json"
    
    candidates_source = None
    is_jsonl = False
    is_compressed = False

    if os.path.exists(full_data_gz):
        candidates_source = full_data_gz
        is_jsonl = True
        is_compressed = True
        print(f"📦 Found production dataset archive: {full_data_gz}")
    elif os.path.exists(full_data_raw):
        candidates_source = full_data_raw
        is_jsonl = True
        print(f"📝 Found uncompressed production dataset: {full_data_raw}")
    elif os.path.exists(sample_data):
        candidates_source = sample_data
        print(f"⚠️ Warning: Full dataset not found. Using snippet: {sample_data}")
    else:
        print("❌ Error: Missing candidate pool source file.")
        sys.exit(1)

    # 1. Fast Streaming File Read
    print("📥 Step 1: Ingesting dataset into vectorized data arrays...")
    records = []
    
    if is_jsonl:
        open_func = gzip.open if is_compressed else open
        mode = "rt" if is_compressed else "r"
        with open_func(candidates_source, mode, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    else:
        with open(candidates_source, "r", encoding="utf-8") as f:
            records = json.load(f)

    total_records = len(records)
    print(f"📋 Ingested {total_records} profiles. Pre-flattening components...")

    # 2. Build Fast Data Structures for Text and Behavioral Signals
    candidate_ids = []
    text_payloads = []
    behavioral_scores = []
    title_multipliers = []
    current_titles = []
    experience_list = []

    for cand in records:
        profile = cand.get("profile", {})
        title = profile.get("current_title", "")
        current_titles.append(title)
        experience_list.append(profile.get("years_of_experience", 0))
        candidate_ids.append(cand.get("candidate_id"))

        # Multiplier block to suppress keyword stuffers
        title_lower = title.lower()
        is_stuffer_trap = any(tw in title_lower for tw in ["marketing", "sales", "hr manager", "recruiter", "writer"])
        title_multipliers.append(0.05 if is_stuffer_trap else 1.0)

        skills_list = [s.get("name", "") for s in cand.get("skills", [])]
        combined_text = f"Title: {title}. Summary: {profile.get('summary', '')}. Skills: {', '.join(skills_list)}"
        text_payloads.append(combined_text[:1200]) # Optimal window length slice

        behavioral_scores.append(calculate_behavioral_multiplier(cand.get("redrob_signals", {})))

    # 3. Vector Batch Model Encoding
    print("⚡ Step 2: Generating model matrix embeddings...")
    cand_embeddings = model.encode(text_payloads, batch_size=256, show_progress_bar=True, convert_to_numpy=True)

    # 4. Fast Vector Matrix Calculations using Numpy
    print("📈 Step 3: Computing semantic alignment scores...")
    norm_candidates = np.linalg.norm(cand_embeddings, axis=1)
    
    # Run matrix dot product division in a single vectorized CPU cycle execution step
    dot_products = np.dot(cand_embeddings, jd_vector)
    norms = norm_jd * norm_candidates
    semantic_similarities = np.where(norms > 0, dot_products / norms, 0.0)

    # Convert everything to Pandas columns to compute final composites instantly
    df_results = pd.DataFrame({
        "candidate_id": candidate_ids,
        "semantic": semantic_similarities,
        "behavioral": behavioral_scores,
        "multiplier": title_multipliers,
        "title": current_titles,
        "exp": experience_list
    })

    # Vectorized scoring calculation
    df_results["score"] = ((df_results["semantic"] * 0.60) + (df_results["behavioral"] * 0.40)) * df_results["multiplier"]
    df_results["score"] = df_results["score"].round(4)
    
    # Generate compliance-grade reasoning rows dynamically
    df_results["reasoning"] = df_results["title"] + " with " + df_results["exp"].astype(str) + " yrs exp; technical profile alignment matches " + (df_results["semantic"] * 100).round(1).astype(str) + "%."

    # 5. Handle Strict Sorting Spec Requirements (Score Descending, ID Ascending)
    print("⚖️ Step 4: Sorting candidates and outputting submission template...")
    df_results = df_results.sort_values(by=["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    
    df_top100 = df_results.head(100).copy()
    df_top100["rank"] = df_top100.index + 1

    # Isolate columns to match exact format rules
    submission_output = df_top100[["candidate_id", "rank", "score", "reasoning"]]
    
    # Save using your registered ID file wrapper name (e.g., team_xxx.csv)
    output_filename = "team.csv" 
    submission_output.to_csv(output_filename, index=False, encoding="utf-8")
    print(f"✨ Successfully generated valid 100-row file matrix inside '{output_filename}'!")

    if os.path.exists("validate_submission.py"):
        print("🔍 Launching local validate_submission.py verification pipeline tool...")
        import subprocess
        subprocess.run(["python", "validate_submission.py", output_filename])

if __name__ == "__main__":
    main()