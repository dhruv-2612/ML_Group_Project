import os
import glob
import pandas as pd
import logging
import time
import shutil
from src.utils import clean_job_data
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Fallback manual parsing if dotenv fails
if not HF_TOKEN and os.path.exists(".env"):
    try:
        with open(".env", "r") as f:
            for line in f:
                if "HUGGINGFACE_API_TOKEN" in line:
                    HF_TOKEN = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    except: pass

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def rebuild_with_huggingface():
    if not HF_TOKEN:
        logger.error("HUGGINGFACE_API_TOKEN not found in .env file.")
        return

    persist_dir = "data/embeddings/chroma_db"
    
    # 1. Clear existing database
    if os.path.exists(persist_dir):
        logger.info(f"Clearing existing vector store at {persist_dir}...")
        shutil.rmtree(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)

    # 2. Initialize Hugging Face Embeddings
    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=HF_TOKEN,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="job_postings_2026",
        client_settings=Settings(anonymized_telemetry=False)
    )

    # 3. Load all raw data
    raw_files = sorted(glob.glob("data/raw/naukri_*.csv"))
    if not raw_files:
        logger.warning("No raw data files found.")
        return

    if not os.path.exists("data/processed/job_clusters.csv"):
        logger.error("job_clusters.csv not found. Please run process_raw.py first.")
        return
    clusters_ref = pd.read_csv("data/processed/job_clusters.csv")

    processed_urls = set()
    all_documents = []

    logger.info("Preparing documents from all raw files...")
    for f in raw_files:
        try:
            df = pd.read_csv(f)
            cleaned_df = clean_job_data(df)
            
            # Merge to get cluster_id
            cleaned_df = cleaned_df.merge(clusters_ref[['job_title', 'company', 'cluster_id']], on=['job_title', 'company'], how='left')
            cleaned_df['cluster_id'] = cleaned_df['cluster_id'].fillna(-1).astype(int)

            for _, row in cleaned_df.iterrows():
                url = str(row.get('job_url', ''))
                if not url or url in processed_urls:
                    continue
                
                processed_urls.add(url)
                
                text_content = f"Job Title: {row.get('job_title')}\nCompany: {row.get('company')}\nDescription: {row.get('job_description')}\nSkills: {row.get('skills_text')}"
                
                metadata = {
                    "source": "job_posting",
                    "job_title": str(row.get('job_title', 'N/A')),
                    "company": str(row.get('company', 'N/A')),
                    "location": str(row.get('location', 'N/A')),
                    "experience_level": str(row.get('experience_level', 'N/A')),
                    "salary": str(row.get('salary_mid', 'N/A')),
                    "cluster_id": int(row.get('cluster_id', -1)),
                    "job_url": url,
                    "source_file": os.path.basename(f)
                }
                all_documents.append(Document(page_content=text_content, metadata=metadata))
        except Exception as e:
            logger.error(f"Error reading {f}: {e}")

    logger.info(f"Total unique documents to index: {len(all_documents)}")

    # 4. Batch Indexing
    batch_size = 50
    for i in range(0, len(all_documents), batch_size):
        batch = all_documents[i:i + batch_size]
        try:
            vectorstore.add_documents(batch)
            logger.info(f"Indexed batch {i//batch_size + 1}/{(len(all_documents)//batch_size)+1} ({len(batch)} docs)...")
            time.sleep(1) 
        except Exception as e:
            logger.error(f"Batch indexing failed: {e}. Retrying in 10s...")
            time.sleep(10)
            vectorstore.add_documents(batch)

    logger.info(f"Migration Complete! Total Jobs indexed: {len(processed_urls)}")

if __name__ == "__main__":
    rebuild_with_huggingface()
