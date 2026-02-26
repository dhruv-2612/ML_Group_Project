import os
import glob
import pandas as pd
import logging
import time
from src.utils import clean_job_data
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def rebuild_vectorstore_with_limit(limit=100, batch_size=10):
    persist_dir = "data/embeddings/chroma_db"
    
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir, exist_ok=True)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY)
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="job_postings_2026",
        client_settings=Settings(anonymized_telemetry=False)
    )

    # Load existing URLs to resume
    processed_urls = set()
    try:
        existing_data = vectorstore.get()
        if existing_data and 'metadatas' in existing_data:
            processed_urls = {m.get('job_url') for m in existing_data['metadatas'] if m.get('job_url')}
        logger.info(f"Resuming: Found {len(processed_urls)} already indexed jobs.")
    except Exception as e:
        logger.warning(f"Could not load existing data: {e}")

    raw_files = sorted(glob.glob("data/raw/naukri_*.csv"))
    if not raw_files:
        logger.warning("No raw data files found.")
        return

    request_count = 0 # This will track batches in this session
    file_status = {}
    
    # Load clusters reference
    if not os.path.exists("data/processed/job_clusters.csv"):
        logger.error("job_clusters.csv not found. Please run process_raw.py first.")
        return
    clusters_ref = pd.read_csv("data/processed/job_clusters.csv")

    for f in raw_files:
        logger.info(f"Checking file: {f}")
        try:
            df = pd.read_csv(f)
            cleaned_df = clean_job_data(df)
            
            # Merge to get cluster_id
            cleaned_df = cleaned_df.merge(clusters_ref[['job_title', 'company', 'cluster_id']], on=['job_title', 'company'], how='left')
            cleaned_df['cluster_id'] = cleaned_df['cluster_id'].fillna(-1).astype(int)

            file_new_docs = []
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
                file_new_docs.append(Document(page_content=text_content, metadata=metadata))

            if not file_new_docs:
                file_status[f] = "Already Processed or Empty"
                continue

            # Add in batches
            file_indexed_count = 0
            for i in range(0, len(file_new_docs), batch_size):
                # Total sessions limit (user said they have 100 requests limit)
                # If they already used 89, we should check what's left.
                # I'll let it run and catch the 429 again if it's a hard limit.
                
                batch = file_new_docs[i:i + batch_size]
                vectorstore.add_documents(batch)
                request_count += 1
                file_indexed_count += len(batch)
                
                logger.info(f"Request {request_count} in this session: Indexed {len(batch)} docs (Total Indexed: {len(processed_urls)})")
                
                # Sleep to stay under 15 RPM (4 requests per minute = 15s delay)
                logger.info("Sleeping 15 seconds to avoid Rate Limit...")
                time.sleep(15)

            if file_indexed_count == len(file_new_docs):
                file_status[f] = "Full"
            else:
                file_status[f] = f"Partial ({file_indexed_count}/{len(file_new_docs)} new jobs)"

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                logger.error(f"Quota exceeded (429) at file {f}. Stopping.")
                file_status[f] = "Stopped due to Quota"
                break
            logger.error(f"Error processing {f}: {e}")
            file_status[f] = f"Error: {str(e)}"

    logger.info("--- Resumed Processing Summary ---")
    for f, status in file_status.items():
        print(f"{os.path.basename(f)}: {status}")
    
    print(f"\nRequests used in this session: {request_count}")
    print(f"Total Unique Jobs now in vector store: {len(processed_urls)}")

if __name__ == "__main__":
    # The user said 89 used, total 100 limit. 
    # But let's just run until it fails or completes.
    rebuild_vectorstore_with_limit(limit=1000, batch_size=10)
