import os
import re
import json
import hashlib
import logging
import asyncio
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import multiprocessing
import time

import httpx
import spacy
import textstat
from web3 import Web3

# Hugging Face / Transformers
from transformers import pipeline

# SentenceTransformers
from sentence_transformers import SentenceTransformer

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from dotenv import load_dotenv
load_dotenv()  # Loads environment variables from a .env file

# Ollama (optional for bias detection)
try:
    import ollama  # type: ignore
except ImportError:
    ollama = None

###############################################################################
#                               GLOBAL CONFIG                                 #
###############################################################################

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Validate required environment variables
REQUIRED_ENV_VARS = ["WEB3_PROVIDER", "TOKEN_CONTRACT_ADDRESS", "VOTING_CONTRACT_ADDRESS", "PRIVATE_KEY", "ACCOUNT_ADDRESS"]
for var in REQUIRED_ENV_VARS:
    if not os.getenv(var):
        raise EnvironmentError(f"Required environment variable {var} is not set.")

# Configuration
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
QDRANT_URL = os.getenv("QDRANT_URL", ":memory:")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "text_embeddings")
DAILY_TOKEN_BUDGET = 1000
API_SAVE_ENDPOINT = "http://localhost:3000/api/ratings/ai"
API_UPDATE_ENDPOINT = "http://localhost:3000/api/ratings/ai"

# Blockchain setup
WEB3_PROVIDER = os.getenv("WEB3_PROVIDER")
TOKEN_CONTRACT_ADDRESS = os.getenv("TOKEN_CONTRACT_ADDRESS")
VOTING_CONTRACT_ADDRESS = os.getenv("VOTING_CONTRACT_ADDRESS")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
ACCOUNT_ADDRESS = os.getenv("ACCOUNT_ADDRESS")

w3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER))
if not w3.is_connected():
    logger.error("Web3 connection failed!")
    raise Exception("Web3 connection failed!")

# Load ABIs
with open(os.path.join("abi", "CurateAIToken.json"), "r") as f:
    token_abi = json.load(f)["abi"]
with open(os.path.join("abi", "CurateAIVote.json"), "r") as f:
    voting_abi = json.load(f)["abi"]

token_contract = w3.eth.contract(address=Web3.to_checksum_address(TOKEN_CONTRACT_ADDRESS), abi=token_abi)
voting_contract = w3.eth.contract(address=Web3.to_checksum_address(VOTING_CONTRACT_ADDRESS), abi=voting_abi)

###############################################################################
#                           ONE-TIME GLOBAL LOAD                              #
###############################################################################

logger.info("Loading spaCy model...")
nlp = spacy.load(SPACY_MODEL, disable=["ner", "parser"])

logger.info("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

logger.info("Initializing sentiment pipeline...")
sentiment_analyzer = pipeline("sentiment-analysis", truncation=True, max_length=512)

logger.info(f"Connecting to Qdrant at {QDRANT_URL}...")
client = QdrantClient(QDRANT_URL)

def ensure_qdrant_collection() -> None:
    """Ensure the Qdrant collection exists, creating it if necessary."""
    try:
        client.get_collection(QDRANT_COLLECTION_NAME)
        logger.info(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' found.")
    except Exception:
        logger.info(f"Creating Qdrant collection '{QDRANT_COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

ensure_qdrant_collection()

###############################################################################
#                           HELPER FUNCTIONS                                  #
###############################################################################

async def fetch_posts_async(url: str) -> List[Dict[str, Any]]:
    """Fetch posts asynchronously from the given URL."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            logger.info(f"Fetched posts from {url}: {response.status_code}")
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch posts from {url}: {e}")
            raise

def get_embedding(text: str) -> List[float]:
    """Compute sentence embedding for a given text."""
    return embedding_model.encode(text).tolist()

def remove_large_quotes(text: str) -> str:
    """Remove quoted blocks of 30+ words to prevent plagiarism gaming."""
    return re.sub(r'"(?:\S+\s+){30,}"', '', text)

def save_report_to_file(report: Dict[str, Any], post_id: int) -> None:
    """Save the analysis report to a JSON file in the generation_reports folder."""
    folder = "generation_reports"
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, f"post_{post_id}_report.json")
    with open(filename, 'w') as f:
        json.dump(report, f, indent=4)
    logger.info(f"Saved report for post {post_id} to {filename}")

###############################################################################
#                            ANALYSIS STRATEGIES                              #
###############################################################################

class AnalysisStrategy(ABC):
    """Abstract base class for analysis strategies."""
    @abstractmethod
    def analyze(self, text: str) -> Dict[str, Any]:
        pass

class SentimentAnalysisStrategy(AnalysisStrategy):
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text sentiment using the Hugging Face pipeline."""
        start_time = time.time()
        try:
            result = sentiment_analyzer(text)[0]
            label = result["label"]
            raw_score = float(result["score"])
            score = 1.0 - raw_score if label.upper() == "NEGATIVE" else raw_score
            return {"label": label, "score": round(score, 3), "time_taken": time.time() - start_time}
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"label": "UNKNOWN", "score": 0.5, "time_taken": time.time() - start_time}

class BiasDetectionStrategy(AnalysisStrategy):
    def analyze(self, text: str) -> Dict[str, float]:
        """Detect bias using Ollama or a fallback zero-shot classifier."""
        start_time = time.time()
        if ollama:
            try:
                prompt = f"Analyze bias (0=neutral, 1=biased):\n\n{text}"
                response = ollama.generate(model="llama3.2", prompt=prompt)
                match = re.search(r"0?\.?\d+", response.get("response", ""))
                return {"score": round(float(match.group()), 3), "time_taken": time.time() - start_time} if match else {"score": 0.5, "time_taken": time.time() - start_time}
            except Exception as e:
                logger.error(f"Ollama bias detection failed: {e}")
                return {"score": 0.5, "time_taken": time.time() - start_time}
        else:
            logger.info("Ollama unavailable, using zero-shot classification.")
            try:
                classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
                result = classifier(text, candidate_labels=["neutral", "biased"])
                return {"score": round(result["scores"][result["labels"].index("biased")], 3), "time_taken": time.time() - start_time}
            except Exception as e:
                logger.error(f"Bias detection fallback failed: {e}")
                return {"score": 0.5, "time_taken": time.time() - start_time}

class OriginalityEvaluationStrategy(AnalysisStrategy):
    def analyze(self, text: str) -> Dict[str, float]:
        """Evaluate text originality using Qdrant similarity search."""
        start_time = time.time()
        try:
            embedding = get_embedding(text)
            results = client.query_points(collection_name=QDRANT_COLLECTION_NAME, query_vector=embedding, limit=5)
            text_hash = hashlib.md5(text.encode()).hexdigest()
            client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=[PointStruct(id=text_hash, vector=embedding, payload={"text": text})]
            )
            if not results:
                return {"score": 1.0, "average_similarity": 0.0, "time_taken": time.time() - start_time}
            avg_sim = sum(r.score for r in results) / len(results)
            return {"score": round(1.0 - avg_sim, 3), "average_similarity": round(avg_sim, 3), "time_taken": time.time() - start_time}
        except Exception as e:
            logger.error(f"Originality check failed: {e}")
            return {"score": 1.0, "average_similarity": 0.0, "time_taken": time.time() - start_time}

class PlagiarismCheckStrategy(AnalysisStrategy):
    def analyze(self, text: str) -> Dict[str, float]:
        """Check for plagiarism by comparing embeddings in Qdrant."""
        start_time = time.time()
        try:
            cleaned_text = remove_large_quotes(text)
            embedding = get_embedding(cleaned_text)
            results = client.query_points(collection_name=QDRANT_COLLECTION_NAME, query_vector=embedding, limit=1)
            text_hash = hashlib.md5(text.encode()).hexdigest()
            client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=[PointStruct(id=text_hash, vector=embedding, payload={"text": text})]
            )
            score = round(float(results[0].score), 3) if results else 0.0
            return {"score": score, "time_taken": time.time() - start_time}
        except Exception as e:
            logger.error(f"Plagiarism check failed: {e}")
            return {"score": 0.0, "time_taken": time.time() - start_time}

class ReadabilityEvaluationStrategy(AnalysisStrategy):
    def analyze(self, text: str) -> Dict[str, float]:
        """Evaluate text readability using Flesch-Kincaid and Gunning Fog indices."""
        start_time = time.time()
        try:
            fk = textstat.flesch_kincaid_grade(text)
            gf = textstat.gunning_fog(text)
            paragraphs = [p for p in text.split("\n") if len(p.strip()) > 20]
            paragraph_factor = 1.0 if len(paragraphs) >= 2 else 0.8
            words = text.split()
            uniqueness_factor = 1.0 if len(set(words)) / len(words) >= 0.4 else 0.8

            def clamp(x): return max(0.0, min(x, 1.0))
            read_flesch = clamp(1.0 - abs(fk - 8.0) / 10.0)
            read_fog = clamp(1.0 - abs(gf - 10.0) / 10.0)
            final_score = (read_flesch + read_fog) / 2.0 * paragraph_factor * uniqueness_factor
            return {
                "flesch_kincaid_grade": round(float(fk), 2),
                "gunning_fog_index": round(float(gf), 2),
                "readability_score": round(final_score, 3),
                "time_taken": time.time() - start_time
            }
        except Exception as e:
            logger.error(f"Readability evaluation failed: {e}")
            return {"flesch_kincaid_grade": 0.0, "gunning_fog_index": 0.0, "readability_score": 0.0, "time_taken": time.time() - start_time}

ANALYSIS_STRATEGIES = {
    "sentiment": SentimentAnalysisStrategy(),
    "bias": BiasDetectionStrategy(),
    "originality": OriginalityEvaluationStrategy(),
    "plagiarism": PlagiarismCheckStrategy(),
    "readability": ReadabilityEvaluationStrategy(),
}

async def dynamic_analysis(text: str, steps: Optional[List[str]] = None) -> Dict[str, Any]:
    """Perform dynamic text analysis using specified strategies with timings."""
    if steps is None:
        steps = list(ANALYSIS_STRATEGIES.keys())

    start_time = time.time()
    doc = nlp(text)
    cleaned_text = " ".join(token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct)
    arg_map = {"sentiment": cleaned_text, "bias": cleaned_text, "originality": text, "plagiarism": text, "readability": cleaned_text}

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as pool:
        tasks = {step: loop.run_in_executor(pool, partial(ANALYSIS_STRATEGIES[step].analyze, arg_map[step])) for step in steps}
        results = await asyncio.gather(*tasks.values())
    analysis_results = dict(zip(tasks.keys(), results))

    final_score = compute_final_score(analysis_results)
    word_count = len(text.split())
    if word_count < 300:
        final_score *= 0.8
        analysis_results["length_penalty"] = f"Applied 20% penalty due to low word count ({word_count} words)."

    # Score breakdown
    scores = {k: analysis_results[k]["score"] if k != "readability" else analysis_results[k]["readability_score"] for k in steps}
    score_breakdown = {
        "sentiment": {"raw": scores["sentiment"], "weight": 0.25, "weighted": round(0.25 * scores["sentiment"], 3)},
        "originality": {"raw": scores["originality"], "weight": 0.25, "weighted": round(0.25 * scores["originality"], 3)},
        "inverted_bias": {"raw": 1.0 - scores["bias"], "weight": 0.20, "weighted": round(0.20 * (1.0 - scores["bias"]), 3)},
        "inverted_plagiarism": {"raw": 1.0 - scores["plagiarism"], "weight": 0.15, "weighted": round(0.15 * (1.0 - scores["plagiarism"]), 3)},
        "readability": {"raw": scores["readability"], "weight": 0.15, "weighted": round(0.15 * scores["readability"], 3)},
    }
    analysis_results.update({
        "final_score": round(final_score, 3),
        "final_score_percentage": f"{final_score * 100:.2f}%",
        "score_breakdown": score_breakdown,
        "total_time": time.time() - start_time
    })
    return analysis_results

def compute_final_score(analysis: Dict[str, Any]) -> float:
    """Compute weighted final score from analysis results."""
    scores = {
        "sentiment": analysis["sentiment"]["score"],
        "originality": analysis["originality"]["score"],
        "bias": analysis["bias"]["score"],
        "plagiarism": analysis["plagiarism"]["score"],
        "readability": analysis["readability"]["readability_score"]
    }
    return round(0.25 * scores["sentiment"] + 0.25 * scores["originality"] + 0.20 * (1.0 - scores["bias"]) +
                 0.15 * (1.0 - scores["plagiarism"]) + 0.15 * scores["readability"], 3)

###############################################################################
#                    BLOCKCHAIN TOKEN DISBURSEMENT                            #
###############################################################################

def disburse_tokens(post_id: int, token_reward: float, retries: int = 3) -> str:
    gas_price = w3.eth.gas_price
    for attempt in range(retries):
        try:
            amount = int(token_reward * 10**18)
            nonce = w3.eth.get_transaction_count(Web3.to_checksum_address(ACCOUNT_ADDRESS), 'pending')
            txn = voting_contract.functions.aiVote(post_id, amount).build_transaction({
                'from': Web3.to_checksum_address(ACCOUNT_ADDRESS),
                'nonce': nonce,
                'gas': 200000,
                'gasPrice': gas_price
            })
            signed_txn = w3.eth.account.sign_transaction(txn, PRIVATE_KEY)
            tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            logger.info(f"Transaction successful for post {post_id}: {tx_hash.hex()}, Gas Used: {receipt['gasUsed']}")
            return tx_hash.hex()
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"Retrying disbursement for post {post_id} (attempt {attempt + 1}): {e}")
                time.sleep(2)
            else:
                logger.error(f"Failed to disburse tokens for post {post_id} after {retries} attempts: {e}")
                raise

###############################################################################
#                       SAVE/UPDATE ANALYSIS                                  #
###############################################################################

async def save_analysis(analysis_data: Dict[str, Any], post_id: int) -> None:
    """Save analysis data via API."""
    payload = {
        "postId": post_id,
        "sentimentAnalysisLabel": analysis_data["sentiment"]["label"],
        "sentimentAnalysisScore": analysis_data["sentiment"]["score"],
        "biasDetectionScore": analysis_data["bias"]["score"],
        "biasDetectionDirection": "",
        "originalityScore": analysis_data["originality"]["score"],
        "similarityScore": analysis_data["originality"]["average_similarity"],
        "readabilityFleschKincaid": analysis_data["readability"]["flesch_kincaid_grade"],
        "readabilityGunningFog": analysis_data["readability"]["gunning_fog_index"],
        "mainTopic": "",
        "secondaryTopics": [],
        "rating": round(analysis_data["final_score"] * 100),
        "finalScore": analysis_data["final_score"],
        "finalScorePercentage": analysis_data["final_score_percentage"],
        "tokenReward": analysis_data.get("token_reward", 0.0),
        "scoreBreakdown": analysis_data["score_breakdown"]
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(API_SAVE_ENDPOINT, json=payload)
            response.raise_for_status()
            logger.info(f"Saved analysis for post {post_id}: {response.json()}")
        except Exception as e:
            logger.error(f"Failed to save analysis for post {post_id}: {e}")

async def update_analysis(analysis_data: Dict[str, Any], post_id: int) -> None:
    """Update analysis data via API."""
    payload = {"tokenReward": analysis_data.get("token_reward", 0.0)}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.put(f"{API_UPDATE_ENDPOINT}/{post_id}", json=payload)
            response.raise_for_status()
            logger.info(f"Updated analysis for post {post_id}: {response.json()}")
        except Exception as e:
            logger.error(f"Failed to update analysis for post {post_id}: {e}")

###############################################################################
#                                   MAIN                                      #
###############################################################################

async def main() -> None:
    """Main function to fetch, analyze, reward posts, and save reports."""
    url = "http://localhost:3000/api/posts"
    logger.info(f"Fetching posts from {url}...")
    posts = await fetch_posts_async(url)

    post_results = []
    total_posts = len(posts)
    analyzed_posts = 0
    total_tokens_disbursed = 0.0

    for idx, post in enumerate(posts, start=1):
        content = post.get("content", "")
        if not content:
            logger.warning(f"Post {idx} has no content. Skipping.")
            continue
        try:
            analysis_result = await dynamic_analysis(content)
            post_id = post.get("id", idx)
            post_results.append({
                "post": post,
                "analysis": analysis_result,
                "final_score": analysis_result["final_score"]
            })
            analyzed_posts += 1
            print(f"\n=== ANALYSIS FOR POST ID {post_id} ===")
            await save_analysis(analysis_result, post_id)
        except Exception as e:
            logger.error(f"Error analyzing post {idx} (ID: {post.get('id', idx)}): {e}")

    # Distribute rewards to top 10 posts
    top_posts = sorted(post_results, key=lambda x: x["final_score"], reverse=True)[:10]
    total_score = sum(item["final_score"] for item in top_posts) or 1  # Avoid division by zero
    rewarded_posts = 0
    for item in top_posts:
        token_reward = DAILY_TOKEN_BUDGET * item["final_score"] / total_score
        item["analysis"]["token_reward"] = round(token_reward, 3)
        total_tokens_disbursed += token_reward
        print(f"Post ID {item['post']['id']}: Score = {item['final_score']}, Reward = {item['analysis']['token_reward']} tokens")
        try:
            tx_hash = disburse_tokens(item["post"]["id"], item["analysis"]["token_reward"])
            item["analysis"]["tx_hash"] = tx_hash
            rewarded_posts += 1
            await update_analysis(item["analysis"], item["post"]["id"])
        except Exception as e:
            logger.error(f"Failed to disburse tokens for post {item['post']['id']}: {e}")

    # Save reports for all posts
    for item in post_results:
        report = {
            "post": item["post"],
            "analysis": item["analysis"]
        }
        save_report_to_file(report, item["post"]["id"])

    # Generate summary report
    summary = {
        "total_posts": total_posts,
        "analyzed_posts": analyzed_posts,
        "rewarded_posts": rewarded_posts,
        "total_tokens_disbursed": round(total_tokens_disbursed, 3),
        "average_final_score": round(sum(item["final_score"] for item in post_results) / analyzed_posts, 3) if analyzed_posts > 0 else 0.0
    }
    logger.info(f"Summary Report: {summary}")
    with open("generation_reports/summary_report.json", "w") as f:
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    asyncio.run(main())