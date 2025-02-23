import os
import re
import json
import hashlib
import logging
import asyncio
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional

import requests
import spacy
import textstat
import httpx  # pip install httpx

from web3 import Web3

# Hugging Face / Transformers
from transformers import pipeline

# SentenceTransformers
from sentence_transformers import SentenceTransformer

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Ollama (if installed)
try:
    import ollama  # type: ignore
except ImportError:
    ollama = None

###############################################################################
#                               GLOBAL CONFIG                                 #
###############################################################################

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Environment-based configuration for NLP and Qdrant
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
QDRANT_URL = os.getenv("QDRANT_URL", ":memory:")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "text_embeddings")

# Daily token budget for rewards.
DAILY_TOKEN_BUDGET = 1000

# API endpoints for saving/updating AI post ratings.
API_SAVE_ENDPOINT = "http://localhost:3000/api/ratings/ai"
API_UPDATE_ENDPOINT = "http://localhost:3000/api/ratings/ai"

# Blockchain configuration
WEB3_PROVIDER = os.getenv("WEB3_PROVIDER", "http://localhost:8545")
TOKEN_CONTRACT_ADDRESS = os.getenv("TOKEN_CONTRACT_ADDRESS","0x33b0AA3D65Fda9cB3C80D45fB5b42159623a2759")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
ACCOUNT_ADDRESS = os.getenv("ACCOUNT_ADDRESS")

w3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER))
if not w3.isConnected():
    logger.error("Web3 is not connected!")

# Token ABI (replace with your actual ABI)
tokenAbi = [
  {
    "inputs": [
      {"internalType": "address", "name": "spender", "type": "address"},
      {"internalType": "uint256", "name": "value", "type": "uint256"}
    ],
    "name": "approve",
    "outputs": [
      {"internalType": "bool", "name": "", "type": "bool"}
    ],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      {"internalType": "address", "name": "recipient", "type": "address"},
      {"internalType": "uint256", "name": "amount", "type": "uint256"}
    ],
    "name": "distributeTokens",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
]  # Make sure your ABI is complete.

token_contract = w3.eth.contract(
    address=Web3.toChecksumAddress(TOKEN_CONTRACT_ADDRESS),
    abi=tokenAbi
)

###############################################################################
#                           ONE-TIME GLOBAL LOAD                               #
###############################################################################

logger.info("Loading spaCy model (once)...")
nlp = spacy.load(SPACY_MODEL, disable=["ner", "parser"])

logger.info("Loading embedding model (once)...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

logger.info("Initializing HF sentiment pipeline with truncation (once)...")
sentiment_analyzer = pipeline("sentiment-analysis", truncation=True, max_length=512)

# Initialize Qdrant client
logger.info(f"Connecting to Qdrant at {QDRANT_URL} (once)...")
client = QdrantClient(QDRANT_URL)

def ensure_qdrant_collection():
    """Create the Qdrant collection if it doesn't exist."""
    try:
        client.get_collection(QDRANT_COLLECTION_NAME)
        logger.info(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' found.")
    except Exception:
        logger.info(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' not found. Creating...")
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

ensure_qdrant_collection()

###############################################################################
#                           HELPER FUNCTIONS                                  #
###############################################################################

def fetch_posts(url: str) -> List[Dict[str, Any]]:
    """Fetches a list of posts from the given URL."""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def get_embedding(text: str) -> List[float]:
    """Compute sentence embedding for a given text."""
    return embedding_model.encode(text).tolist()

def remove_large_quotes(text: str) -> str:
    """
    Remove large quoted blocks that may game plagiarism detection.
    """
    return re.sub(r'"(?:\S+\s+){30,}"', '', text)

###############################################################################
#                            ANALYSIS FUNCTIONS                               #
###############################################################################

def analyze_sentiment(text: str) -> Dict[str, Any]:
    result = sentiment_analyzer(text)[0]
    label = result["label"]
    raw_score = float(result["score"])
    score = 1.0 - raw_score if label.upper() == "NEGATIVE" else raw_score
    return {"label": label, "score": round(score, 3)}

def detect_bias(text: str) -> Dict[str, float]:
    if ollama:
        prompt = (
            "Analyze bias in this text. Respond ONLY with a numerical score 0-1 "
            "where 0=very neutral, 1=highly biased:\n\n" + text
        )
        response = ollama.generate(model="llama3.2", prompt=prompt)
        resp_text = response.get("response", "")
        match = re.search(r"0?\.?\d+", resp_text)
        if match:
            val = float(match.group())
            return {"score": round(val, 3)}
        else:
            return {"score": 0.5}
    else:
        logger.info("Ollama not installed. Using zero-shot classification for bias detection.")
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        candidate_labels = ["neutral", "biased"]
        result = classifier(text, candidate_labels)
        bias_score = result["scores"][result["labels"].index("biased")]
        return {"score": round(bias_score, 3)}

def evaluate_originality(text: str) -> Dict[str, float]:
    try:
        K = 5
        embedding = get_embedding(text)
        results = client.search(
            collection_name=QDRANT_COLLECTION_NAME, query_vector=embedding, limit=K
        )
        if not results:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=[PointStruct(id=text_hash, vector=embedding, payload={"text": text})]
            )
            return {"score": 1.0, "average_similarity": 0.0}
        avg_sim = sum(r.score for r in results) / len(results)
        orig_score = 1.0 - avg_sim
        text_hash = hashlib.md5(text.encode()).hexdigest()
        client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[PointStruct(id=text_hash, vector=embedding, payload={"text": text})]
        )
        return {"score": round(orig_score, 3), "average_similarity": round(avg_sim, 3)}
    except Exception as e:
        logger.error(f"Originality check failed: {e}")
        return {"score": 1.0, "average_similarity": 0.0}

def check_plagiarism(text: str) -> Dict[str, float]:
    try:
        cleaned_text = remove_large_quotes(text)
        embedding = get_embedding(cleaned_text)
        results = client.search(
            collection_name=QDRANT_COLLECTION_NAME, query_vector=embedding, limit=1
        )
        text_hash = hashlib.md5(text.encode()).hexdigest()
        client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[PointStruct(id=text_hash, vector=embedding, payload={"text": text})]
        )
        if results and len(results) > 0:
            sim_score = float(results[0].score)
            return {"score": round(sim_score, 3)}
        return {"score": 0.0}
    except Exception as e:
        logger.error(f"Plagiarism check failed: {e}")
        return {"score": 0.0}

def evaluate_readability(text: str) -> Dict[str, float]:
    fk = textstat.flesch_kincaid_grade(text)
    gf = textstat.gunning_fog(text)
    paragraphs = [p for p in text.split("\n") if len(p.strip()) > 20]
    paragraph_factor = 1.0 if len(paragraphs) >= 2 else 0.8
    words = text.split()
    uniqueness = len(set(words)) / len(words) if words else 1.0
    uniqueness_factor = 1.0 if uniqueness >= 0.4 else 0.8
    def clamp(x): 
        return max(0.0, min(x, 1.0))
    flesch_diff = abs(fk - 8.0)
    read_flesch = clamp(1.0 - flesch_diff / 10.0)
    fog_diff = abs(gf - 10.0)
    read_fog = clamp(1.0 - fog_diff / 10.0)
    base_read_score = (read_flesch + read_fog) / 2.0
    final_read_score = base_read_score * paragraph_factor * uniqueness_factor
    return {
        "flesch_kincaid_grade": round(float(fk), 2),
        "gunning_fog_index": round(float(gf), 2),
        "readability_score": round(final_read_score, 3)
    }

###############################################################################
#                           SCORING COMPOSITE                                  #
###############################################################################

def compute_final_score(analysis: Dict[str, Any]) -> float:
    sentiment   = float(analysis["sentiment"]["score"])
    originality = float(analysis["originality"]["score"])
    bias        = float(analysis["bias"]["score"])
    plag        = float(analysis["plagiarism"]["score"])
    read_score  = float(analysis["readability"].get("readability_score", 0.0))
    
    inv_bias = 1.0 - bias
    inv_plag = 1.0 - plag

    final = (
        0.25 * sentiment +
        0.25 * originality +
        0.20 * inv_bias +
        0.15 * inv_plag +
        0.15 * read_score
    )
    final = max(0.0, min(final, 1.0))
    return round(final, 3)

###############################################################################
#                               DYNAMIC ANALYSIS                               #
###############################################################################

ANALYSIS_FUNCTIONS = {
    "sentiment": analyze_sentiment,
    "bias": detect_bias,
    "originality": evaluate_originality,
    "plagiarism": check_plagiarism,
    "readability": evaluate_readability,
}

async def dynamic_analysis(
    text: str,
    steps: Optional[List[str]] = None
) -> Dict[str, Any]:
    if steps is None:
        steps = list(ANALYSIS_FUNCTIONS.keys())

    doc = nlp(text)
    cleaned_text = " ".join(
        token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct
    )

    arg_map = {
        "sentiment": cleaned_text,
        "bias": cleaned_text,
        "originality": text,
        "plagiarism": text,
        "readability": cleaned_text,
    }

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        tasks = {step: loop.run_in_executor(pool, partial(ANALYSIS_FUNCTIONS[step], arg_map[step]))
                 for step in steps}
        step_keys = list(tasks.keys())
        results_list = await asyncio.gather(*tasks.values())

    analysis_results = {key: result for key, result in zip(step_keys, results_list)}
    final_score = compute_final_score(analysis_results)

    MIN_WORD_COUNT = 300
    word_count = len(text.split())
    penalty_info = None
    if word_count < MIN_WORD_COUNT:
        penalty_factor = 0.8
        final_score *= penalty_factor
        final_score = max(0.0, min(final_score, 1.0))
        penalty_info = f"Applied 20% penalty due to low word count ({word_count} words)."
        analysis_results["length_penalty"] = penalty_info

    sentiment_val = float(analysis_results["sentiment"]["score"])
    originality_val = float(analysis_results["originality"]["score"])
    bias_val = float(analysis_results["bias"]["score"])
    plag_val = float(analysis_results["plagiarism"]["score"])
    readability_val = float(analysis_results["readability"].get("readability_score", 0.0))
    
    inv_bias = 1.0 - bias_val
    inv_plag = 1.0 - plag_val
    
    weighted_sentiment = 0.25 * sentiment_val
    weighted_originality = 0.25 * originality_val
    weighted_inverted_bias = 0.20 * inv_bias
    weighted_inverted_plag = 0.15 * inv_plag
    weighted_readability = 0.15 * readability_val

    score_breakdown = {
        "sentiment": {
            "raw": sentiment_val,
            "weight": 0.25,
            "weighted": round(weighted_sentiment, 3),
            "explanation": "Sentiment score reflects the positive tone of the text."
        },
        "originality": {
            "raw": originality_val,
            "weight": 0.25,
            "weighted": round(weighted_originality, 3),
            "explanation": "Originality is measured by comparing semantic similarity to existing posts."
        },
        "inverted_bias": {
            "raw": round(inv_bias, 3),
            "weight": 0.20,
            "weighted": round(weighted_inverted_bias, 3),
            "explanation": "Inverted bias (1 - bias) rewards neutral content."
        },
        "inverted_plagiarism": {
            "raw": round(inv_plag, 3),
            "weight": 0.15,
            "weighted": round(weighted_inverted_plag, 3),
            "explanation": "Inverted plagiarism (1 - plagiarism) penalizes copied content."
        },
        "readability": {
            "raw": readability_val,
            "weight": 0.15,
            "weighted": round(weighted_readability, 3),
            "explanation": "Readability is based on standard indices, adjusted for text structure and uniqueness."
        }
    }
    
    if penalty_info:
        score_breakdown["length_penalty"] = {
            "applied": True,
            "penalty_factor": 0.8,
            "word_count": word_count,
            "explanation": "Score reduced by 20% due to insufficient content length."
        }
    else:
        score_breakdown["length_penalty"] = {"applied": False, "explanation": "No penalty applied."}

    analysis_results["final_score"] = round(final_score, 3)
    analysis_results["final_score_percentage"] = f"{final_score * 100:.2f}%"
    analysis_results["score_breakdown"] = score_breakdown

    return analysis_results

###############################################################################
#                    BLOCKCHAIN TOKEN DISBURSEMENT FUNCTION                    #
###############################################################################

def disburse_tokens(recipient: str, token_reward: float) -> str:
    """
    Calls the token contract's distributeTokens function.
    Assumes tokenReward is in tokens (float) and converts to base units (e.g. 18 decimals).
    Returns the transaction hash.
    """
    # Assume token has 18 decimals.
    amount = int(token_reward * (10 ** 18))
    nonce = w3.eth.getTransactionCount(ACCOUNT_ADDRESS)
    txn = token_contract.functions.distributeTokens(
        Web3.toChecksumAddress(recipient), amount
    ).buildTransaction({
        "chainId": 1,  # adjust chainId as needed
        "gas": 200000,
        "gasPrice": w3.toWei('5', 'gwei'),
        "nonce": nonce,
    })
    signed_txn = w3.eth.account.signTransaction(txn, private_key=PRIVATE_KEY)
    tx_hash = w3.eth.sendRawTransaction(signed_txn.rawTransaction)
    return tx_hash.hex()

###############################################################################
#                       SAVE/UPDATE ANALYSIS VIA API CALL                     #
###############################################################################

async def save_analysis(analysis_data: Dict[str, Any], post_id: int) -> None:
    payload = {
        "postId": post_id,
        "sentimentAnalysisLabel": analysis_data.get("sentiment", {}).get("label", ""),
        "sentimentAnalysisScore": analysis_data.get("sentiment", {}).get("score", 0.0),
        "biasDetectionScore": analysis_data.get("bias", {}).get("score", 0.5),
        "biasDetectionDirection": "",  # Set as needed.
        "originalityScore": analysis_data.get("originality", {}).get("score", 1.0),
        "similarityScore": analysis_data.get("originality", {}).get("average_similarity", 0.0),
        "readabilityFleschKincaid": analysis_data.get("readability", {}).get("flesch_kincaid_grade", 0.0),
        "readabilityGunningFog": analysis_data.get("readability", {}).get("gunning_fog_index", 0.0),
        "mainTopic": "",               # Update as needed.
        "secondaryTopics": [],         # Update as needed.
        "rating": round(analysis_data["final_score"] * 100),
        "finalScore": analysis_data["final_score"],
        "finalScorePercentage": analysis_data["final_score_percentage"],
        "tokenReward": analysis_data.get("token_reward", 0.0),
        "scoreBreakdown": analysis_data["score_breakdown"]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(API_SAVE_ENDPOINT, json=payload)
        response.raise_for_status()
        saved = response.json()
        logger.info(f"Saved analysis for post {post_id}: {saved}")

async def update_analysis(analysis_data: Dict[str, Any], post_id: int) -> None:
    payload = {
        "tokenReward": analysis_data.get("token_reward", 0.0)
    }
    update_url = f"{API_UPDATE_ENDPOINT}/{post_id}"
    async with httpx.AsyncClient() as client:
        response = await client.put(update_url, json=payload)
        response.raise_for_status()
        updated = response.json()
        logger.info(f"Updated analysis for post {post_id}: {updated}")

###############################################################################
#                                   MAIN DEMO                                 #
###############################################################################

async def main():
    url = "http://localhost:3000/api/posts"
    logger.info(f"Fetching posts from {url} ...")
    try:
        posts = fetch_posts(url)
        logger.info(f"Fetched {len(posts)} post(s). Analyzing each...")
    except Exception as e:
        logger.error(f"Failed to fetch posts: {e}")
        posts = []

    post_results = []
    # Analyze and save analysis for all posts.
    for idx, post in enumerate(posts, start=1):
        content = post.get("content", "")
        if not content:
            logger.warning(f"Post {idx} has no content. Skipping.")
            continue
        analysis_result = await dynamic_analysis(content)
        post_id = post.get("id", idx)
        # Also, include the author's wallet address if available.
        author_wallet = post.get("author", {}).get("walletAddress", None)
        post_results.append({
            "post_id": post_id,
            "final_score": analysis_result["final_score"],
            "analysis": analysis_result,
            "author_wallet": author_wallet
        })
        print(f"\n=== ANALYSIS FOR POST ID {post_id} ===")
        print(json.dumps(analysis_result, indent=2))
        try:
            await save_analysis(analysis_result, post_id)
        except Exception as error:
            logger.error(f"Error saving analysis for post {post_id}: {error}")

    # Compute reward distribution for top 10 posts.
    top_posts = sorted(post_results, key=lambda x: x["final_score"], reverse=True)[:10]
    total_score = sum(item["final_score"] for item in top_posts)
    
    if total_score > 0:
        for item in top_posts:
            token_reward = DAILY_TOKEN_BUDGET * item["final_score"] / total_score
            item["analysis"]["token_reward"] = round(token_reward, 3)
    else:
        equal_share = DAILY_TOKEN_BUDGET / len(top_posts) if top_posts else 0
        for item in top_posts:
            item["analysis"]["token_reward"] = round(equal_share, 3)

    print("\n=== TOP POSTS REWARD DISTRIBUTION ===")
    for item in top_posts:
        print(f"Post ID {item['post_id']}: Final Score = {item['final_score']}, "
              f"Token Reward = {item['analysis']['token_reward']} tokens")
        try:
            # Update the analysis record with the tokenReward.
            await update_analysis(item["analysis"], item["post_id"])
        except Exception as error:
            logger.error(f"Error updating analysis for post {item['post_id']}: {error}")
        # Additionally, call the blockchain to disburse tokens if the author wallet exists.
        if item.get("author_wallet"):
            try:
                tx_hash = disburse_tokens(item["author_wallet"], item["analysis"]["token_reward"])
                logger.info(f"Disbursed {item['analysis']['token_reward']} tokens to {item['author_wallet']} in tx {tx_hash}")
            except Exception as bc_error:
                logger.error(f"Error disbursing tokens for post {item['post_id']}: {bc_error}")
        else:
            logger.warning(f"Post {item['post_id']} has no author wallet; skipping blockchain disbursement.")

if __name__ == "__main__":
    asyncio.run(main())
