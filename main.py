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
import httpx  # make sure to install httpx: pip install httpx

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

# Environment-based configuration
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
QDRANT_URL = os.getenv("QDRANT_URL", ":memory:")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "text_embeddings")

# Daily token budget for rewards.
DAILY_TOKEN_BUDGET = 1000

# API endpoints.
API_SAVE_ENDPOINT = "http://localhost:3000/api/ratings/ai"
# Assuming your API supports updates via PUT at this URL with /{postId} appended.
API_UPDATE_ENDPOINT = "http://localhost:3000/api/ratings/ai"

###############################################################################
#                           ONE-TIME GLOBAL LOAD                               #
###############################################################################

logger.info("Loading spaCy model (once)...")
nlp = spacy.load(SPACY_MODEL, disable=["ner", "parser"])

logger.info("Loading embedding model (once)...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

logger.info("Initializing HF sentiment pipeline with truncation (once)...")
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    truncation=True,
    max_length=512
)

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
    Remove large quoted blocks (e.g., blocks of text in quotes) that may be
    present to game plagiarism detection. This is a simple heuristic.
    """
    # Remove text between double quotes spanning more than 30 words.
    return re.sub(r'"(?:\S+\s+){30,}"', '', text)

###############################################################################
#                            ANALYSIS FUNCTIONS                               #
###############################################################################

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Use a Hugging Face pipeline for sentiment analysis (0..1).
    Higher = more positive.
    """
    result = sentiment_analyzer(text)[0]
    label = result["label"]  # e.g. "POSITIVE" or "NEGATIVE"
    raw_score = float(result["score"])
    if label.upper() == "NEGATIVE":
        score = 1.0 - raw_score
    else:
        score = raw_score
    return {"label": label, "score": round(score, 3)}

def detect_bias(text: str) -> Dict[str, float]:
    """
    Return a bias score 0..1, where 0 = neutral and 1 = strongly biased.
    If Ollama is installed, use it; otherwise, use zero-shot classification as a fallback.
    """
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
    """
    Originality check with Qdrant: search top-K neighbors,
    compute average similarity, originality = 1 - avg_sim.
    Then store the text for future references.
    """
    try:
        K = 5
        embedding = get_embedding(text)
        results = client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=embedding,
            limit=K
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
    """
    Plagiarism check: search top-1 in Qdrant after removing large quoted blocks,
    store text, return similarity as 'score'.
    0 => no overlap, 1 => near identical.
    """
    try:
        cleaned_text = remove_large_quotes(text)
        embedding = get_embedding(cleaned_text)
        results = client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=embedding,
            limit=1
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
    """
    Compute Flesch-Kincaid and Gunning Fog with textstat.
    Additionally, check for natural paragraph structure and text uniqueness 
    to prevent artificially gaming the readability score.
    """
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
    """
    Combine sentiment, originality, (1-bias), (1-plagiarism), readability 
    into one final 0..1 score using recommended weights:
      - sentiment:        25%
      - originality:      25%
      - (1 - bias):       20%
      - (1 - plagiarism): 15%
      - readability:      15%
    """
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
    """
    Run the selected analysis steps concurrently on the given text,
    then compute a 'final_score' with the recommended formula.
    Additionally, if the text is too short (indicating low effort),
    apply a penalty to the final score.
    Also, add a detailed breakdown of points explaining the score.
    """
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
#                       SAVE/UPDATE ANALYSIS VIA API CALL                     #
###############################################################################

async def save_analysis(analysis_data: Dict[str, Any], post_id: int) -> None:
    """
    Save the analysis result by calling the API endpoint.
    Constructs the payload expected by the API and makes an async POST request.
    """
    payload = {
        "postId": post_id,
        "sentimentAnalysisLabel": analysis_data.get("sentiment", {}).get("label", ""),
        "sentimentAnalysisScore": analysis_data.get("sentiment", {}).get("score", 0.0),
        "biasDetectionScore": analysis_data.get("bias", {}).get("score", 0.5),
        "biasDetectionDirection": "",  # Set a default or compute as needed.
        "originalityScore": analysis_data.get("originality", {}).get("score", 1.0),
        "similarityScore": analysis_data.get("originality", {}).get("average_similarity", 0.0),
        "readabilityFleschKincaid": analysis_data.get("readability", {}).get("flesch_kincaid_grade", 0.0),
        "readabilityGunningFog": analysis_data.get("readability", {}).get("gunning_fog_index", 0.0),
        "mainTopic": "",  # Update as needed.
        "secondaryTopics": [],  # Update as needed.
        "rating": round(analysis_data["final_score"] * 100),  # Convert final score (0-1) to an integer rating.
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
    """
    Update the tokenReward field (and optionally other fields) for an existing AI post rating.
    Assumes your API supports PUT requests at /api/ratings/ai/{postId}.
    """
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
    # First, analyze and save all posts without tokenReward computed.
    for idx, post in enumerate(posts, start=1):
        content = post.get("content", "")
        if not content:
            logger.warning(f"Post {idx} has no content. Skipping.")
            continue
        analysis_result = await dynamic_analysis(content)
        post_id = post.get("id", idx)
        post_results.append({
            "post_id": post_id,
            "final_score": analysis_result["final_score"],
            "analysis": analysis_result
        })
        print(f"\n=== ANALYSIS FOR POST ID {post_id} ===")
        print(json.dumps(analysis_result, indent=2))
        try:
            await save_analysis(analysis_result, post_id)
        except Exception as error:
            logger.error(f"Error saving analysis for post {post_id}: {error}")

    # Now, compute reward distribution for the top 10 posts.
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
            # Update the saved analysis record with the tokenReward.
            await update_analysis(item["analysis"], item["post_id"])
        except Exception as error:
            logger.error(f"Error updating analysis for post {item['post_id']}: {error}")

if __name__ == "__main__":
    asyncio.run(main())
