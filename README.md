# Analyse Post

Analyse Post is a Python-based script that performs multi-dimensional analysis on text posts. It uses several Natural Language Processing (NLP) techniques to evaluate sentiment, detect bias, check for originality and plagiarism, and assess readability. Based on these evaluations, it computes a composite score for each post. Additionally, it integrates with blockchain smart contracts to disburse token rewards to authors based on the quality of their posts and communicates analysis results with a backend API.

## Features

- **Sentiment Analysis:**  
  Uses Hugging Face’s sentiment pipeline to assess the overall tone of the text.

- **Bias Detection:**  
  Detects bias using either the Ollama library (if installed) or a zero-shot classification model from Hugging Face.

- **Originality & Plagiarism Checks:**  
  Leverages SentenceTransformers and Qdrant to compute text embeddings and compare them against existing posts for originality and plagiarism.

- **Readability Evaluation:**  
  Utilizes textstat to calculate readability metrics (Flesch-Kincaid Grade, Gunning Fog Index) adjusted for text structure and lexical uniqueness.

- **Composite Scoring:**  
  Aggregates individual scores (sentiment, originality, bias, plagiarism, readability) into a final score using a weighted sum. A length penalty is applied if the post contains fewer than 300 words.

- **Blockchain Integration:**  
  Uses Web3 to interact with Ethereum-based smart contracts. The script disburses tokens as rewards by calling a voting contract function based on the final post score.

- **API Integration:**  
  Saves and updates analysis results via REST API calls, allowing easy integration with a broader system.

## Requirements

- Python 3.8+
- [Poetry](https://python-poetry.org/) for dependency management
- External libraries:
  - `spacy` (with an English model, e.g., `en_core_web_sm`)
  - `transformers`
  - `sentence_transformers`
  - `textstat`
  - `qdrant-client`
  - `web3`
  - `httpx`
  - `requests`
  - Other standard libraries: `os`, `json`, `logging`, `asyncio`, etc.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install Dependencies with Poetry:**

   If you want to manage dependencies without installing the project as a package, run:

   ```bash
   poetry install --no-root
   ```

   If you intend to install the project as a package, ensure that your `pyproject.toml` specifies the correct package location, then run:

   ```bash
   poetry install
   ```

3. **Configure Environment Variables:**

   Create a `.env` file or export the following variables as needed:

   - `SPACY_MODEL` (default: `en_core_web_sm`)
   - `EMBEDDING_MODEL_NAME` (default: `all-MiniLM-L6-v2`)
   - `QDRANT_URL` (default: `:memory:`)
   - `QDRANT_COLLECTION_NAME` (default: `text_embeddings`)
   - `WEB3_PROVIDER` (e.g., `https://base-sepolia.g.alchemy.com/v2/your-api-key`)
   - `TOKEN_CONTRACT_ADDRESS` (Ethereum address for the token contract)
   - `VOTING_CONTRACT_ADDRESS` (Ethereum address for the voting contract)
   - `PRIVATE_KEY` (Your Ethereum wallet’s private key)
   - `ACCOUNT_ADDRESS` (Your Ethereum wallet’s public address)

4. **Smart Contract ABI Files:**

   Ensure that the ABI files (`CurateAIToken.json` and `CurateAIVote.json`) are available in the `abi/` directory.

## Usage

Run the main script using Poetry:

```bash
poetry run python main.py
```

The script will:

- Fetch posts from a specified API endpoint.
- Process each post’s content through several NLP analysis functions.
- Compute a final score based on sentiment, bias, originality, plagiarism, and readability.
- Save the analysis results via a REST API.
- Calculate token rewards based on the final score.
- Optionally disburse tokens via blockchain transactions if the author’s wallet address is provided.

## Detailed Functionality

### NLP Analysis

- **Dynamic Analysis:**  
  The script uses spaCy to clean and preprocess text, then runs concurrent analysis functions for:

  - **Sentiment Analysis:** Determines the positive or negative tone.
  - **Bias Detection:** Evaluates if the text is neutral or biased.
  - **Originality & Plagiarism Checks:** Uses semantic similarity via vector embeddings.
  - **Readability:** Computes Flesch-Kincaid and Gunning Fog scores, adjusting for structure and word uniqueness.

- **Final Scoring:**  
  Each metric is weighted, and the final composite score is calculated. If a post contains fewer than 300 words, a 20% penalty is applied.

### Blockchain Token Disbursement

- **Integration with Web3:**  
  The script connects to an Ethereum provider and uses smart contract functions to reward authors.
- **Reward Calculation:**  
  Token rewards are computed relative to the final score and a predetermined daily token budget.
- **Transaction Execution:**  
  If an author’s wallet address is available, the script calls the `aiVote` function on the voting contract to disburse tokens.

### API Integration

- **Saving Analysis:**  
  Analysis results are posted to an API endpoint for persistent storage.
- **Updating Analysis:**  
  Once token rewards are computed, the analysis records are updated through a PUT request to the backend.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```

This README provides an overview of the script's purpose, key features, installation instructions, and detailed explanations of its functionality. Feel free to customize it further based on your project’s specifics or additional requirements.
```
