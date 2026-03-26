## Quick Use

Follow these steps to run the complete text alignment pipeline (from embedding generation to final alignment results) with a single command.

### 1. Data Preparation
Organize your raw text files (`.txt`) and ground truth labels according to the following directory structure:

```text
project_root/
├── data/
│   └── sample_data/
│       ├── vi/                # Source Vietnamese files
│       ├── zh/                # Target Chinese files
│       └── ground_truth.tsv   # (Optional) Ground truth file for evaluation
├── main/
│   └── run_all.sh             # Master control script (Entry point)
└── lib/                       # Core processing modules (.py)
```

### 2. Configuration
Open `main/run_all`.sh to adjust the hyperparameters. The default configuration is optimized for historical Hán-Việt texts using the LaBSE model and Sentence-level (n=1) granularity:
```text
# Key hyperparameters in run_all.sh
MODEL_PATH="sentence-transformers/LaBSE"
SPLIT_MODE="sentence"
NUM_SENT=1         # Keep n=1 for the highest semantic resolution
CSLS_K=15          # Increase to 20 to penalize "Hub" documents (Hubness)
THRESHOLD=0.08     # Final similarity score threshold (Edge filtering)
```
### 3. Execution
Navigate to the main directory, grant execution permissions, and run the script:
```bash
cd main
chmod +x run_all.sh
./run_all.sh
``` 

### 4. Output & Evaluation
Once the pipeline completes, the following directories will be created/updated:

* **`db_embeddings/`**: Stores feature vectors (`.npz`) and metadata for each language.
* **`results/`**: 
    * `alignment_results.tsv`: A list of aligned document pairs with confidence scores.
    * `confusion_matrix.png`: (If `--viz` is enabled) A visual representation of alignment accuracy compared to the Ground Truth.