# SarcasmExplanation# ExMore: Multimodal Sarcasm Explanation

This project contains the code for the ExMore model, which is designed for multimodal sarcasm explanation. It takes a text-image pair and generates a textual explanation for why the input might be sarcastic.

## Project Structure

- `data/`: Contains the training, validation, and test datasets.
- `saved_models/`: Directory to save trained model checkpoints.
- `predictions/`: Directory to store the output from the model.
- `src/`: Contains all the source code.
- `requirements.txt`: A list of Python packages required to run the project.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd ExMore_Project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download NLTK data:**
    ```python
    import nltk
    nltk.download('wordnet')
    ```

## How to Run

### 1. Configure the Project
Edit `src/config.py` to set the correct paths for your dataset, saved models, and prediction outputs. You can also adjust hyperparameters in this file.

### 2. Train the Model
To start training the model, run the `train.py` script:
```bash
python src/train.py