# Fake News Detection Pipeline

This repository provides an end-to-end fake news detection pipeline implemented in a Jupyter Notebook. It explores multiple NLP and deep learning approaches to classify news articles as real or fake.

## Repository Structure
```
├── code03
│   ├── code03.ipynb         # Jupyter notebook with the complete model pipeline
│   └── glove.6B.100d.txt     # Pre-trained GloVe embeddings (100-dimensional)
├── data03
│   └── news.csv             # Dataset containing news articles and their labels
├── requirements.txt         # Python package dependencies
└── README.md                # This file
```

## Prerequisites
- Python 3.7 or higher
- pip package manager
- Jupyter Notebook or JupyterLab

## Installation
1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download data and embeddings**
   - Place `news.csv` in the `data03/` folder.
   - Download [GloVe 6B embeddings](https://nlp.stanford.edu/projects/glove/) and copy **`glove.6B.100d.txt`** into the `code03/` folder.

## Configuration
The notebook uses two variables at the top to locate files:
```python
ENVIRONMENT = '/path/to/code03'  # Notebook working directory
DATA_PATH   = 'news.csv'         # Relative path to the dataset inside ENVIRONMENT
```
Update these paths if you move files or run the notebook from another location.

## Usage
1. Navigate to the notebook directory:
   ```bash
   cd code03
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook code03.ipynb
   ```
3. Run all cells in order. The notebook covers:
   - Data loading and preprocessing
   - Exploratory data analysis and visualization
   - Feature engineering and label mapping
   - Model training for:
     - Bag-of-Words + Logistic Regression
     - TF-IDF + Logistic Regression
     - GloVe embeddings + Logistic Regression
     - GloVe embeddings + LSTM
     - BERT fine-tuning
     - Sentence Transformer embeddings + Logistic Regression
   - Evaluation, confusion matrices, and comparison plots
   - Exporting trained model files and inference examples

## Outputs
- **Model files** (in `code03/`):
  - `bow_logistic_model.pkl`
  - `tfidf_logistic_model.pkl`
  - `glove_logistic_model.pkl`
  - `glove_lstm_model.pt`
  - `bert_model.pt`
  - `sentence_transformer_model.pkl`
  - `all_models.pkl`
- **Plots** (PNG format):
  - `*_confusion_matrix.png`
  - Model comparison bar chart

## requirements.txt
See the root of this repository for a list of required Python packages. Install them via:
```bash
pip install -r requirements.txt
```

## License
This project is licensed under the MIT License.

## Contact
For questions or issues, please open a GitHub issue or reach out at zhengqianchiu@gmail.com
