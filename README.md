
# Identification of Suicide Risk Factors in the Penitentiary Population

This repository contains the code used to analyze and generate a machine learning model based on blending techniques to classify suicide risk in Spanish prisoners. This project is part of the master's dissertation of **Silvana Ruiz Medina** for the Master's in Bioinformatics and Computational Biology at the **Universidad Autónoma de Madrid**.

## Project Description

Suicide risk among inmates in developed countries is substantially higher than in the general population. This study uses anonymized historical data provided by Spanish penitentiary institutions to identify predictive factors of suicidal behavior. 

The goal is to support prevention by applying machine learning techniques — from traditional models to deep learning — to detect high-risk profiles in a scalable and interpretable way.

**Disclaimer**: The data used in this project is confidential and not included in this repository.

---

## Repository Structure

```
.
├── Code/
│   ├── EDA_*.ipynb         # Exploratory Data Analysis scripts
│   ├── concat_df_*.ipynb   # Data concatenation and preprocessing
│   ├── modelo_*.ipynb      # ML model training
│   └── prueba_model_*.ipynb# Model testing and evaluation
├── Objetos/                # Serialized objects (models, encoders, etc.)
└── requirements.txt  # Python package dependencies
```

---

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/SilvanaRuiz/TFM.git
   cd TFM
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage Workflow

Due to the academic nature of the project and data confidentiality, this code is intended for educational and research purposes only. Here's the logical workflow to run the analysis:

1. **EDA Phase** – Analyze and understand the structure and distribution of the data.
2. **Preprocessing Phase** – Concatenate, clean, and prepare data using `concat_df_*.ipynb`.
3. **Modeling Phase** – Train predictive models with `modelo_*.ipynb`.
4. **Evaluation Phase** – Evaluate and interpret model performance with `prueba_model_*.ipynb`.

---

## Data Privacy

The dataset used in this study contains sensitive information about inmates and **is not publicly available**. All data processing and modeling has been done in compliance with ethical and legal standards regarding confidentiality.

---

## Author

**Silvana Ruiz Medina**  
Master's student in Bioinformatics and Computational Biology     
**Universidad Autónoma de Madrid**  
Supervised by:  
- Dr. Hilario Blasco-Fontecilla (External Director)  
- Dr. Aythami Morales Moreno (Academic Tutor)


