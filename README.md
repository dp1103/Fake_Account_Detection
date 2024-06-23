
# Identification of Fake Instagram Accounts

This project aims to detect fake Instagram accounts using machine learning techniques, specifically the Support Vector Machine (SVM) model. The project involves creating a web framework using Flask and designing the frontend with HTML and CSS.

## Abstract

Detecting fake Instagram accounts is critical for maintaining trust and security on the platform. This study employs machine learning techniques, specifically the Support Vector Machine (SVM) model, to identify deceptive Instagram profiles.

We begin by assembling a diverse dataset that includes both genuine and fake Instagram profiles. Extracted features encompass user engagement, posting habits, and content analysis, forming a robust feature set. SVM, a potent supervised learning algorithm, is then trained on this dataset to classify accounts as genuine or deceptive.


## Features

- **SVM Classifier:** A supervised learning algorithm used to classify accounts as genuine or deceptive.
- **Web Framework:** Developed using Flask.
- **Frontend Design:** Implemented using HTML and CSS.

## Requirements

- Python 3.x
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML and CSS

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/fake-instagram-detection.git
   cd fake-instagram-detection
   ```

2. **Create a virtual environment:**
   ```sh
   python3 -m venv venv
   source venv/Scripts/activate  
   ```

3. **Install the required packages:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the dataset:**
   - Ensure your training dataset, `train.csv`, is placed in the `template` directory.

2. **Run the Flask app:**
   ```sh
   python app.py
   ```

3. **Access the web app:**
   - Open your web browser and go to `http://127.0.0.1:5000`.

## Project Structure

- `app.py`: The main Flask application file where the SVM model is trained.
- `static/`: Directory for static files (CSS, images).
- `templates/`: Directory for HTML templates. Ensure `train.csv` is placed here for training a model.
- `requirements.txt`: List of required Python packages.

