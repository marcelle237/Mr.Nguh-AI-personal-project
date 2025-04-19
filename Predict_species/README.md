# Animal Species Predictor Web App

This web application allows users to upload images of animals and predicts the species using a VGG-16 deep learning model trained on the Animals-10 dataset from Kaggle.

## Project Structure

- `requirements.txt`: Python dependencies
- `download_dataset.py`: Script to download Animals-10 dataset from Kaggle
- `train_model.py`: Script to train the VGG-16 model
- `app/`: Web app code (Flask backend, templates, static files)

## Setup Instructions

### 1. Clone the Repository
```
git clone <your-repo-url>
cd animal_species_predictor
```

### 2. Set Up Virtual Environment
```
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Kaggle API Setup (for dataset download)
- Create a Kaggle account at https://www.kaggle.com/
- Go to 'Account' -> 'API' -> 'Create New API Token'. This downloads `kaggle.json`.
- Place `kaggle.json` in a safe location (e.g., your user home directory or project root).

### 5. Download Dataset
```
python download_dataset.py
```

### 6. Train the Model
```
python train_model.py
```

### 7. Run the Web App
```
cd app
python app.py
```

---

## Next Steps
- Download and preprocess the dataset
- Train the model
- Launch the web app and upload images for prediction
