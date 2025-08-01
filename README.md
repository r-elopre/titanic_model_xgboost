# Titanic Survival Prediction Model with XGBoost

This repository contains a Python script that builds and evaluates an XGBoost classifier to predict passenger survival on the Titanic, based on a preprocessed dataset.

## Overview

The script uses machine learning to analyze the Titanic dataset and predict whether a passenger survived or not. It includes data preprocessing, model training, evaluation, and visualization of results.

## Features

- Loads and processes a pre-scaled dataset (`train_scaled.csv`).
- Trains an XGBoost classifier with optimized settings.
- Splits data into training and test sets.
- Evaluates model performance with a classification report.
- Visualizes the confusion matrix.
- Saves predictions to a CSV file.

## Requirements

- Python 3.x
- Required libraries:
  - `pandas`
  - `xgboost`
  - `sklearn`
  - `matplotlib`
  - `seaborn`

Install dependencies using:
```bash
pip install pandas xgboost scikit-learn matplotlib seaborn
```

## File Structure

- `data/train_scaled.csv`: Preprocessed dataset with features and target variable.
- `titanic_model_xgboost.py`: Main Python script for model training and evaluation.
- `data/predictions.csv`: Output file containing actual vs. predicted survival outcomes.

## Usage

1. Ensure all dependencies are installed.
2. Place your preprocessed `train_scaled.csv` in the `data` folder.
3. Run the script from the command line:
   ```bash
   python titanic_model_xgboost.py
   ```
4. Check the console for the classification report and view the confusion matrix plot.
5. Find predictions in `data/predictions.csv`.

## Dataset

The script expects a CSV file (`train_scaled.csv`) with columns including `Survived`, `PassengerId`, and various features (e.g., `Pclass`, `Age`, `Sex_male`, etc.). The first two rows of a sample dataset are:
- `PassengerId,Survived,Pclass,Age,SibSp,Parch,Fare,Sex_male,Embarked_q,Embarked_s,Deck_B,Deck_C,Deck_D,Deck_E,Deck_F,Deck_G,Deck_T,Deck_Unknown`
- `1,0,3,-0.5348911628688886,0.4327933656785018,-0.4736736092984604,-0.5641087873940193,True,False,True,False,False,False,False,False,False,False,True`
- `2,1,1,0.6683917564561223,0.4327933656785018,-0.4736736092984604,0.9425480546638041,False,False,False,False,True,False,False,False,False,False,False`

## Results

Running the script generates:
- A classification report (e.g., accuracy ~82%, precision/recall for each class).
- A confusion matrix visualization.
- A `predictions.csv` file with actual and predicted survival outcomes.

## Contributing

Feel free to fork this repository and submit pull requests for improvements or bug fixes.

## License

This project is open-source. See the [LICENSE](LICENSE) file for details (if applicable).