# Housing Price Prediction

This project focuses on predicting housing prices using a dataset from Kaggle. The application uses a web interface built with Streamlit and a Random Forest model for predictions. It also includes SHAP (SHapley Additive exPlanations) for interpreting model outputs.

## Dataset

The dataset used for this project is sourced from Kaggle. It contains features related to housing attributes, such as:
- Lot area
- Year built
- Overall quality
- Number of rooms
- Sale price (target variable)

## Usage

1. Clone this repository:
   ```bash
   git clone <repository_url>
   ```

2. Download the dataset from Kaggle and place it in the project directory.

3. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Use the interactive interface to explore features, visualize data, and make predictions.

## Result

Submissions are evaluated on *Root-Mean-Squared-Error (RMSE)* between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)

SHAP value visualizations highlight the top contributors to housing price predictions.

## Acknowledgments

- Kaggle for providing the dataset
- scikit-learn for the Random Forest implementation
- Streamlit for the interactive interface

## Future Improvements

- Experiment with other models (e.g., Gradient Boosting, XGBoost)
- Perform advanced hyperparameter tuning
- Integrate real-time data updates for predictions
