import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from io import BytesIO

# Set page config
st.set_page_config(page_title="Credit Risk Predictor", page_icon="💰", layout="wide")

# Title and description
st.title("💰 Credit Risk Predictor")
st.markdown("""
This app predicts credit risk based on customer characteristics. 
Upload your data or use the sample data to make predictions.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a page", ["Data Exploration", "Model Training", "Make Predictions"])

# Load data function
@st.cache_data
def load_data():
    df = pd.read_csv("german_credit_data.csv", index_col=0)
    return df

# Load the data
try:
    df = load_data()
except:
    st.error("Unable to load the sample data. Please upload your own data.")
    df = pd.DataFrame()

# Data Exploration Page
if app_mode == "Data Exploration":
    st.header("Data Exploration")
    
    if not df.empty:
        # Show raw data
        if st.checkbox("Show raw data"):
            st.subheader("Raw Data")
            st.write(df)
        
        # Data summary
        st.subheader("Data Summary")
        st.write(f"Number of records: {df.shape[0]}")
        st.write(f"Number of features: {df.shape[1]}")
        
        # Show column descriptions
        if st.checkbox("Show column descriptions"):
            col_desc = {
                "Age": "Age of the customer",
                "Sex": "Gender of the customer (male/female)",
                "Job": "Job category (0-unskilled, 1-skilled, 2-highly skilled, 3-management)",
                "Housing": "Housing situation (own/rent/free)",
                "Saving accounts": "Saving account status",
                "Checking account": "Checking account status",
                "Credit amount": "Amount of credit",
                "Duration": "Duration of credit in months",
                "Purpose": "Purpose of the credit",
                "credit_risk": "Credit risk (0=good, 1=bad)"
            }
            st.table(pd.DataFrame.from_dict(col_desc, orient='index', columns=["Description"]))
        
        # Show missing values
        if st.checkbox("Show missing values"):
            st.subheader("Missing Values")
            missing = df.isnull().sum().to_frame("Missing Values")
            missing["Percentage"] = (missing["Missing Values"] / len(df)) * 100
            st.table(missing)
        
        # Show distribution of credit risk
        st.subheader("Credit Risk Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='credit_risk', data=df, ax=ax)
        ax.set_title("Distribution of Credit Risk")
        st.pyplot(fig)
        
        # Show numerical features distribution
        st.subheader("Numerical Features Distribution")
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        selected_num_col = st.selectbox("Select numerical feature", num_cols)
        
        fig, ax = plt.subplots()
        sns.histplot(df[selected_num_col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {selected_num_col}")
        st.pyplot(fig)
        
        # Show categorical features distribution
        st.subheader("Categorical Features Distribution")
        cat_cols = df.select_dtypes(include=['object']).columns
        selected_cat_col = st.selectbox("Select categorical feature", cat_cols)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.countplot(x=selected_cat_col, data=df, ax=ax)
        ax.set_title(f"Distribution of {selected_cat_col}")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Show correlation matrix
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Please upload data to explore")

# Model Training Page
elif app_mode == "Model Training":
    st.header("Model Training")
    
    if not df.empty:
        # Preprocessing
        st.subheader("Data Preprocessing")
        # Drop rows where target is NaN
        df = df.dropna(subset=["credit_risk"])

# Continue as before
       
        # Define features and target
        X = df.drop('credit_risk', axis=1)
        y = df['credit_risk']
        
        # Split data
        test_size = st.slider("Test set size (%)", 10, 40, 20)
        random_state = st.slider("Random state", 0, 100, 42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=random_state, stratify=y
        )
        
        st.write(f"Training set size: {X_train.shape[0]} samples")
        st.write(f"Test set size: {X_test.shape[0]} samples")
        
        # Define preprocessing for numerical and categorical features
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Model selection
        st.subheader("Model Selection and Training")
        model_name = st.selectbox("Select model", 
                                 ["Random Forest", "Gradient Boosting", "Logistic Regression"])
        
        # Hyperparameter tuning
        st.subheader("Hyperparameter Tuning")
        cv_folds = st.slider("Number of cross-validation folds", 3, 10, 5)
        
        if model_name == "Random Forest":
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 5, 10],
                'classifier__min_samples_split': [2, 5, 10]
            }
            model = RandomForestClassifier(random_state=random_state)
        elif model_name == "Gradient Boosting":
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.5],
                'classifier__max_depth': [3, 5, 7]
            }
            model = GradientBoostingClassifier(random_state=random_state)
        else:  # Logistic Regression
            param_grid = {
                'classifier__C': [0.1, 1, 10],
                'classifier__penalty': ['l2'],
                'classifier__solver': ['lbfgs', 'liblinear']
            }
            model = LogisticRegression(random_state=random_state, max_iter=1000)
        
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train model
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                # Grid search with cross-validation
                grid_search = GridSearchCV(
                    pipeline,
                    param_grid=param_grid,
                    cv=cv_folds,
                    scoring='f1',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                
                # Best model
                best_model = grid_search.best_estimator_
                
                # Save model
                joblib.dump(best_model, 'credit_risk_model.pkl')
                
                # Display best parameters
                st.subheader("Best Parameters")
                st.write(grid_search.best_params_)
                
                # Make predictions
                y_pred = best_model.predict(X_test)
                
                # Evaluation metrics
                st.subheader("Model Evaluation")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
                col2.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
                col3.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
                col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")
                
                # Classification report
                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred))
                
                # Confusion matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
                
                # Feature importance (for tree-based models)
                if model_name in ["Random Forest", "Gradient Boosting"]:
                    try:
                        st.subheader("Feature Importance")
                        # Get feature names after one-hot encoding
                        preprocessor.fit(X_train)
                        feature_names = numeric_features.tolist()
                        ohe_categories = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].categories_
                        for i, cat in enumerate(categorical_features):
                            feature_names.extend([f"{cat}_{val}" for val in ohe_categories[i]])
                        
                        # Get feature importances
                        importances = best_model.named_steps['classifier'].feature_importances_
                        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                        importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                        ax.set_title('Top 10 Important Features')
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Could not display feature importance: {e}")
                
                # Cross-validation results
                st.subheader("Cross-Validation Results")
                cv_results = pd.DataFrame(grid_search.cv_results_)
                st.write(cv_results[['params', 'mean_test_score', 'std_test_score']])
                
                # Download model
                st.subheader("Download Model")
                with open('credit_risk_model.pkl', 'rb') as f:
                    st.download_button(
                        label="Download trained model",
                        data=f,
                        file_name="credit_risk_model.pkl",
                        mime="application/octet-stream"
                    )
    else:
        st.warning("Please upload data to train models")

# Make Predictions Page
elif app_mode == "Make Predictions":
    st.header("Make Predictions")
    
    # Load model
    try:
        model = joblib.load('credit_risk_model.pkl')
        st.success("Model loaded successfully!")
    except:
        st.error("No trained model found. Please train a model first.")
        model = None
    
    if model is not None:
        # Create input form
        st.subheader("Enter Customer Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            sex = st.selectbox("Sex", ["male", "female"])
            job = st.selectbox("Job", [0, 1, 2, 3], format_func=lambda x: 
                             ["Unskilled", "Skilled", "Highly Skilled", "Management"][x])
            housing = st.selectbox("Housing", ["own", "rent", "free"])
            saving_accounts = st.selectbox("Saving Accounts", 
                                         ["little", "moderate", "quite rich", "rich", "NA"])
        
        with col2:
            checking_account = st.selectbox("Checking Account", 
                                          ["little", "moderate", "quite rich", "rich", "NA"])
            credit_amount = st.number_input("Credit Amount", min_value=0, value=5000)
            duration = st.number_input("Duration (months)", min_value=1, max_value=120, value=12)
            purpose = st.selectbox("Purpose", [
                "radio/TV", "education", "furniture/equipment", "car", 
                "business", "domestic appliances", "repairs", "vacation/others"
            ])
        
        # Create input dataframe
        input_data = pd.DataFrame({
            "Age": [age],
            "Sex": [sex],
            "Job": [job],
            "Housing": [housing],
            "Saving accounts": [saving_accounts],
            "Checking account": [checking_account],
            "Credit amount": [credit_amount],
            "Duration": [duration],
            "Purpose": [purpose]
        })
        
        # Make prediction
        if st.button("Predict Credit Risk"):
            try:
                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data)
                
                st.subheader("Prediction Result")
                
                if prediction[0] == 0:
                    st.success("✅ Low Credit Risk (Good)")
                else:
                    st.error("❌ High Credit Risk (Bad)")
                
                st.write(f"Probability of Low Risk: {prediction_proba[0][0]:.2f}")
                st.write(f"Probability of High Risk: {prediction_proba[0][1]:.2f}")
                
                # Show feature contributions (for tree-based models)
                if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                    st.subheader("Feature Contributions")
                    
                    # Get feature names
                    preprocessor = model.named_steps['preprocessor']
                    preprocessor.fit(input_data)  # Fit to get feature names
                    
                    numeric_features = input_data.select_dtypes(include=['int64', 'float64']).columns
                    categorical_features = input_data.select_dtypes(include=['object']).columns
                    
                    feature_names = numeric_features.tolist()
                    ohe_categories = preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_
                    for i, cat in enumerate(categorical_features):
                        feature_names.extend([f"{cat}_{val}" for val in ohe_categories[i]])
                    
                    # Get SHAP values if possible
                    try:
                        import shap
                        explainer = shap.TreeExplainer(model.named_steps['classifier'])
                        transformed_input = preprocessor.transform(input_data)
                        if hasattr(transformed_input, 'toarray'):  # for sparse matrices
                            transformed_input = transformed_input.toarray()
                        shap_values = explainer.shap_values(transformed_input)
                        
                        fig, ax = plt.subplots()
                        shap.summary_plot(shap_values[1], transformed_input, feature_names=feature_names, plot_type="bar", show=False)
                        st.pyplot(fig)
                    except:
                        st.warning("Could not display SHAP values. Showing feature importances instead.")
                        importances = model.named_steps['classifier'].feature_importances_
                        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                        importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                        ax.set_title('Top 10 Important Features')
                        st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
        
        # Batch prediction option
        st.subheader("Batch Prediction")
        uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])
        
        if uploaded_file is not None:
            try:
                batch_data = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.write(batch_data.head())
                
                if st.button("Predict Batch"):
                    with st.spinner("Making predictions..."):
                        batch_predictions = model.predict(batch_data)
                        batch_proba = model.predict_proba(batch_data)
                        
                        result_df = batch_data.copy()
                        result_df['Predicted_Risk'] = batch_predictions
                        result_df['Probability_Low_Risk'] = batch_proba[:, 0]
                        result_df['Probability_High_Risk'] = batch_proba[:, 1]
                        
                        st.success("Predictions completed!")
                        st.write(result_df)
                        
                        # Download results
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download predictions",
                            data=csv,
                            file_name="credit_risk_predictions.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"Error processing batch file: {e}")

# Run the app
if __name__ == "__main__":
    pass