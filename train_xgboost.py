import os
import glob
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATA_FOLDER = "./training_setA" 

def load_and_preprocess(folder_path, max_files=500):
    print("Loading PhysioNet .psv files...")
    all_files = glob.glob(os.path.join(folder_path, "*.psv"))
    
    # We limit to max_files so your laptop doesn't freeze during testing
    file_list = all_files[:max_files] 
    
    df_list = []
    for file in file_list:
        df = pd.read_csv(file, sep='|')
        
        # We use Forward Fill (LOCF): carry the last known vital sign forward
        df = df.ffill().fillna(0) 
        df_list.append(df)
        
    master_df = pd.concat(df_list, ignore_index=True)
    return master_df

def train():
    df = load_and_preprocess(DATA_FOLDER)
    
    # Map PhysioNet's exact column names to the ones your FastAPI backend already uses
    X = df[['HR', 'Temp', 'Resp', 'WBC', 'SBP']].copy()
    X.columns = ['heart_rate', 'temp', 'resp_rate', 'wbc', 'systolic_bp']
    
    y = df['SepsisLabel']
    
    print("Splitting data and training XGBoost...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Calculate the class weight to handle the imbalanced dataset
    # scale_pos_weight = sum(negative instances) / sum(positive instances)
    weight_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Applying scale_pos_weight of {weight_ratio:.2f} to prioritize sepsis cases.")
    
    # Initialize and train the Gradient-Boosted Model with scale_pos_weight
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        scale_pos_weight=weight_ratio,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    print("\n--- Model Evaluation ---")
    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print(classification_report(y_test, predictions))
    
    # Save the new model over the old one
    joblib.dump(model, "sepsis_model.pkl")
    print("\nSuccess! Saved new weighted XGBoost model as sepsis_model.pkl")

if __name__ == "__main__":
    train()