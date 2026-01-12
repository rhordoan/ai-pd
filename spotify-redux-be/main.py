import pandas as pd
import numpy as np
import mlflow
import optuna
import pickle
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

# ==========================================
# 1. CONFIGURATION
# ==========================================
EXPERIMENT_NAME = "Music_Recommender_SVD_NoFilter"
DATA_PATH = "Lastfm_data.csv"
BEST_RMSE = float('inf')

mlflow.set_experiment(EXPERIMENT_NAME)

# ==========================================
# 2. DATA PREPROCESSING (Filters Removed)
# ==========================================
def load_and_process_data(filepath, sample_size=300000): # Increased sample size
    print("⏳ Loading Data...")
    df = pd.read_csv(filepath, usecols=['Username', 'Artist', 'Track'])
    
    # 1. Create unique Song ID
    df['song_id'] = df['Artist'] + " - " + df['Track']
    
    # 2. Aggregation
    print("⏳ Aggregating Play Counts...")
    df_agg = df.groupby(['Username', 'song_id']).size().reset_index(name='play_count')
    
    # 3. Log Normalization
    df_agg['rating'] = np.log10(df_agg['play_count'] + 1)
    
    
    # Optional: If dataset is massive (>500k rows), sample it to keep training fast
    # But strictly random sampling, not "popularity" filtering.
    if len(df_agg) > sample_size:
        print(f"⚠️ Downsampling to {sample_size} rows for speed...")
        df_agg = df_agg.sample(n=sample_size, random_state=42)

    print(f"✅ Data Ready! {len(df_agg)} interactions loaded.")
    return df_agg

# ==========================================
# 3. OPTUNA OBJECTIVE FUNCTION
# ==========================================
def objective(trial):
    global BEST_RMSE
    
    # Search Space
    params = {
        'n_factors': trial.suggest_int('n_factors', 20, 100),
        'n_epochs': trial.suggest_int('n_epochs', 10, 30),
        'lr_all': trial.suggest_float('lr_all', 0.002, 0.01),
        'reg_all': trial.suggest_float('reg_all', 0.02, 0.1)
    }

    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df[['Username', 'song_id', 'rating']], reader)

    model = SVD(**params)

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        print(f"🔄 Training Trial {trial.number}...")
        
        # Using 3-fold CV
        results = cross_validate(model, data, measures=['RMSE'], cv=3, verbose=False)
        mean_rmse = results['test_rmse'].mean()
        
        mlflow.log_metric("rmse", mean_rmse)
        
        if mean_rmse < BEST_RMSE:
            BEST_RMSE = mean_rmse
            print(f"⭐ New Best Model Found! RMSE: {mean_rmse:.4f}")
            
            # Retrain on full dataset
            trainset = data.build_full_trainset()
            model.fit(trainset)
            
            with open("best_model.pkl", "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact("best_model.pkl")
            
    return mean_rmse

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    df = load_and_process_data(DATA_PATH)
    
    print("🚀 Starting Bayesian Optimization (No Filters)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10) # 10 trials is enough for demo
    
    print("------------------------------------------------")
    print("🏆 Optimization Complete!")
    print(f"Best RMSE: {study.best_value}")