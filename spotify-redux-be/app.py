import pickle
import numpy as np
import faiss
import sys

# ==========================================
# 1. CONFIGURATION & INPUTS
# ==========================================
# SIMULATED USER INPUT: The 3 songs a new user selects during onboarding
# (Change these to test different "Tastes")
INPUT_SONGS = [
   # "Kendrick Lamar - Money Trees",
    #"Kendrick Lamar - These Walls", 
    #"Frank Ocean - Seigfried",
    #"Isaiah Rashad - West Savannah (feat. SZA)",
    #"Mac Miller - Self Care",
    #"Yung Lean - Kyoto",
    #"Travis Scott - SICKO MODE",
    "Kanye West - Saint Pablo",
    "Kanye West - Ultralight Beam",
    #"Childish Gambino - Redbone",
    #"Mac Miller - Woods",
    "Mac Miller - Come Back to Earth",
    "Kanye West - Ghost Town",
]

MODEL_PATH = "best_model.pkl"

# ==========================================
# 2. SYSTEM SETUP (The "Engine")
# ==========================================
def load_and_build_engine():
    print("⏳ Loading SVD Model...")
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Error: '{MODEL_PATH}' not found. Run the training script first.")
        sys.exit(1)

    print("⚙️ Building FAISS Index from Item Vectors...")
    # 1. Extract the Item Factors (The 'Q' Matrix) from SVD
    # Shape: (Number of Songs, Number of Factors)
    item_vectors = model.qi
    
    # 2. FAISS requires float32 data
    item_vectors = item_vectors.astype('float32')

    # 3. Create the Index (L2 = Euclidean Distance)
    # This finds vectors that are geometrically close to each other
    dimension = item_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(item_vectors)
    
    print(f"✅ Indexed {index.ntotal} songs in vector space.")
    return model, index

# ==========================================
# 3. THE COLD START LOGIC
# ==========================================
def get_cold_start_recommendations(model, index, selected_song_names, k=10):
    print(f"\n🧠 Analyzing Taste Profile for: {selected_song_names}")
    
    valid_vectors = []
    
    # 1. Get Vectors for Selected Songs
    for name in selected_song_names:
        try:
            # Convert String Name -> Internal Integer ID
            inner_id = model.trainset.to_inner_iid(name)
            
            # Extract the vector for this specific song
            vector = model.qi[inner_id]
            valid_vectors.append(vector)
            print(f"   ✅ Found vector for: {name}")
        except ValueError:
            print(f"   ⚠️ Warning: '{name}' not found in DB. Skipping.")
            # (In a real app, you would auto-correct spelling here)

    if not valid_vectors:
        return ["Error: No valid songs found in selection."]

    # 2. Calculate the "Pseudo-User" Vector (The Centroid)
    # We average the vectors of the songs they liked.
    # This creates a new point in space representing their "Vibe".
    pseudo_user_vector = np.mean(valid_vectors, axis=0).astype('float32')
    
    # Reshape for FAISS (1, Dimensions)
    query_vector = np.expand_dims(pseudo_user_vector, axis=0)
    
    # 3. Search FAISS for Nearest Neighbors
    # We ask for k + len(input) because the search might return the input songs themselves
    distances, indices = index.search(query_vector, k + len(selected_song_names))
    
    # 4. Map Results back to Song Names
    recommendations = []
    found_count = 0
    
    # indices[0] is the list of nearest neighbor IDs
    for idx in indices[0]:
        # Convert Internal Integer ID -> String Name
        song_name = model.trainset.to_raw_iid(idx)
        
        # Filter: Don't recommend the exact songs they just picked
        if song_name not in selected_song_names:
            recommendations.append(song_name)
            found_count += 1
            
        if found_count >= k:
            break
            
    return recommendations

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Initialize
    model, index = load_and_build_engine()
    
    # 2. Run the "Hybrid / Cold Start" Algorithm
    recs = get_cold_start_recommendations(model, index, INPUT_SONGS, k=10)
    
    # 3. Display Results
    print("\n" + "="*60)
    print("🚀 COLD START RECOMMENDATIONS (FAISS)")
    print("="*60)
    for i, song in enumerate(recs):
        print(f"{i+1}. {song}")
    print("="*60)
    print("Interpretation: These songs are geometrically closest to the average")
    print("of the input songs in the SVD latent space.")