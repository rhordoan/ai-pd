import pickle
import numpy as np
import faiss

# 1. Load the SVD Model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. Extract Item Vectors (The "Q" Matrix)
# shape: (Number of Songs, 50)
item_vectors = model.qi 
# FAISS expects float32
item_vectors = item_vectors.astype('float32')

# 3. Create FAISS Index (L2 Distance = Euclidean)
dimension = item_vectors.shape[1] # e.g., 50
index = faiss.IndexFlatL2(dimension)
index.add(item_vectors)

# 4. Save Everything
faiss.write_index(index, "song_vectors.index")
print(f"✅ Indexed {index.ntotal} songs into FAISS.")

# We also need a map to convert FAISS ID (0, 1, 2) back to Song Name ("Metallica")
# The SVD model stores this internally as 'to_raw_iid'
with open('faiss_id_map.pkl', 'wb') as f:
    # Create a simple list where index 0 = Song Name at index 0
    song_map = [model.trainset.to_raw_iid(i) for i in range(model.qi.shape[0])]
    pickle.dump(song_map, f)