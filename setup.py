from sentence_transformers import SentenceTransformer

# local_model_path = "./models/paraphrase-MiniLM-L3-v2"
# model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
# local_model_path = "./models/all-MiniLM-L6-v2"
# model = SentenceTransformer("all-MiniLM-L6-v2")


local_model_path = "./models/all-mpnet-base-v2"
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


model.save(local_model_path)

print(f"Model saved locally at: {local_model_path}")
