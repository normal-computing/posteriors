from datasets import load_dataset
import pickle 

dataset = load_dataset("pg19", split="test", streaming=True)

subset = []
for i in range(50):
    subset.append(next(iter(dataset)))


with open("./experiments/data/pg19-small.pkl", "wb") as f:
    pickle.dump(subset, f)