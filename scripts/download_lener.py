# Work with datasets 3.6
# venv dataset_old

from datasets import load_dataset

print("Downloading lener_br dataset with old library version...")

dataset = load_dataset("peluz/lener_br", trust_remote_code=True)

dataset.save_to_disk("./lener_br_local")

print("\nDataset successfully downloaded and saved to './lener_br_local'")
print(dataset)