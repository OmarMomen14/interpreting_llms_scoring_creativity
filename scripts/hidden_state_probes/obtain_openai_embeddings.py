import os
import pandas as pd
import torch
from openai import OpenAI
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
from tqdm import tqdm

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_NEU"))

d = pd.read_csv("../../data/input/metaphor.csv")
d = d.dropna(subset=["response", "rating"]).reset_index(drop=True)
d["label"] = d["rating"].astype(float)
d["response"] = d["response"].astype(str).str.strip()

d_input = d.filter(["response", "label", "set"], axis=1)
dataset = Dataset.from_pandas(d_input, preserve_index=False)

train_val_test_dataset = DatasetDict(
    {
        "train": dataset.filter(lambda example: example["set"] == "training"),
        "validation": dataset.filter(lambda example: example["set"] == "validation"),
        "test": dataset.filter(lambda example: example["set"] == "test"),
        "heldout": dataset.filter(lambda example: example["set"] == "heldout"),
    }
)


def get_batch_embeddings(texts, batch_size=256):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-large",
        )
        batch_embs = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embs)
    return all_embeddings


tensor = {}

for split in train_val_test_dataset:
    ratings = torch.tensor(train_val_test_dataset[split]["label"], dtype=torch.float32)
    responses = train_val_test_dataset[split]["response"]

    embs = get_batch_embeddings(responses)

    tensor[split] = {
        "embeddings": torch.tensor(embs, dtype=torch.float32),
        "ratings": ratings,
    }

torch.save(tensor, "../../data/input/openai_embeddings.pt")
