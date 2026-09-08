import torch
from tqdm.auto import tqdm
import transformers
import tqdm
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import GPT2Tokenizer

# init
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding_side="right")
tokenizer.pad_token = tokenizer.eos_token

config = transformers.AutoConfig.from_pretrained("gpt2", num_labels=1)
model = transformers.AutoModelForSequenceClassification.from_config(config)

state_dict = torch.load("../../data/input/pytorch_model_gpt.bin")
model.load_state_dict(state_dict, strict=False)

model.to("cuda")


d = pd.read_csv("../../data/input/metaphor.csv")
d["label"] = d["rating"]
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


def tokenize_function(text, max_length=256):
    return tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


# for every text in a list of metaphors get the average embedding of the metaphor for each gpt2 layer (including initial embeddings)
def get_average_embeddings(metaphors):
    model.eval()
    all_layers_embeds = [[] for _ in range(model.config.num_hidden_layers + 1)]

    for text in tqdm.tqdm(metaphors):
        tokenized = tokenize_function(text)
        tokenized = {k: v.to("cuda") for k, v in tokenized.items()}
        with torch.no_grad():
            outputs = model(**tokenized, output_hidden_states=True, return_dict=True)

        for layer in range(model.config.num_hidden_layers + 1):
            embed = (
                outputs.hidden_states[layer][0].sum(dim=0)
                / (outputs.hidden_states[layer][0].shape[0])
            ).cpu()

            all_layers_embeds[layer].append(embed)

    return torch.stack([torch.stack(layer_list) for layer_list in all_layers_embeds])


# for every text in a list of metaphors get the embedding of the last token of the metaphor for each gpt2 layer (including initial embeddings)
def get_last_token_embeddings(metaphors):
    model.eval()
    all_layers_embeds = [[] for _ in range(model.config.num_hidden_layers + 1)]

    for text in tqdm.tqdm(metaphors):
        tokenized = tokenize_function(text)
        tokenized = {k: v.to("cuda") for k, v in tokenized.items()}
        with torch.no_grad():
            outputs = model(**tokenized, output_hidden_states=True, return_dict=True)

        for layer in range(model.config.num_hidden_layers + 1):
            embed = outputs.hidden_states[layer][0][-1].cpu()
            all_layers_embeds[layer].append(embed)

    return torch.stack([torch.stack(layer_list) for layer_list in all_layers_embeds])


def extract_mean_std_features(texts, tokenizer, model, device="cuda", max_length=256):
    model.eval()

    number_of_layers = model.config.num_hidden_layers + 1

    features_by_layer = [[] for _ in range(number_of_layers)]

    for text in tqdm.tqdm(texts):
        encoded = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )

        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True, return_dict=True)

        token_mask = encoded["attention_mask"][0].bool()

        for layer, hidden_states in enumerate(outputs.hidden_states):
            token_hidden_states = hidden_states[0][token_mask]

            mean_features = token_hidden_states.mean(dim=0)

            std_features = token_hidden_states.std(dim=0, correction=0)

            sentence_features = torch.cat([mean_features, std_features], dim=0)

            features_by_layer[layer].append(sentence_features.cpu())

    features = torch.stack(
        [torch.stack(layer_features) for layer_features in features_by_layer]
    )

    return features


mean_std_tensor = {}
mean_tensor = {}
cls_tensor = {}


for split in train_val_test_dataset:
    ratings = torch.tensor(train_val_test_dataset[split][:]["label"])

    mean_std = extract_mean_std_features(
        train_val_test_dataset[split][:]["response"], tokenizer, model
    )
    mean = get_average_embeddings(train_val_test_dataset[split][:]["response"])
    cls = get_last_token_embeddings(train_val_test_dataset[split][:]["response"])

    mean_std_tensor[split] = {"embeddings": mean_std, "ratings": ratings}
    mean_tensor[split] = {"embeddings": mean, "ratings": ratings}
    cls_tensor[split] = {"embeddings": mean_std, "ratings": ratings}


torch.save(mean_std_tensor, "../../data/input/gpt2/distefano_gpt2_embd_mean_std.pt")
torch.save(mean_tensor, "../../data/input/gpt2/distefano_gpt2_embd_mean.pt")
torch.save(cls_tensor, "../../data/input/gpt2/distefano_gpt2_embd_cls_token.pt")
