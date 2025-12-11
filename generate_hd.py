import os

# --------------------------------------------- #
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_type = "7b"
model_family = "llama"
result_path = f"./auto-labeled/output/{model_family}{model_type}"
# --------------------------------------------- #


print(f"{model_family}{model_type}")

import torch
from utils.model import get_model
from utils.gen import chat_change_with_answer

model, tokenizer, generation_config, at_id = get_model(model_type, model_family, 1)




from tqdm import tqdm
import json

import json
import torch
from tqdm import tqdm


    
def prompt_chat(title):
    return [{"role": "user", "content": f"Question: Tell me something about {title}.\nAnswer: "}]


def get_tokenized_ids(otext, title=None):
    text = otext.replace("@", "").replace("  ", " ").replace("  ", " ")
    text = tokenizer.decode(tokenizer(text.strip(), return_tensors='pt')['input_ids'].tolist()[0]).replace("<s>", "").replace("</s>", "")
    if model_family == "vicuna":
        text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: Question: Tell me something about {title}.\nAnswer: \nASSISTANT: {text}"
    if "chat" in model_family:
        return chat_change_with_answer(prompt_chat(title), text.strip(), tokenizer)
    return tokenizer(text.strip(), return_tensors='pt')['input_ids'].tolist()

def get_hd(text, title=None):
    ids = get_tokenized_ids(text, title)
    hd = model(torch.tensor(ids).to(model.device), output_hidden_states=True).hidden_states
    hds = hd[1][0][-1].clone().detach()
    for i in range(2, len(hd)):
        hds += hd[i][0][-1].clone().detach()
    hds = hds / (len(hd) - 1)
    
    # only for llamachat

    if model_family == "llamachat":
        start_at = -1
        for i in range(len(ids[0])):
            if ids[0][i:i+4] == [518, 29914, 25580, 29962]:
                start_at = i
                break
        if start_at == -1:
                print("not found")
                start_at = 1
        else:
            start_at += 4
    elif model_family == "vicuna":
        ids2 = tokenizer(f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: Question: Tell me something about {title}.\nAnswer: \nASSISTANT: ")['input_ids']
        start_at = -1
        ids1 = ids[0]
        for i in range(len(ids1)):
            if i >= len(ids2) or ids1[i] != ids2[i]:
                start_at = i
                break
        assert start_at != -1
    else:
        start_at = 2
    
    hds_mean_1 = torch.mean(hd[1][0][start_at-1:], dim=0)
    assert hds_mean_1.shape[0] == hd[1][0][-1].shape[-1]
    hds_mean_2 = torch.mean(hd[-1][0][start_at-1:], dim=0)

    return hds.tolist(), hds_mean_1.tolist(), hds_mean_2.tolist()


def get_hd_batch(texts, titles=None, batch_size=8):
    all_results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer_outputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(model.device)
        hd_batch = model(**tokenizer_outputs, output_hidden_states=True).hidden_states

        for b in range(len(batch_texts)):
            ids = tokenizer_outputs['input_ids'][b].tolist()
            title = titles[b] if titles is not None else None

            if model_family == "llamachat":
                start_at = -1
                for idx in range(len(ids)):
                    if ids[idx:idx+4] == [518, 29914, 25580, 29962]:
                        start_at = idx
                        break
                if start_at == -1:
                    print("not found")
                    start_at = 1
                else:
                    start_at += 4
            elif model_family == "vicuna":
                ids2 = tokenizer(f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: Question: Tell me something about {title}.\nAnswer: \nASSISTANT: ")['input_ids']
                start_at = -1
                for idx in range(len(ids)):
                    if idx >= len(ids2) or ids[idx] != ids2[idx]:
                        start_at = idx
                        break
                assert start_at != -1
            else:
                start_at = 2

            hds = hd_batch[1][b][start_at-1].clone().detach()
            for layer_idx in range(2, len(hd_batch)):
                hds += hd_batch[layer_idx][b][start_at-1].clone().detach()
            hds = hds / (len(hd_batch) - 1)

            hds_mean_1 = torch.mean(hd_batch[1][b][start_at-1:], dim=0)
            assert hds_mean_1.shape[0] == hd_batch[1][b][-1].shape[-1]
            hds_mean_2 = torch.mean(hd_batch[-1][b][start_at-1:], dim=0)

            all_results.append((hds.tolist(), hds_mean_1.tolist(), hds_mean_2.tolist()))
    return all_results



for data_type in ["train", "valid", "test"]:
    data = json.load(open(f"{result_path}/data_{data_type}.json", encoding='utf-8'))
    results_last = []
    results_mean1 = []
    results_mean2 = []

    results_last_harmfull = []
    results_mean1_harmfull = []
    results_mean2_harmfull = []

    # Prepare batch inputs
    batch_size = 24
    texts, titles, labels = [], [], []

    for k in data:
        # print("Processing:", k)

        # k["original_text"] = " ".join(k["sentences"][:2])
        texts.append(k["original_text"])
        titles.append(k["title"])
        labels.append(k["prompt_label"])

    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        batch_titles = titles[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        batch_results = get_hd_batch(batch_texts, batch_titles, batch_size=batch_size)

        for (hdl, hdm1, hdm2), label in zip(batch_results, batch_labels):
            if label == 1:
                results_last_harmfull.append({"harmfull": hdl})
                results_mean1_harmfull.append({"harmfull": hdm1})
                results_mean2_harmfull.append({"harmfull": hdm2})
            else:
                results_last.append({"right": hdl})
                results_mean1.append({"right": hdm1})
                results_mean2.append({"right": hdm2})

    with open(f"{result_path}/last_token_mean_{data_type}.json", "w+") as f:
        json.dump(results_last, f)
    with open(f"{result_path}/last_mean_{data_type}.json", "w+") as f:
        json.dump(results_mean2, f)

    with open(f"{result_path}/last_token_mean_harmfull_{data_type}.json", "w+") as f:
        json.dump(results_last_harmfull, f)
    with open(f"{result_path}/last_mean_harmfull_{data_type}.json", "w+") as f:
        json.dump(results_mean2_harmfull, f)




# for data_type in ["train", "valid", "test"]:
#     data = json.load(open(f"{result_path}/data_{data_type}.json", encoding='utf-8'))
    
#     #truncate data for testing purposes
#     #data = data[:split_limit[data_type]]
    
#     results_last = []
#     results_mean1 = []
#     results_mean2 = []

#     results_last_harmfull = []
#     results_mean1_harmfull = []
#     results_mean2_harmfull = []

#     for k in tqdm(data):
#         hd_last = []
#         hd_mean1 = []
#         hd_mean2 = []
        
#         hdl_origin, hdm1_origin, hdm2_origin = get_hd(k["original_text"], k["title"])

#         if k["prompt_label"] == 1:
#             results_last_harmfull.append({"harmfull": hdl_origin})
#             results_mean1_harmfull.append({"harmfull": hdm1_origin})
#             results_mean2_harmfull.append({"harmfull": hdm2_origin})
#         else:
#             results_last.append({"right": hdl_origin})
#             results_mean1.append({"right": hdm1_origin})
#             results_mean2.append({"right": hdm2_origin})

#     with open(f"{result_path}/last_token_mean_{data_type}.json", "w+") as f:
#         json.dump(results_last, f)
#     with open(f"{result_path}/last_mean_{data_type}.json", "w+") as f:
#         json.dump(results_mean2, f)

#     with open(f"{result_path}/last_token_mean_harmfull_{data_type}.json", "w+") as f:
#         json.dump(results_last_harmfull, f)
#     with open(f"{result_path}/last_mean_harmfull_{data_type}.json", "w+") as f:
#         json.dump(results_mean2_harmfull, f)
