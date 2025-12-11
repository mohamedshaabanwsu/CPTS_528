import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import datetime
from dataset import TrainDataset
import json
import numpy as np
from sklearn.metrics import accuracy_score
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(0)


def binary_eval(predy, testy):
    acc = accuracy_score(testy, predy)
    return acc
    
def get_data(root_path, data_model, hd_model, type_="train"):
    
    hd = json.load(open(os.path.join(root_path, data_model, f"last_token_mean_{type_}.json"), encoding='utf-8'))
    hd3 = json.load(open(os.path.join(root_path, data_model, f"last_mean_{type_}.json"), encoding='utf-8'))

    hd_harmfull = json.load(open(os.path.join(root_path, data_model, f"last_token_mean_harmfull_{type_}.json"), encoding='utf-8'))
    hd3_harmfull = json.load(open(os.path.join(root_path, data_model, f"last_mean_harmfull_{type_}.json"), encoding='utf-8'))
    
    halu_hd = []
    right_hd = []
    halu_hd3 = []
    right_hd3 = []

    for hs, hs3 in zip(hd, hd3):
        right_hd.append(hs["right"])
        right_hd3.append(hs3["right"])
    
    for hs, hs3 in zip(hd_harmfull, hd3_harmfull):
        halu_hd.append(hs["harmfull"])
        halu_hd3.append(hs3["harmfull"])
        # halu_hd += hs["hallu"]
        # halu_hd3 += hs3["hallu"]

    # right_hd = right_hd[:len(halu_hd)]
    # right_hd3 = right_hd3[:len(halu_hd)]
    enddata = []

    # print(f"Length right_hd: {len(right_hd)}, halu_hd: {len(halu_hd)}")
    # print(f"Sample right_hd[0]: {right_hd[0]}, sample halu_hd[0]: {halu_hd[0]}")


    print(len(right_hd))
    print(len(right_hd3))
    print(len(halu_hd))
    print(len(halu_hd3))

    for i in range(len(hd)):
        # print(type(halu_hd[i]+halu_hd3[i]))
        # print(type(halu_hd[i]))
        # print(type(halu_hd3[i]))
        if not right_hd[i] or not right_hd3[i] or not halu_hd[i] or not halu_hd3[i]:
            print("skipped")
            continue  # skip if any is empty
        enddata.append({
            "hd": right_hd[i]+right_hd3[i],
            "label": 0
        })
        enddata.append({
            "hd": halu_hd[i]+halu_hd3[i],
            "label": 1
        })


        # enddata.append({
        #     "hd": hd["right"][i]+hd3["right"][i],
        #     "label": 0
        # })
        # enddata.append({
        #     "hd": hd_harmfull["hallu"][i]+hd3_harmfull["hallu"][i],
        #     "label": 1
        # })
        
    return enddata
    
from torch.nn.utils.rnn import pad_sequence


def custom_collate(batch):
    inputs = []
    labels = []
    for i, b in enumerate(batch):
        tensor_input = b['input']

        # print(f"Sample {i} shape: {tensor_input.shape}")
        
        if tensor_input.numel() == 0 or 0 in tensor_input.shape:
            raise ValueError(f"Empty or zero-dim tensor found in batch at index {i}, shape: {tensor_input.shape}")
        inputs.append(tensor_input)
        labels.append(b['y'])

    # print(len(inputs))
    # print(len(labels))

    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return {'input': inputs_padded, 'y': labels}



class Model():
    def __init__(self, args, path=None):
        self.args = args
        input_size = (4096*2 if "falcon" not in args.model_name else 4544*2) if "7b" in args.model_name else (5120*2 if "13b" in args.model_name else 8192*2)
        self.model = nn.Sequential()
        self.model.add_module("dropout", nn.Dropout(args.dropout))
        self.model.add_module(f"linear1", nn.Linear(input_size, 256))
        self.model.add_module(f"relu1", nn.ReLU())
        self.model.add_module(f"linear2", nn.Linear(256, 128))
        self.model.add_module(f"relu2", nn.ReLU())
        self.model.add_module(f"linear3", nn.Linear(128, 64))
        self.model.add_module(f"relu3", nn.ReLU())
        self.model.add_module(f"linear4", nn.Linear(64, 2))
        if path is not None:
            self.model.load_state_dict(torch.load(path, map_location = "cpu")["model_state_dict"])
        self.model.to(args.device)
        
    
    def save(self, acc, ei, prefix, name):
        torch.save({"model_state_dict": self.model.state_dict(),
                    "valid_acc": acc,
                    "epoch": ei},
                    prefix + f"{name}_model.pt")
        

    # def run(self, optim):
    #     now = datetime.datetime.now()
    #     prefix = f"{self.args.output_path}/{self.args.model_name}/train_log/"
    #     print(prefix)
    #     print(os.getcwd())
    #     if not os.path.exists(prefix):
    #         os.mkdir(prefix)
    #     epoch, epoch_start = self.args.train_epoch, 1
        
        
    #     train_data = get_data(self.args.data_path, self.args.model_name, self.args.model_name, "train")
    #     valid_data = get_data(self.args.data_path, self.args.model_name, self.args.model_name, "valid")
    #     test_data = get_data(self.args.data_path, self.args.model_name, self.args.model_name, "test")
        
    #     # In downstream tasks, we use the train+valid split as the training dataset and the test set as the validation set,
    #     # taking the best checkpoint from the validation set for the evaluation of downstream tasks.

    #     # rtrain_data = train_data+valid_data
    #     # train_dataset = TrainDataset(rtrain_data, self.args)
    #     # valid_dataset = TrainDataset(test_data, self.args, typ="valid")

    #     rtrain_data = train_data
    #     train_dataset = TrainDataset(rtrain_data, self.args)
    #     valid_dataset = TrainDataset(valid_data, self.args, typ="valid")
    #     test_dataset = TrainDataset(test_data, self.args, typ="test")
        
    #     # sample = train_dataset[0]
    #     # print(sample)

    #     # for i in range(len(train_dataset)):
    #     #     sample = train_dataset[i]
    #     #     if sample['input'].nelement() == 0:
    #     #         print(f"Empty tensor at index {i}")

    #     train_dataloader = DataLoader(dataset=train_dataset,
    #                                            batch_size=self.args.batch_size,
    #                                            shuffle=True,
    #                                            num_workers=4,
    #                                            collate_fn=custom_collate)
    #     valid_dataloader = DataLoader(dataset=valid_dataset,
    #                                            batch_size=self.args.batch_size // 2,
    #                                            shuffle=False,
    #                                            num_workers=4,
    #                                            collate_fn=custom_collate)
    #     nSamples = [len(train_dataset) - train_dataset.halu_num, train_dataset.halu_num]
    #     normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    #     normedWeights = torch.FloatTensor(normedWeights).to(self.args.device)
    #     loss_func = nn.CrossEntropyLoss(weight=normedWeights).to(self.args.device)
        
    #     best_acc = -1
    #     best_epoch = [0]
    #     for ei in range(epoch_start, epoch+1):
    #         cnt = 0
    #         self.model.train()
    #         train_loss = 0
    #         predy, trainy, hallu_sm_score = [], [], []
    #         for step, batch in enumerate(train_dataloader):
    #             input_ = batch["input"].to(self.args.device)
    #             # label_ids = torch.LongTensor([k[0] for k in batch["y"].tolist()]).to(self.args.device)
    #             label_ids = torch.LongTensor(batch["y"].tolist()).to(self.args.device)

    #             score = self.model(input_)
    #             hallu_sm = F.softmax(score, dim=1)[:, 1]
    #             _, pred = torch.max(score, dim=1)

    #             trainy.extend(label_ids.tolist())
    #             predy.extend(pred.tolist())
    #             hallu_sm_score.extend(hallu_sm.tolist())
    #             loss = loss_func(score, label_ids)
    #             train_loss += loss.item()
    #             optim.zero_grad()
    #             loss.backward()
    #             optim.step()
    #             cnt += 1
    #             if cnt % 10 == 0:
    #                 print("Training Epoch {} - {:.2f}% - Loss : {}".format(ei, 100.0 * cnt/len(train_dataloader), train_loss/cnt))
    #         print("Training Epoch {} ...".format(ei))
    #         acc = binary_eval(predy, trainy)
            
    #         print("Train Epoch {} end ! Loss : {}; Train Acc: {}".format(ei, train_loss, acc))

    #         self.model.eval()
    #         predy, validy, hallu_sm_score = [], [], []
    #         valid_loss = 0
    #         for step, batch in enumerate(valid_dataloader):
    #             input_ = batch["input"].to(self.args.device)
    #             # label_ids = torch.LongTensor([k[0] for k in batch["y"].tolist()]).to(self.args.device)
    #             label_ids = torch.LongTensor(batch["y"].tolist()).to(self.args.device)
    #             score = self.model(input_)
    #             hallu_sm = F.softmax(score, dim=1)[:, 1]
    #             _, pred = torch.max(score, dim=1)
    #             validy.extend(label_ids.tolist())
    #             predy.extend(pred.tolist())
    #             hallu_sm_score.extend(hallu_sm.tolist())
    #             loss = loss_func(score, label_ids)
    #             valid_loss += loss.item()
    #         print("Valid Epoch {} ...".format(ei))

    #         acc = binary_eval(predy, validy)

    #         if acc > best_acc:
    #             best_acc = acc
    #             best_epoch[0] = ei
    #             self.save(acc, ei, prefix, "best_acc")
                
        
    #     self.save(acc, ei, prefix, "last")
    #     print(f"Best acc : {best_acc} from epoch {best_epoch[0]}th;")

    #     batch_size = self.args.batch_size // 2
    #     test_dataloader = DataLoader(dataset=test_dataset,
    #                                  batch_size=batch_size,
    #                                  shuffle=False,
    #                                  num_workers=4,
    #                                  collate_fn=custom_collate)
    #     self.model.eval()
    #     predy, testy = [], []
    #     with torch.no_grad():
    #         for batch in test_dataloader:
    #             input_ = batch["input"].to(self.args.device)
    #             label_ids = torch.LongTensor(batch["y"].tolist()).to(self.args.device)
    #             score = self.model(input_)
    #             _, pred = torch.max(score, dim=1)
    #             predy.extend(pred.tolist())
    #             testy.extend(label_ids.tolist())
    #     accuracy = binary_eval(predy, testy)
    #     print(f"Test Accuracy: {accuracy}")
    #     return accuracy

    def run(self, optim):
        now = datetime.datetime.now()
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        prefix = f"{self.args.output_path}/{self.args.model_name}/train_log/"
        os.makedirs(prefix, exist_ok=True)

        # Full log file path under train_output folder
        train_output_dir = os.path.join(prefix, "train_output")
        os.makedirs(train_output_dir, exist_ok=True)
        log_path = os.path.join(train_output_dir, f"trainlog_{timestamp}.txt")

        epoch, epoch_start = self.args.train_epoch, 1
        
        
        train_data = get_data(self.args.data_path, self.args.model_name, self.args.model_name, "train")
        valid_data = get_data(self.args.data_path, self.args.model_name, self.args.model_name, "valid")
        test_data = get_data(self.args.data_path, self.args.model_name, self.args.model_name, "test")

        # In downstream tasks, we use the train+valid split as the training dataset and the test set as the validation set,
        # taking the best checkpoint from the validation set for the evaluation of downstream tasks.

        rtrain_data = train_data+valid_data
        train_dataset = TrainDataset(rtrain_data, self.args)
        valid_dataset = TrainDataset(test_data, self.args, typ="valid")
        train_dataloader = DataLoader(dataset=train_dataset,
                                               batch_size=self.args.batch_size,
                                               shuffle=True,
                                               num_workers=4)
        valid_dataloader = DataLoader(dataset=valid_dataset,
                                               batch_size=self.args.batch_size // 2,
                                               shuffle=False,
                                               num_workers=4)
        nSamples = [len(train_dataset) - train_dataset.halu_num, train_dataset.halu_num]
        normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        normedWeights = torch.FloatTensor(normedWeights).to(self.args.device)
        loss_func = nn.CrossEntropyLoss(weight=normedWeights).to(self.args.device)
        
        best_acc = -1
        best_epoch = [0]
        for ei in range(epoch_start, epoch+1):
            cnt = 0
            self.model.train()
            train_loss = 0
            predy, trainy, hallu_sm_score = [], [], []
            for step, batch in enumerate(train_dataloader):
                input_ = batch["input"].to(self.args.device)
                # label_ids = torch.LongTensor([k[0] for k in batch["y"].tolist()]).to(self.args.device)
                label_ids = torch.LongTensor(batch["y"].tolist()).to(self.args.device)
                # print(label_ids)
                score = self.model(input_)
                # print(score)
                hallu_sm = F.softmax(score, dim=1)[:, 1]
                _, pred = torch.max(score, dim=1)

                trainy.extend(label_ids.tolist())
                predy.extend(pred.tolist())
                hallu_sm_score.extend(hallu_sm.tolist())
                loss = loss_func(score, label_ids)
                train_loss += loss.item()
                optim.zero_grad()
                loss.backward()
                optim.step()
                cnt += 1
                if cnt % 10 == 0:
                    print("Training Epoch {} - {:.2f}% - Loss : {}".format(ei, 100.0 * cnt/len(train_dataloader), train_loss/cnt))
                    with open(log_path, "a") as f:
                        f.write("Training Epoch {} - {:.2f}% - Loss : {}".format(ei, 100.0 * cnt/len(train_dataloader), train_loss/cnt))

            print("Training Epoch {} ...".format(ei))
            with open(log_path, "a") as f:
                f.write("Training Epoch {} ...".format(ei))
                
            acc = binary_eval(predy, trainy)
            
            print("Train Epoch {} end ! Loss : {}; Train Acc: {}".format(ei, train_loss, acc))
            with open(log_path, "a") as f:
                f.write("Train Epoch {} end ! Loss : {}; Train Acc: {}\n".format(ei, train_loss, acc))

            self.model.eval()
            predy, validy, hallu_sm_score = [], [], []
            valid_loss = 0
            for step, batch in enumerate(valid_dataloader):
                input_ = batch["input"].to(self.args.device)
                # label_ids = torch.LongTensor([k[0] for k in batch["y"].tolist()]).to(self.args.device)
                label_ids = torch.LongTensor(batch["y"].tolist()).to(self.args.device)
                score = self.model(input_)
                # print(score)
                hallu_sm = F.softmax(score, dim=1)[:, 1]
                _, pred = torch.max(score, dim=1)
                validy.extend(label_ids.tolist())
                predy.extend(pred.tolist())
                hallu_sm_score.extend(hallu_sm.tolist())
                loss = loss_func(score, label_ids)
                valid_loss += loss.item()
            print("Valid Epoch {} ...".format(ei))
            with open(log_path, "a") as f:
                f.write(f"Valid Epoch {ei} ...\n")
                
            # print("predy", predy[:10])
            # print("validy", validy[:10])
            acc = binary_eval(predy, validy)

            if acc > best_acc:
                best_acc = acc
                best_epoch[0] = ei
                self.save(acc, ei, prefix, "best_harmful")
                
        #log train output
        #log_dir = os.path.join("train_output")
        #os.makedirs(log_dir, exist_ok=True)
        #log_path = os.path.join(log_dir, f"trainlog_{timestamp}.txt")

        with open(log_path, "a") as f:
            f.write(f"Best acc : {best_acc} from epoch {best_epoch[0]}th;\n")
        
        #save model                         
        self.save(acc, ei, prefix, "last")
        print(f"Best acc : {best_acc} from epoch {best_epoch[0]}th;")
        
        with open(log_path, "a") as f:
            f.write(args.model_name)

        print(f"Train output steps saved to train_log/train_output/trainlog_(time).txt")

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama7b")
    parser.add_argument("--output_path", default="auto-labeled/output", type=str)
    parser.add_argument("--data_path", default="auto-labeled/output", type=str)

    parser.add_argument("--train_epoch", default=3, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--wd", default=1e-5, type=float)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--device", default="cuda:0", type=str)
    
    args = parser.parse_args()

    
    
    model = Model(args)
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optim_func = torch.optim.Adam
    named_params = list(model.model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': args.wd, 'lr': args.lr},
        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.lr}
    ]
    optimizer = optim_func(optimizer_grouped_parameters)

    model.run(optimizer)
    print(args.model_name)
    