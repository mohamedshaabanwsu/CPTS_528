from torch.utils.data import Dataset
import torch
torch.manual_seed(0)

class TrainDataset(Dataset):
    def __init__(self, train_data, args, typ="train"):
        self.all_data = []

        for _, data in enumerate(train_data):
            if data["hd"] and len(data["hd"]) > 0:  # Filter empty hd vectors
                self.all_data.append({
                    "label": data["label"],
                    "hd": data["hd"]
                })
        self.halu_num = len([d for d in self.all_data if d["label"]])
        print(f"{typ} data: [0, 1] - [{len(self.all_data) - self.halu_num}, {self.halu_num}]")
            
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        item = self.all_data[idx]
        input_tensor = torch.tensor(item['hd'])
        if input_tensor.numel() == 0 or 0 in input_tensor.shape:
            raise ValueError(f"Empty tensor found in Dataset at index {idx} with shape {input_tensor.shape}")
        label = torch.tensor(item['label'])
        return {'input': input_tensor, 'y': label}

    
    # def __getitem__(self, idx):
    #     data = self.all_data[idx]  
    #     return {
    #         "input": torch.tensor(data["hd"]),
    #         "y": torch.LongTensor([data["label"]]),
    #     }

