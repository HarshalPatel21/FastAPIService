from torch.utils.data import Dataset

class customDataset(Dataset):

    def __init__(self , tokenizer,text,block_size) -> None:
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text =text
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx) :

        tokenized_inputs = self.tokenizer(
            self.text[idx],
            truncation = True,
            padding = "max_length",
            max_length=self.block_size,
            return_tensor="pt")
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"]
        return tokenized_inputs