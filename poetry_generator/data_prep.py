import os
import string
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import kagglehub

# Download the william shakespears sonnets text dataset
path = kagglehub.dataset_download("shivamshinde123/william-shakespeares-sonnet")

text_file_path = os.path.join(path, 'Sonnet.txt')

with open(text_file_path, 'r', encoding='utf-8') as f:
    sonnet_dataset = f.read()

word_to_index = {"<UNK>": 0}  # Assign index 0 to <UNK>
index_to_word = {0: "<UNK>"}

# Create the dictionary

for line in sonnet_dataset.split('\n'):
    line = line.strip().lower()  # Convert to lowercase and remove extra spaces
    line = line.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    for word in line.split():
        if word not in word_to_index:
            index = len(word_to_index)
            word_to_index[word] = index
            index_to_word[index] = word

# Print the dictionary for the poetry data
print(word_to_index)

# Map the poetry dataset with the created dictionary indexes
indexed_dataset = [
    [
        word_to_index.get(word.lower().translate(str.maketrans('', '', string.punctuation)),
                       word_to_index["<UNK>"]) 
        for word in line.split()
    ] 
    for line in sonnet_dataset.split('\n') if line.strip() #skip empty lines
]

# Split the data for training and testing
n = int(len(indexed_dataset)*0.9)
train_data = indexed_dataset[:n]  # 90% will be used for training
valid_data = indexed_dataset[n:]  # Remaining 10% will be used for validation


class SonnetDataset(Dataset):
    def __init__(self, indexed_dataset):
        self.data = []

        for seq in indexed_dataset:
            if len(seq) > 1:
                input_seq = seq[:-1]
                output_seq = seq[1:]
                self.data.append((torch.tensor(input_seq), torch.tensor(output_seq)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# Create the collate function for padding the sequences
def collate_fn(batch):
    inputs, outputs = zip(*batch)
    padded_input = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_output = pad_sequence(outputs, batch_first=True, padding_value=0)
    return padded_input, padded_output

# Create the dataset and data loader
train_dataset = SonnetDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

valid_dataset = SonnetDataset(valid_data)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

if __name__ == "__main__":
    for input, output in train_loader:
        print('padded input sequence :', input)
        print('padded output sequence :', output)
        break