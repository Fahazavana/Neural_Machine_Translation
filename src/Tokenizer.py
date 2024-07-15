import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class Corpus:
    def __init__(self, file_name, lang):
        self.file_name = file_name
        self.lang = lang
        self.vocab_size = 11
        self.data_nbr = []
        self.data_str = []
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3, "<num>": 4, "<com>": 5, "<prc>": 6, "<opn>": 7,
                     "<cld>": 8, "<apo>": 9, "<ltx>": 10, }
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>", 4: "<num>", 5: "<com>", 6: "<prc>", 7: "<opn>",
                     8: "<cld>", 9: "<apo>", 10: "<ltx>", }
        self.__init_data()
        self.__encode()

    def __init_data(self):
        with open(self.file_name, "r") as file:
            for line in file:
                line = line.strip().split()
                self.data_str.append(line)
                for word in line:
                    if not self.stoi.get(word):
                        self.vocab_size += 1
                        self.stoi[word] = self.vocab_size - 1
                        self.itos[self.vocab_size - 1] = word

    def __encode(self):
        self.data_nbr = [[self.stoi[word] for word in sentence] for sentence in self.data_str]

    def decode(self, data):
        return [[self.itos[word] for word in sentence] for sentence in data]

    def encode_new(self, data):
        return [[self.itos[word] if word in self.stoi else 3 for word in sentence] for sentence in data]


class LangData(Dataset):
    def __init__(self, source, target):
        if len(source.data_nbr) != len(target.data_nbr):
            raise RuntimeError("Source and target must have the same lenght")
        self.target = target.data_nbr
        self.source = source.data_nbr

    def __getitem__(self, idx):
        x = torch.tensor(self.source[idx], dtype=torch.long)
        y = torch.tensor(self.target[idx], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.source)


def collate_fn(batch):
    """
     Pad shorter sequence with 0 (<pad>) to match the longest sequence
     to get a uniform batch size.
    """
    source, target = zip(*batch)
    # Pad sequences
    source = pad_sequence(source, batch_first=False, padding_value=0)
    target = pad_sequence(target, batch_first=False, padding_value=0)
    return source, target


def dataLoader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
