import torch
from util import load_jsonl
from torch_geometric.data import Data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = load_jsonl(data_path)
        self.input_text_list = self.set_input_text_list()
        self.answer_list = self.set_answer_list()
        self.continuous_prompt_input_list = self.set_continuous_prompt_input_list()
        

    def set_input_text_list(self):
        return [x['question'] for x in self.data]
    

    def set_answer_list(self):
        return [x['answer'][:-1] for x in self.data]
    

    def set_continuous_prompt_input_list(self):
        return [{'input_text_list':'\n'.join([x['node_information'],x['edge_information']])} for x in self.data]


    def __len__(self):
        return len(self.input_text_list)


    def __getitem__(self, idx):
        return self.input_text_list[idx], self.continuous_prompt_input_list[idx], self.answer_list[idx]
    

class GraphDataset(Dataset):
    def __init__(self, data_path):
        super().__init__(data_path)


    def set_continuous_prompt_input_list(self):
        graph_list = [self.pre_transform(x) for x in self.data]
        return [{'x': graph.x, 'edge_index': graph.edge_index} for graph in graph_list]
    
    
    def binary(self, x, bits):
        mask = 2**torch.arange(bits)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


    def pre_transform(self, data):
        edges_str = data['edge_information'].split(':')[-1].strip()[:-1].split(' ')
        h_list = []
        t_list = []
        for h,t in zip(edges_str[0::2], edges_str[1::2]):
            h_list.append(int(h[1]))
            t_list.append(int(t[0]))
        edge_index = torch.tensor([h_list, t_list], dtype=torch.int64)
        node_features = self.binary(torch.arange(int(data['nnodes'])), 5).type(torch.float32)
        return Data(x=node_features, edge_index=edge_index)