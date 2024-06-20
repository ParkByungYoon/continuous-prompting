import torch
import pandas as pd
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


class RecsysDataset(Dataset):
    def __init__(self, data_path, edge_data_path):
        self.user_mapping = self.load_node_csv(edge_data_path, index_col='user_id')
        self.item_mapping = self.load_node_csv(edge_data_path, index_col='item_id')
        super().__init__(data_path)
        
    def set_input_text_list(self):
        input_text_list = []
        for data in self.data:
            user = data['user_id']
            prompt =f'사용자 {user}의 TV 프로그램 시청 기록:\n'
            for idx, item in enumerate(data['iteracted_items']):
                prompt += f'{idx}. {item}\n'
            prompt +='\n타겟 TV 프로그램:\n* ' + data['target_item'] + '\n\n'
            prompt += data['question']
            input_text_list.append(prompt)
        return input_text_list

    def set_continuous_prompt_input_list(self):
        continuous_prompt_input_list = []
        for x in self.data:
            interacted_items = list(map(lambda item:self.item_mapping[item], x['iteracted_items']))
            target_item = [self.item_mapping[x['target_item']]]
            item_ids = torch.Tensor(interacted_items+target_item).type(torch.long)
            user_id = torch.Tensor([self.user_mapping[x['user_id']]]).type(torch.long)
            continuous_prompt_input_list.append({'user_id':user_id, 'item_ids':item_ids})
        return continuous_prompt_input_list
    
    def load_node_csv(self, path, index_col):
        df = pd.read_csv(path, index_col=index_col)
        mapping = {index: i for i, index in enumerate(df.index.unique())}
        return mapping

    def set_answer_list(self):
        return [x['answer'] for x in self.data]