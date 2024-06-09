import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ContinuousPromptingLLM(torch.nn.Module):
    def __init__(self, huggingface_model_name_or_path, continuous_prompt_model, continuous_embedding_dim, projection_module=None):
        super().__init__()
        self.huggingface_model_name_or_path = huggingface_model_name_or_path

        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.huggingface_model_name_or_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.huggingface_model_name_or_path)

        self.continuous_prompt_model = continuous_prompt_model
        self.continuous_embedding_dim = continuous_embedding_dim

        with torch.no_grad():
            self.padding_embedding = self.llm_model.get_input_embeddings()(torch.tensor(self.llm_tokenizer.pad_token_id))

        if projection_module == None:
            self.projection_module = torch.nn.Linear(self.continuous_embedding_dim, 
                                                     self.llm_model.config.hidden_size)
        else:
            self.projection_module = projection_module

    def projection(self, continuous_prompt_vector):
        return [self.projection_module(x) for x in continuous_prompt_vector]
    
    def make_input_embed(self, text_input, continuous_prompt_input_dict, padding_side='left', embedding_first=False):
        device = self.llm_model.device
        tokenized_input = self.llm_tokenizer.batch_encode_plus(text_input, return_tensors='np')
        word_embedding_list = [self.llm_model.get_input_embeddings()(torch.tensor(x).to(device)) for x in tokenized_input['input_ids']]

        continuous_prompt_vector = self.continuous_prompt_model(**continuous_prompt_input_dict)
        if continuous_prompt_vector.dim() != 3:
            continuous_prompt_vector = continuous_prompt_vector.unsqueeze(dim=0)
        projected_continuous_prompt_vector = self.projection(continuous_prompt_vector)

        if embedding_first==True:
            llm_input_embedding_list = [torch.cat([projected_continuous_prompt_vector[i], word_embedding_list[i]], dim=0) for i in range(len(word_embedding_list))]
        else:
            llm_input_embedding_list = [torch.cat([word_embedding_list[i], projected_continuous_prompt_vector[i]], dim=0) for i in range(len(word_embedding_list))]
        batch_max_length = max([len(x) for x in llm_input_embedding_list])

        attention_mask = []
        if padding_side=='left':
            for i in range(len(llm_input_embedding_list)):
                cur_length = len(llm_input_embedding_list[i])
                
                #left padding
                cur_attention_mask = [0]*(batch_max_length-cur_length)+[1]*cur_length    
                cur_pad = self.padding_embedding.expand(batch_max_length-cur_length,len(self.padding_embedding)).to(device)
                llm_input_embedding_list[i] = torch.cat([cur_pad, llm_input_embedding_list[i]], dim=0)
                attention_mask.append(cur_attention_mask)


        elif padding_side=='right':
            for i in range(len(llm_input_embedding_list)):
                cur_length = len(llm_input_embedding_list[i])
                
                #right padding
                cur_attention_mask = [1]*cur_length+[0]*(batch_max_length-cur_length)  
                cur_pad = self.padding_embedding.expand(batch_max_length-cur_length,len(self.padding_embedding)).to(device)
                llm_input_embedding_list[i] = torch.cat([llm_input_embedding_list[i], cur_pad], dim=0)
                attention_mask.append(cur_attention_mask)

        llm_input_embedding_tensor = torch.stack(llm_input_embedding_list, dim=0).to(device)
        attention_mask = torch.tensor(attention_mask).to(device)

        return llm_input_embedding_tensor, attention_mask

    def make_seq2seq_input_label(self, text_input, continuous_prompt_input_dict, target_text_list, padding_side='left', embedding_first=False):
        device = self.llm_model.device

        tokenized_input = self.llm_tokenizer.batch_encode_plus(text_input, return_tensors='np')
        input_word_embedding_list = [self.llm_model.get_input_embeddings()(torch.tensor(x).to(device)) for x in tokenized_input['input_ids']]

        continuous_prompt_vector = self.continuous_prompt_model(**continuous_prompt_input_dict)
        if continuous_prompt_vector.dim() != 3:
            continuous_prompt_vector = continuous_prompt_vector.unsqueeze(dim=0)
        
        

        projected_continuous_prompt_vector = self.projection(continuous_prompt_vector)
        tokenized_target = self.llm_tokenizer.batch_encode_plus(target_text_list, return_tensors='np')
        
        for i in range(len(tokenized_target.input_ids)):
            tokenized_target.input_ids[i] = tokenized_target.input_ids[i].tolist()[1:]+[self.llm_tokenizer.eos_token_id] #remove the bos and add the eos
        
        target_word_embedding_list = [self.llm_model.get_input_embeddings()(torch.tensor(x).to(device)) for x in tokenized_target['input_ids']]

        if embedding_first ==  True:
            llm_input_embedding_list = [torch.cat([projected_continuous_prompt_vector[i], input_word_embedding_list[i], target_word_embedding_list[i]], dim=0) for i in range(len(input_word_embedding_list))]
        else:
            llm_input_embedding_list = [torch.cat([input_word_embedding_list[i], projected_continuous_prompt_vector[i], target_word_embedding_list[i]], dim=0) for i in range(len(input_word_embedding_list))]
        
        batch_max_length = max([len(x) for x in llm_input_embedding_list])

        attention_mask = []
        labels = []
        if padding_side=='left':
            for i in range(len(llm_input_embedding_list)):
                cur_length = len(llm_input_embedding_list[i])
                #left padding
                cur_attention_mask = [0]*(batch_max_length-cur_length)+[1]*cur_length    
                cur_pad = self.padding_embedding.expand(batch_max_length-cur_length,len(self.padding_embedding)).to(device)
                llm_input_embedding_list[i] = torch.cat([cur_pad, llm_input_embedding_list[i]], dim=0)
                attention_mask.append(cur_attention_mask)
            
                cur_label = tokenized_target['input_ids'][i].tolist()
                cur_label = [-100]*(batch_max_length-len(cur_label)) + cur_label
                labels.append(cur_label)
        
        
        #return torch.stack(llm_input_embedding_list).to(self.llm_model.device), torch.stack(attention_mask).to(self.llm_model.device), torch.tensor(labels, dtype=torch.int64).to(self.llm_model.device)
        return torch.stack(llm_input_embedding_list).to(self.llm_model.device), torch.tensor(attention_mask).to(self.llm_model.device), torch.tensor(labels, dtype=torch.int64).to(self.llm_model.device)