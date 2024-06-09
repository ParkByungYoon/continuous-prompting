import torch
from transformers import AutoTokenizer, AutoModel

class TextContinuousPromptModel(torch.nn.Module):
    def __init__(self, huggingface_model_name_or_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name_or_path)
        self.model = AutoModel.from_pretrained(huggingface_model_name_or_path)
        
    def forward(self, input_text_list):
        tokenized_input = self.tokenizer(input_text_list, padding=True, truncation=True, return_tensors='pt')
        tokenized_input = tokenized_input.to(self.model.device)

        # return self.model(**tokenized_input).last_hidden_state
        outputs = self.model(**tokenized_input, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        return last_hidden_states[0,0,:].unsqueeze(dim=0)