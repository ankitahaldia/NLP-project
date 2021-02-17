

from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


# input must be str like a text"
def predict(X_val):

    industy_dict = {'Agriculture': 5,
     'Consumer Products': 2,
     'Energy': 8,
     'Finance': 4,
     'Health Care': 11,
     'Manufacturing': 7,
     'Media': 6,
     'Pharmaceuticals': 9,
     'Public and Social sector': 10,
     'Telecom': 0,
     'Transport & Logistics': 1,
     'automative': 3}

    label_dict_inverse = {v: k for k,v in industy_dict.items()}

    X_val = list(X_val)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
        num_labels=12,
        output_attentions=False,
        output_hidden_states=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.load_state_dict(torch.load('Models/BERT_art_epoch15.model', map_location=torch.device(device)))


    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        do_lower_case=True
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        X_val,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    
    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']

    dataset_val = TensorDataset(input_ids_val, 
                                  attention_masks_val)

    dataloader_val = DataLoader(
        dataset_val,
        sampler=RandomSampler(dataset_val),
        batch_size=32
    )

    def evaluate(dataloader_val):

        model.eval()
        predictions = []
        
        for batch in dataloader_val:
            
            batch = tuple(b.to(device) for b in batch)
            
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1]
                    }

            with torch.no_grad():        
                outputs = model(**inputs)

            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            predictions.append(logits)
        
        predictions = np.concatenate(predictions, axis=0)
                
        return predictions


    preds = evaluate(dataloader_val)
    preds_flat = np.argmax(preds, axis=1).flatten()
    insdustry_type = label_dict_inverse[preds_flat[0]]


    # return "str"
    return insdustry_type


