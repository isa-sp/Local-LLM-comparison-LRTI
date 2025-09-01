from transformers import EncoderDecoderModel, AutoTokenizer, PreTrainedModel
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
from datetime import datetime
import os
import time
from typing import Union
import numpy as np

from prompt_classification_utils import symptom_dict, construct_prompt

PARENT_DIR = 'Z:/E_ResearchData/2_ResearchData/NLP project Tuur/Thesis Matthew/'

class SymptomPromptModel:
    def __init__(self, model: PreTrainedModel, tokenizer: AutoTokenizer, symptom: str):
        self.model = model
        self.tokenizer = tokenizer
        self.symptom = symptom


    def classify(self, note: str, examples: Union[list[tuple], None]):
        prompt = construct_prompt(note, self.symptom, examples)
        prob_pos = self._predict_token_probability(prompt, symptom_dict[self.symptom]["pos"],)
        prob_neg = self._predict_token_probability(prompt, symptom_dict[self.symptom]["neg"])
        prob_abs = self._predict_token_probability(prompt, symptom_dict[self.symptom]["abs"])
        #print(f"Prompt: {prompt}")
        #print(f"Probabilities: pos={prob_pos}, neg={prob_neg}, abs={prob_abs}")
        print("Prompt length: ", len(prompt))
        
        # return the label with the highest probability
        if prob_neg < prob_pos and prob_neg < prob_abs:
            return 0    # negatief
        elif prob_pos <= prob_neg and prob_pos <= prob_abs:
            return 1    # positief
        else:
            if prob_abs == prob_neg and prob_abs == prob_pos:
                print("[Warning] All probabilities are equal. Returning 'afwezig' as default label.")
            return 2    # afwezig
        
    
    
    def _predict_token_probability(self, prompt: str, token: str):
        inputs = self.tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        
        tokenized_token = self.tokenizer.batch_encode_plus([token], return_tensors='pt', padding='max_length', truncation=True, max_length=len(token))

        # Pass tokenized prompt to the model
        with torch.no_grad():  # No need to calculate gradients
            outputs = self.model(input_ids=inputs['input_ids'], 
                                 labels = tokenized_token["input_ids"],
                                 attention_mask=inputs['attention_mask'], 
                                 output_hidden_states=False, 
                                 return_dict=True)

        del inputs
        del tokenized_token
        return outputs.loss


    def evaluate(self, dataset: pd.DataFrame, examples: pd.DataFrame, example_amt: int, txt_column: str):
        predicted_labels = []
        correct_labels = []

        for i, ehr in tqdm(dataset.iterrows(), total=len(dataset), desc="Predicting..."):
            #note = ehr.DEDUCE_omschrijving
            note = ehr[txt_column]
            label = ehr[self.symptom]
         
            
            examples = examples.iloc[i:example_amt*(i+1)] 
            prediction = self.classify(note, examples)
            correct_labels.append(label)
            predicted_labels.append(prediction)
        return predicted_labels, correct_labels
    
    
    def _generate_response(prompt, model: PreTrainedModel, tokenizer: AutoTokenizer):
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=512)

        # Pass the tokenized prompt to the model
        outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], do_sample=True, top_p=0.95, no_repeat_ngram_size=2)

        # Decode the output ids to get the generated response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response    

def generate_run_dir(model_name: str, label, DS_SIZE, k=5) -> str:
    current = datetime.now().strftime('%Y%m%d_%H%M')
    dir_name = f'runs/pbrun_{label}_{model_name}_{DS_SIZE}-samples_{current}/'
    
    path = os.path.join(PARENT_DIR, dir_name)
    os.mkdir(path)
    
    # Create subdirs for folds
    for i in range(k):
        fold_path = os.path.join(PARENT_DIR, dir_name, f'fold_{i}')
        os.mkdir(fold_path)
    
    
    print(f'Created folder \'{path}\' for current run.')
    return path


def main(symptom, DS_SIZE, model_name, txt_column):
    #symptom = "Koorts"
    k=5
    #model_name = "medroberta-prompt"#"robbert-prompt"
    tokenizer_name = 'models--pdelobelle--robbert-v2-dutch-base'
    #DS_SIZE = 1 # Samples
    run_dir = generate_run_dir(model_name, symptom, DS_SIZE)

    for current_fold in range(k):
        print(f'Fold {current_fold}')
        # Load the trained model and tokenizer
        model = EncoderDecoderModel.from_pretrained(f"{PARENT_DIR}/models/{model_name}/")
        print(model_name, model.num_parameters())
        tokenizer = AutoTokenizer.from_pretrained(f"{PARENT_DIR}/models/{tokenizer_name}/snapshots/snap")

        model.config.vocab_size = model.config.encoder.vocab_size
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if model.config.decoder_start_token_id is None:
            model.config.decoder_start_token_id = tokenizer.cls_token_id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.eos_token_id

        model.generation_config.decoder_start_token_id = tokenizer.cls_token_id

        # Load the dataset
        test_data_fp = f"data/test/fold_{current_fold}.csv"
        example_data_fp = f"data/train/fold_{current_fold}.csv" 
        test_data = pd.read_csv(test_data_fp, sep='|')
        example_data = pd.read_csv(example_data_fp, sep='|').sample(frac=1, random_state=420, replace=True)

        # Initialize the model
        model = SymptomPromptModel(model, tokenizer, symptom)
        predicted_labels, correct_labels = model.evaluate(test_data, example_data, DS_SIZE, txt_column)
        precision, recall, f1, _ = precision_recall_fscore_support(correct_labels, predicted_labels, average='weighted')
        accuracy = accuracy_score(correct_labels, predicted_labels)
        conf_matrix = confusion_matrix(correct_labels, predicted_labels)

        results_json = {
            "Eval Logs": [
                {
                    "eval_accuracy": accuracy,
                    "eval_f1": f1,
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_confusion_matrix": conf_matrix.tolist()
                }
            ]
        }

        save_path = os.path.join(run_dir, f'fold_{current_fold}/log_history.json')
        with open(save_path, "w") as f:
            json.dump(results_json, f)
            
        del model
        del tokenizer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    
    
    #main("Kortademigheid", 3)
    #main("Hoesten", 3)
    #main("Koorts", 3)
    
    model_names = ["medroberta-prompt", "robbert-prompt"]
    
    #SYMPTOMS =  ['Koorts','Hoesten','Dyspnoe','Verwardheid','Pijn_Borst','Rillingen','Zieke_Indruk', 'Crepitaties', 'Sputum']
    SYMPTOMS =  ['Dyspnoe','Koorts','Hoesten', 'Verwardheid','Pijn_Borst','Rillingen','Zieke_Indruk', 'Crepitaties', 'Sputum']
    SYMPTOM_TO_TXT_MAP = {s:'DEDUCE_omschrijving_S' for s in SYMPTOMS} # all S regel
    SYMPTOM_TO_TXT_MAP['Verwardheid']='DEDUCE_omschrijving' # Verwardheid S+O
    SYMPTOM_TO_TXT_MAP['Crepitaties']='DEDUCE_omschrijving_O' # Crepitaties O
    SYMPTOM_TO_TXT_MAP['Zieke_Indruk']='DEDUCE_omschrijving_O' # Zieke Indruk O
    
    #DS_SIZES = [25, 50] #[ 100, 200, 400, 800]
    DS_SIZES = [1,2,3]
    
    
    todo_initial = [(symptom,  ds, model_name) for symptom in SYMPTOMS for ds in DS_SIZES for model_name in model_names]
    
    #finished_exps = [('Koorts',1,'medroberta-prompt'),('Koorts',2,'medroberta-prompt'),('Koorts',1,'robbert-prompt'),('Koorts',2,'robbert-prompt') ]
    
    finished_exps = [(s,n,m) for m in ['medroberta-prompt','robbert-prompt'] for s in ['Dyspnoe','Koorts','Hoesten', 'Verwardheid','Pijn_Borst'] for n in [1,2,3]]
    
    todo = [exp for exp in todo_initial if not exp in finished_exps]
    
    print(todo)
    
    #exit()
   
    
    for (symptom, size, model_name) in todo:
        print('_'*100)
        print('STARTING: ******* ' + symptom + ' ' +str(size)  + ' ' + model_name + ' ********' )
        print('_'*100)
        t0 = time.time()
        main(symptom, size, model_name, SYMPTOM_TO_TXT_MAP[symptom])
        took = time.time()-t0
        print('TOOK' + ' ' + str(took/60) +' minutes')

