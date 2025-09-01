from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
import numpy as np
import os
import time
from datetime import datetime
torch.set_num_threads(4)

from utils import compute_metrics, SymptomDataset
import json

PARENT_DIR = 'Z:/E_ResearchData/2_ResearchData/NLP project Tuur/Thesis Matthew/'
#LABEL_TO_CLASSIFY = 'Koorts'
#DS_SIZE = 100


class SymptomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
        
def generate_run_dir(model_name: str, k:int, label, DS_SIZE) -> str:
    current = datetime.now().strftime('%Y%m%d_%H%M')
    dir_name = f'runs/run_{label}_{model_name}_{DS_SIZE}-samples_{current}/'
    
    path = os.path.join(PARENT_DIR, dir_name)
    os.mkdir(path)
    
    # Create subdirs for folds
    '''for i in range(k):
        fold_path = os.path.join(PARENT_DIR, dir_name, f'/fold_{i}')
        print(fold_path)
        #os.mkdir(fold_path)
        '''
        
    print(f'Created folder \'{path}\' for current run.')
    return path
    


def main_dev_one_model(LABEL_TO_CLASSIFY, model_name, training_data_csv_path, TEXT_COLUMN):
    train_df = pd.read_csv(training_data_csv_path, sep='|')
    
    DS_SIZE=len(train_df)
    
    current = datetime.now().strftime('%Y%m%d_%H%M')
    dir_name = f'runs/run_{LABEL_TO_CLASSIFY}_{model_name}_{DS_SIZE}_{current}/'
    run_dir = os.path.join(PARENT_DIR, dir_name)
    os.mkdir(run_dir)
    
    # Get X and y
    train_texts = [str(text) for text in train_df[TEXT_COLUMN].tolist()]
    train_labels = train_df[LABEL_TO_CLASSIFY].tolist()

    # Load the pre-trained model and tokenizer
    
    model = RobertaForSequenceClassification.from_pretrained(f"{PARENT_DIR}/huggingface/hub/{model_name}/snapshots/snap", num_labels=3)
    tokenizer = RobertaTokenizerFast.from_pretrained(f"{PARENT_DIR}/huggingface/hub/{model_name}/snapshots/snap")

    # Tokenize the dataset
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    train_dataset = SymptomDataset(train_encodings, train_labels)

    # Fine-tune the model on the dataset
    training_args = TrainingArguments(
        output_dir=f'{run_dir}/results',
        eval_strategy='no',
        save_strategy='no',
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01, 
        logging_steps=10,
        logging_strategy='epoch',
        logging_dir=f'{run_dir}/logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        
    )

    trainer.train()

    #print(trainer.state.log_history)

    # Evaluate the model
    #eval_results = trainer.evaluate()


    save_path = f'{run_dir}/finetuned_model'

    # Save the fine-tuned model
    model.save_pretrained(save_path)

    ### log saving
    # Initialize lists for each type of log
    train_logs = []
    eval_logs = []
    total_logs = []

    # Iterate over the log history
    for log in trainer.state.log_history:
        # Check the keys in the log to determine its type
        if 'loss' in log and 'learning_rate' in log:
            train_logs.append(log)
        elif 'eval_loss' in log:
            eval_logs.append(log)
        elif 'train_runtime' in log:
            total_logs.append(log)

    # Save the logs to the file
    with open(f'{run_dir}/log_history.json', 'w') as f:
        json.dump({'Train Logs': train_logs, 'Eval Logs': eval_logs, 'Total Logs': total_logs}, f)
        
    del model
    del tokenizer
    del train_encodings
    del trainer
    torch.cuda.empty_cache()
    
def main(LABEL_TO_CLASSIFY, model_name, DS_SIZE, TEXT_COLUMN):
    #model_name = 'models--CLTL--MedRoBERTa.nl' #'models--pdelobelle--robbert-v2-dutch-base' #
    k = 5
    
    run_dir = generate_run_dir(model_name, k, label=LABEL_TO_CLASSIFY, DS_SIZE=DS_SIZE)
    
    
    for current_fold in range(k):
        print(f"Starting training for fold {current_fold}")
        
        # Load data
        train_df = pd.read_csv(f'data/train/fold_{current_fold}.csv', sep='|')
        test_df = pd.read_csv(f'data/test/fold_{current_fold}.csv', sep='|')
        
        # Get X and y
        train_texts = [str(text) for text in train_df[TEXT_COLUMN].tolist()[:DS_SIZE]]
        train_labels = train_df[LABEL_TO_CLASSIFY].tolist()[:DS_SIZE]
        #print(len(train_labels))
        test_texts = [str(text) for text in test_df[TEXT_COLUMN].tolist()]
        test_labels = test_df[LABEL_TO_CLASSIFY].tolist()


        # Load the pre-trained model and tokenizer
        
        model = RobertaForSequenceClassification.from_pretrained(f"{PARENT_DIR}/huggingface/hub/{model_name}/snapshots/snap", num_labels=3)
        tokenizer = RobertaTokenizerFast.from_pretrained(f"{PARENT_DIR}/huggingface/hub/{model_name}/snapshots/snap")

        
        # Tokenize the dataset
        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        test_encodings = tokenizer(test_texts, truncation=True, padding=True)

        train_dataset = SymptomDataset(train_encodings, train_labels)
        test_dataset = SymptomDataset(test_encodings, test_labels)



        # Fine-tune the model on the dataset
        training_args = TrainingArguments(
            output_dir=f'{run_dir}/fold_{current_fold}/results',
            eval_strategy='epoch',
            save_strategy='no',
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01, 
            logging_steps=10,
            logging_strategy='epoch',
            logging_dir=f'{run_dir}/fold_{current_fold}/logs',
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        #print(trainer.state.log_history)

        # Evaluate the model
        eval_results = trainer.evaluate()


        save_path = f'{run_dir}/fold_{current_fold}/finetuned_model'

        # Save the fine-tuned model
        #model.save_pretrained(save_path)

        ### log saving
        # Initialize lists for each type of log
        train_logs = []
        eval_logs = []
        total_logs = []

        # Iterate over the log history
        for log in trainer.state.log_history:
            # Check the keys in the log to determine its type
            if 'loss' in log and 'learning_rate' in log:
                train_logs.append(log)
            elif 'eval_loss' in log:
                eval_logs.append(log)
            elif 'train_runtime' in log:
                total_logs.append(log)

        # Save the logs to the file
        with open(f'{run_dir}/fold_{current_fold}/log_history.json', 'w') as f:
            json.dump({'Train Logs': train_logs, 'Eval Logs': eval_logs, 'Total Logs': total_logs}, f)
            
        del model
        del tokenizer
        del train_encodings
        del test_encodings
        del trainer
        torch.cuda.empty_cache()
        
        
if __name__ == "__main__":
  
  
  
            
   SYMPTOMS =  ['Dyspnoe','Koorts','Hoesten', 'Verwardheid','Pijn_Borst','Rillingen','Zieke_Indruk', 'Crepitaties', 'Sputum']
   SYMPTOM_TO_TXT_MAP = {s:'DEDUCE_omschrijving_S' for s in SYMPTOMS} # all S regel
   SYMPTOM_TO_TXT_MAP['Verwardheid']='DEDUCE_omschrijving' # Verwardheid S+O
   SYMPTOM_TO_TXT_MAP['Crepitaties']='DEDUCE_omschrijving_O' # Crepitaties O
   SYMPTOM_TO_TXT_MAP['Zieke_Indruk']='DEDUCE_omschrijving_O' # Zieke Indruk O

   MODELS = ['models--pdelobelle--robbert-v2-dutch-base', 'models--CLTL--MedRoBERTa.nl']
   if False: # ---------------------------- CROSS VALIDATION EXPERIMENTS
    
        #'Dyspnoe', 'Koorts',

        #DS_SIZES = [25, 50] #[ 100, 200, 400, 800]
        DS_SIZES = [1600, 800, 400, 200]
        
        todo = [(symptom, model, ds) for symptom in SYMPTOMS for model in MODELS for ds in DS_SIZES]
        
        #finished_exps = [(s,'models--pdelobelle--robbert-v2-dutch-base',800) for s in ['Rillingen','Zieke_Indruk', 'Auscultatie', 'Sputum']]
        
        #todo = [exp for exp in exp_triplets_800 if not exp in finished_exps]
        print(todo)

        for (symptom, model, size) in todo:
            print('_'*100)
            print('STARTING: ******* ' + symptom +' '+ model + ' ' +str(size)+ ' ' + SYMPTOM_TO_TXT_MAP[symptom] + ' ********' )
            print('_'*100)
            t0 = time.time()
            main(LABEL_TO_CLASSIFY=symptom, model_name=model, DS_SIZE=size, TEXT_COLUMN=SYMPTOM_TO_TXT_MAP[symptom])
            took = time.time()-t0
            print('TOOK' + ' ' + str(took/60) +' minutes')
        
        
        
   if False:# STARTED AFTER FIRST SET 
    
        EXTRA_DS_SIZES = [1400, 1200, 1000, 600]
        for size in EXTRA_DS_SIZES:
            for model in MODELS:
                for symptom in SYMPTOMS:
                    print('_'*100)
                    print('STARTING (EXTRA): ******* ' + symptom +' '+ model + ' ' +str(size)+ ' '+ SYMPTOM_TO_TXT_MAP[symptom] + ' ********' )
                    print('_'*100)
                    t0 = time.time()
                    main(LABEL_TO_CLASSIFY=symptom, model_name=model, DS_SIZE=size, TEXT_COLUMN=SYMPTOM_TO_TXT_MAP[symptom])
                    took = time.time()-t0
                    print('TOOK' + ' ' + str(took/60) +' minutes')
       
                    
   if True: # ----------------------------FINAL MODEL DEVELOPMENT
       # NOT: 'Rillingen','Zieke_Indruk',,'Verwardheid'
       for symptom in SYMPTOMS:
            print('_'*100)
            print('STARTING: ******* ' + symptom +  ' '+SYMPTOM_TO_TXT_MAP[symptom]+ '********' )
            print('_'*100)    
            t0 = time.time()
            main_dev_one_model(symptom, 'models--CLTL--MedRoBERTa.nl', f'data/dataset_complete.csv', TEXT_COLUMN=SYMPTOM_TO_TXT_MAP[symptom])
            took = time.time()-t0
            print('TOOK' + ' ' + str(took/60) +' minutes')
            
         