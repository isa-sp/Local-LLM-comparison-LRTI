import pandas as pd
import pyreadstat
from sklearn.model_selection import KFold

import locale
locale.setlocale(locale.LC_ALL, "nl_NL")

def prepare_freetext_data():
    # Load ehr dataset
    df_ehr = pd.read_csv('Element_freetext_SO.csv', sep='|', usecols=['Patnr', 'start_epi', 'eind_epi', 'SOEPcode', 'start_icpc', 'DEDUCE_omschrijving,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,'])
    
    df_ehr['start_epi'] = pd.to_datetime(df_ehr['start_epi'], format='%d/%m/%Y')
    #df_ehr['eind_epi'] = pd.to_datetime(df_ehr['eind_epi'], format='%d/%m/%Y')
    return df_ehr
    
def prepare_labeled_data():
    # Load labeled symptoms sav dataset
    #symptoms = ['Temp', 'Hoesten', 'Dyspnoe']
    symptoms = ['Koorts', 'Hoesten', 'Dyspnoe', 'Sputum', 'Verwardheid','Pijn_Borst','Rillingen','Zieke_Indruk', 'Crepitaties']

    keep_columns = ['Patnr', 'start_epi', 'start_icpc'] + symptoms

    df_spss, meta = pyreadstat.read_sav('sample_2000.sav')
    df_symptoms = df_spss[keep_columns]

    # Parse koorts data (0=geen koorts, 1=koorts, 2=niet aangegeven)
    #df_symptoms['Temp'] = df_symptoms['Temp'].apply(lambda x: 0 if x in [0, 1] else 1 if x == 2 else 2)


    #print(df_symptoms['Auscultatie'])
    #df_symptoms['Auscultatie'] = df_symptoms['Auscultatie'].apply(lambda x: 1 if x in [1, 8] else (0 if x in [0,2,3,4,5,6,7] else 2))


    df_symptoms.loc[:, symptoms] = df_symptoms[symptoms].fillna(2)
    
    
    df_symptoms[symptoms]=df_symptoms[symptoms].astype('int')

    # Parse date-related fields
    df_symptoms['start_epi'] = pd.to_datetime(df_symptoms['start_epi'], format='%d-%b-%Y')
    
    #df_symptoms['eind_epi'] = pd.to_datetime(df_symptoms['eind_epi'], format='%d-%b-%Y')
    
    print(df_symptoms)
    print("*** Data Statistics ***")
    
   
    for symptom in symptoms:
        print(symptom)
        print('0 - absent: ' + str(list(df_symptoms[symptom]).count(0)))
        print('1 - present: ' + str(list(df_symptoms[symptom]).count(1)))
        print('2 - not reported: ' + str(list(df_symptoms[symptom]).count(2)))
    
    #df_symptoms = df_symptoms.rename(columns={'Dyspnoe': 'Kortademigheid', 'Temp': 'Koorts'})

    return df_symptoms
    
def combine_datasets(df_ehr, df_symptoms):
    # Combine dataframes
    df = pd.merge(df_ehr, df_symptoms, how='inner', on=['Patnr', 'start_epi', 'start_icpc'])

    df = df.rename(columns={'DEDUCE_omschrijving,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,': 'DEDUCE_omschrijving'})
    df['DEDUCE_omschrijving'] = df['DEDUCE_omschrijving'].apply(lambda x: x.strip(',')[:500])   # remove trailing commas


    print(df['SOEPcode'])
    # Combine entries based on SOEPcode
    #for r in df.iterrows():
    #    print(r[1]['SOEPcode'])
    #exit()
    df['DEDUCE_omschrijving_S'] = [r[1]['DEDUCE_omschrijving']  if r[1]['SOEPcode']=='S' else "" for r in df.iterrows()]  # remove trailing commas
    df['DEDUCE_omschrijving_O'] = [r[1]['DEDUCE_omschrijving']  if r[1]['SOEPcode']=='O' else "" for r in df.iterrows()]  # remove trailing commas

    #print(df['DEDUCE_omschrijving_S'])
    df = df.copy().groupby('Patnr').apply(pd.DataFrame.assign, SOEPcode='SO', DEDUCE_omschrijving=lambda x: '\n'.join(x.DEDUCE_omschrijving), DEDUCE_omschrijving_S=lambda x: '\n'.join(x.DEDUCE_omschrijving_S), DEDUCE_omschrijving_O=lambda x: '\n'.join(x.DEDUCE_omschrijving_O)).drop_duplicates()

    #print(df['DEDUCE_omschrijving_S'])
    #print(df.columns)

    #df[['Hoesten', 'Kortademigheid']] = df[['Hoesten', 'Kortademigheid']].astype('int')
    #exit()
    return df


def split_dataset(df: pd.DataFrame, k: int, shuffle: bool = True) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    kf = KFold(n_splits=k, shuffle=shuffle, random_state=42)
    folds = []
    for train_index, test_index in kf.split(df):
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]
        folds.append((train_data, test_data))
    return folds

def main():
    df_ehr = prepare_freetext_data()
    df_symptoms = prepare_labeled_data()
    merged_df = combine_datasets(df_ehr, df_symptoms)
    
    print(merged_df.head())
    
    
    # Save final dataframe, rearranging col_names
    ordered_columns = ['Patnr', 'start_epi', 'start_icpc', 'SOEPcode'] + ['Koorts', 'Hoesten', 'Dyspnoe', 'Sputum', 'Verwardheid','Pijn_Borst','Rillingen','Zieke_Indruk', 'Crepitaties'] + ['DEDUCE_omschrijving','DEDUCE_omschrijving_S','DEDUCE_omschrijving_O']
    merged_df[ordered_columns].to_csv('data/dataset_complete.csv', sep='|', index=False)


    
    k = 5
    folds = split_dataset(merged_df, k)
    for fold_index, (train_data, test_data) in enumerate(folds):
        train_data[ordered_columns].to_csv(f'data/train/fold_{fold_index}.csv', sep='|', index=False)
        test_data[ordered_columns].to_csv(f'data/test/fold_{fold_index}.csv', sep='|', index=False)
        

if __name__ == "__main__":
    main()

