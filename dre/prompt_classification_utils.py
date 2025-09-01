import pandas as pd
from typing import Union

symptom_dict = {
    "Hoesten": {
        "pos": "hoesten positief",
        "neg": "hoesten negatief", 
        "abs": "hoesten afwezig",
    },
    "Koorts": {
        "pos": "ĠKoorts positief",
        "neg": "ĠKoorts negatief",
        "abs": "ĠKoorts afwezig",
    },
    "Kortademigheid": {
        "pos": "ĠKortademigheid positief",
        "neg": "ĠKortademigheid negatief",
        "abs": "ĠKortademigheid afwezig",
    }, #'Verwardheid','Pijn_Borst','Rillingen','Zieke_Indruk', 'Auscultatie', 'Sputum'
    "Verwardheid": {
        "pos": "ĠVerwardheid positief",
        "neg": "ĠVerwardheid negatief",
        "abs": "ĠVerwardheid afwezig",
    },
    "Pijn_Borst": {
        "pos": "pijn borst positief",
        "neg": "pijn borst negatief",
        "abs": "pijn borst afwezig",
    },
    "Rillingen": {
        "pos": "ĠRillingen positief",
        "neg": "ĠRillingen negatief",
        "abs": "ĠRillingen afwezig",
    },
    "Zieke_Indruk": {
        "pos": "zieke indruk positief",
        "neg": "zieke indruk negatief",
        "abs": "zieke indruk afwezig",
    },
    "Auscultatie": {
        "pos": "Ġcrepitaties positief",
        "neg": "Ġcrepitaties negatief",
        "abs": "Ġcrepitaties afwezig",
    },
    "Sputum": {
        "pos": "Ġsputum positief",
        "neg": "Ġsputum negatief",
        "abs": "Ġsputum afwezig",
    },
    "Dyspnoe": {
        "pos": "ĠKortademigheid positief",
        "neg": "ĠKortademigheid negatief",
        "abs": "ĠKortademigheid afwezig",
    },
    "Crepitaties": {
        "pos": "Ġcrepitaties positief",
        "neg": "Ġcrepitaties negatief",
        "abs": "Ġcrepitaties afwezig",
    }
    
}

#    SYMPTOMS =  ['Dyspnoe','Koorts','Hoesten', 'Verwardheid','Pijn_Borst','Rillingen','Zieke_Indruk', 'Crepitaties', 'Sputum']


def construct_prompt(note: str, symptom: str, examples: Union[pd.DataFrame, None] = None):
    symptom_in_text = symptom.replace('_',' ').replace('Auscultatie','Ġcrepitaties')
    prompt = \
        f"""
        Classificeer de volgende teksten op basis van aanwezigheid van het symptoom '{symptom}'.
        Word '{symptom}' vermeld als positief, antwoord '{symptom_dict[symptom]['pos']}'.
        Word '{symptom}' vermeld als negatief, antwoord '{symptom_dict[symptom]['neg']}'.
        Word '{symptom}' niet vermeld in de tekst, antwoord '{symptom_dict[symptom]['abs']}'.
        """
        
    if examples is not None:
        prompt += f"Gebruik de volgende voorbeelden om de classificatie te illustreren:\n"
        for i, row in examples.iterrows():
            if row[symptom] == 2:
                label = 'afwezig'
            if row[symptom] == 1:
                label = 'positief'
            if row[symptom] == 0:
                label = 'negatief'
            prompt += f"Tekst: '{row['DEDUCE_omschrijving']}'\tBijbehorend label: '{label}'\n"
    prompt += f"Print exclusief het label bijbehorende aan de volgende tekst: '{note}'"
    return prompt