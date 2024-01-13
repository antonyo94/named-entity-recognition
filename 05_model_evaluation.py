import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer
import json
from pathlib import Path

def evaluate(ner_model, test_data):
    scorer = Scorer()
    for input_, annot in test_data:
        doc_gold_text = ner_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot['entities'])
        pred_value = ner_model(input_)
        scorer.score(pred_value, gold)
    return scorer


TEST_DATA = \
[
#"Michael Dell", "J.K. Rowling", "Walt Disney", "Howard Schultz", "Daniel Ek"
('Michael Saul Dell (Houston, 23 febbraio 1965) è un imprenditore statunitense.', {'entities': [(19, 26, 'CITTA'), (13, 17, 'COGNOME')]}), ("Fondatore e CEO della Dell, Inc., un'azienda tra i leader mondiali nel settore delle forniture hardware per computer, nell'ottobre del 2015 rileva per 67 miliardi di dollari la EMC Corporation, gigante americano di servizi informatici per le imprese.", {'entities': [(177, 192, 'AZIENDA'), (22, 26, 'AZIENDA')]}), ("Con questa operazione di fusione, perfezionata nel settembre 2016, nasce Dell Technology, un colosso mondiale nella tecnologia dell'informazione.", {'entities': [(73, 88, 'AZIENDA')]}), ('Secondo la classifica stilata dalla rivista Forbes Michael Dell è, a dicembre 2020, il 33 uomo più ricco del mondo con un patrimonio stimato in 40 miliardi di dollari americaniJoanne Rowling (pron.', {'entities': [(59, 63, 'COGNOME'), (44, 50, 'AZIENDA')]}), ('[joæn roulin]; Yate, 31 luglio 1965) è una scrittrice, sceneggiatrice e produttrice cinematografica britannica.', {'entities': [(15, 19, 'CITTA')]}), ('La sua fama è legata alla serie di romanzi di Harry Potter, che ha scritto firmandosi con lo pseudonimo J. K. Rowling (in cui "K" sta per Kathleen, nome della nonna paterna), motivo per cui la scrittrice è spesso indicata impropriamente come Joanne Kathleen Rowling.', {'entities': [(258, 265, 'COGNOME'), (110, 117, 'COGNOME'), (52, 58, 'COGNOME')]}), ('Nel 2013 pubblica la sua prima opera con lo pseudonimo di Robert Galbraith.', {'entities': [(65, 74, 'COGNOME')]}), ("Nel 2011 è stata inserita da Forbes nella classifica delle donne più ricche del Regno Unito.Walt Disney, all'anagrafe Walter Elias Disney (Chicago, 5 dicembre 1901 – Burbank, 15 dicembre 1966), è stato un animatore, imprenditore, produttore cinematografico, regista e doppiatore statunitense.", {'entities': [(166, 173, 'CITTA'), (139, 146, 'CITTA'), (131, 137, 'COGNOME'), (97, 103, 'COGNOME')]}), ("Annoverato tra i principali cineasti del XX secolo e riconosciuto come uno dei padri dei film d'animazione, ha inoltre creato Disneyland, il primo di una serie di parchi a tema; è altresì noto per la sua grande abilità nella narrazione di storie, come divo televisivo e uno dei grandi artisti del XX secolo nel campo dell'intrattenimento.", {'entities': [(126, 136, 'AZIENDA')]}), ('Detiene il record di Premi Oscar vinti, avendo ricevuto, in 34 anni di carriera, per i suoi cortometraggi e documentari, 59 candidature e 26 premi, di cui 3 onorari e un Premio alla memoria Irving G. Thalberg.', {'entities': [(200, 208, 'COGNOME')]}), ('È noto per essere stato amministratore delegato di Starbucks dal 1987.Daniel Georg Ek (Stoccolma, 23 febbraio 1983) è un imprenditore svedese.', {'entities': [(87, 96, 'CITTA'), (83, 85, 'COGNOME'), (51, 60, 'AZIENDA')]})
]
#PATH in cui è salvato il modello
model_output_dir=Path(r'SpacyNerTrainedModel')

# Caricamento modello addestrato dalla directory precedente
print("[INFO] Caricamento modello dal path: ", model_output_dir)
trained_nlp = spacy.load(model_output_dir)
results = evaluate(trained_nlp, TEST_DATA)

#Print evaluation scores for every entity
print(json.dumps(results.ents_per_type, indent=4))

#Print model Precision, Recall, F-Score
print("Precision: ",round(results.ents_p,2),"\nRecall: ",round(results.ents_r,2),"\nF-Score: ",round(results.ents_f,2))
