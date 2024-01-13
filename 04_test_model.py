import spacy
from pathlib import Path
import wikipedia

#PATH in cui è salvato il modello
model_output_dir=Path(r'SpacyNerTrainedModel')

# Caricamento modello addestrato dalla directory precedente
print("[INFO] Caricamento modello dal path: ", model_output_dir)
trained_nlp = spacy.load(model_output_dir)

#Set lingua italiana per wikipedia
wikipedia.set_lang("it")
articles = ["Michael Dell", "J.K. Rowling", "Walt Disney", "Howard Schultz", "Daniel Ek"]
test_text = ""

#Estrazione articoli
for i in articles:
    test_text += wikipedia.summary(i)

#Test del modello addestrato
doc=trained_nlp(test_text)
for ent in doc.ents:
  print(ent.text,ent.label_)