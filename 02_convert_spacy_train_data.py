import json

#Set nome del file in input
filename = "testo_annotato.json"
print("[INFO] Nome file in input da convertire: " + filename)

#Caricamento del file in input
with open(filename, mode="r", encoding="utf-8") as train_data:
	train = json.load(train_data)

#Conversione del testo annotato nel formato Spacy
TRAIN_DATA = []
for data in train:
	ents = [tuple(entity[:3]) for entity in data['entities']]
	TRAIN_DATA.append((data['content'],{'entities':ents}))

#Salvataggio del testo annotato nel formato Spacy
with open('{}'.format(filename.replace('json','txt')),'w',encoding="utf-8") as write:
	write.write(str(TRAIN_DATA))

print("[INFO] Dati di addestramento salvati. Il nome del file è: {}".format(filename.replace('json','txt')))