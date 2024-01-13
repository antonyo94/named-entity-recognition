import wikipedia
import nltk
#nltk.download('punkt')

#Set lingua italiana per wikipedia
wikipedia.set_lang("it")
articles = ["Michael Dell", "J.K. Rowling", "Walt Disney", "Howard Schultz", "Daniel Ek"]
text = ""

#Estrazione articoli
for i in articles:
    text += wikipedia.summary(i)

#Preprocessing articoli da annotare
sentences = nltk.tokenize.sent_tokenize(text)
for i in sentences:
    print(i)

#Salvataggio testo da annotare
file = open("testo_da_annotare.txt","w+", encoding="utf-8")
for i in sentences:
    file.write(i + "\n")
file.close
