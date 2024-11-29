# Named-entity recognition

## 🛠 Skills
- Python
- NLTK
- Spacy

## 📝 Requirements
*The objective is to develop an AI model capable of extrapolating, from the unstructured texts provided as input, the previously established key entities. This process is better known in the literature as [Named-entity recognition (NER)](https://en.wikipedia.org/wiki/Named-entityrecognition).*

## 🔧 Usage
- **Datasets**
	- In order to automate the extraction of the textual contents to be analysed, it was decided to proceed with the use of the Python library [wikipedia](https://pypi.org/project/wikipedia/), which allows you to recover the sections interested in the articles in the famous free encyclopedia. In this way, the number of unstructured textual information that makes up the dataset to be fed to the model can be quickly increased. The texts obtained are then processed using [nltk](https://www.nltk.org/), in order to best adapt them for subsequent annotation.

- **Annotations**
	- To proceed with the annotations, the open-source tool [Spacy-NER-Annotator](https://github.com/ManivannanMurugavel/spacy-ner-annotator) is used. The tool takes as input a `.txt` file and, after defining the labels to be identified, allows it to be manually annotated. The output thus obtained must subsequently be converted into a format compatible with the Python library [Spacy](https://spacy.io/).

- **Process**
	- The tasks to be performed to train and use the model are as follows:
		1. Run the `01_get_text_from_wikipedia.py` script
		2. Open the `spacy-ner-annotator` tool by inputting the file to process
		3. Proceed to manually annotate the text
		4. Save the annotated file as `annotated_text.json`
		5. Run the `02_convert_spacy_train_data.py` script
		6. Copy the annotated text in the Spacy format present in the `annotated_text.txt` file
		7. Paste the text above into the `03_train_model.py` file
		8. Run the `03_train_model.py` script
		9. Run the `04_test_model.py` script
		10. Run the `05_model_evaluation.py` script
