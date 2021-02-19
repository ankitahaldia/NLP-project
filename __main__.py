from NLP_project.pipeline.preprocess.nlp_model import Model
import pandas as pd

if __name__ == "__main__" :
    DATA_PATH = (r'/home/becode/NLP-project/assets/new_file.csv')
    #testing a 100 docs sample
    df = pd.read_csv(DATA_PATH, delimiter='\t')
    model = Model(nlp_model='en_core_web_md')
    powersetsvc, vectorizer = model.train(data= df,X_column= 'text')

