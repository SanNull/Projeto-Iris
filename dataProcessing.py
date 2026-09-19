import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import spacy


class dataProcessing:
    def __init__(self, data):
        self.data = data
        self.SPECIAL_CHARS_REGEX = 'R\\$|[_!@#$^º;&*?.\\-\\|\\—\\(\\)]'

    def process_data(self):
        df = pd.read_csv("data.csv")
        df = df.rename(columns=({"Descrição":"Descricao", "Vaga Arriscada?": "Arriscada", "Salário": "Salario Coletado"}))

        data = self.__clean_special_characters(df)
        data = self.__convert_salaries(data)
        data = self.__remove_stopwords_and_lematize(self, df)

        #Convert categories into one-hot booleans
        fatores_df = data['Fatores de Risco'].str.get_dummies(", ", 'bool')

        data = df.drop(['Fonte','Link', 'Termo_Busca', 'Comentários', 'Nome do Membro', 'Fatores de Risco'], axis= 1)
        data = pd.concat([data, fatores_df], axis=1)
        data['Arriscada'] = data['Arriscada'].fillna(0).astype('bool')
        return data



    def __clean_special_characters(self, df):
        #Limpeza de caracteres especiais
        df['Descricao'] = df['Descricao'].str.replace(self.SPECIAL_CHARS_REGEX, '', regex=True)
        df['Cargo'] = df['Cargo'].str.replace(self.SPECIAL_CHARS_REGEX + '|[0-9]', '', regex=True)
        df['Salario Coletado'] = df['Salario Coletado'].str.replace(self.SPECIAL_CHARS_REGEX, '', regex=True)
        df['Salario Coletado'] = df['Salario Coletado'].str.replace(',', '.', regex=True)
        df['Salario Coletado'] = df['Salario Coletado'].fillna("A combinar").astype("str")
        return df

    def __convert_salaries(self,df):
        #Transformar salários em valores numéricos, faixas de valores em uma média
        salaries = df['Salario Coletado'].str.findall(r'(\d+(?:\.\d+)?)').to_list()
        df['Salario'] = 0.0
        df['Media'] = 0.0

        for i in range(0, len(salaries)):
            if len(salaries[i]) < 2: 
                if len(salaries[i]) == 0: continue
                df.at[i, 'Salario'] = float(salaries[i][0])
                continue
            df.at[i, 'Salario'] = (float(salaries[i][0]) + float(salaries[i][1]))/2.0

        jobs_average_salary = df[['Salario', 'Termo_Busca']].groupby(['Termo_Busca']).mean()
        jobs_average_dict = dict(zip(jobs_average_salary.axes[0], np.round(jobs_average_salary.values,2)))

        #The average across all ocupations
        all_jobs_average = np.mean(list(jobs_average_dict.values()))

        for i in range(0, len(salaries)):
            if df.at[i,'Salario'] == 0:
                df.at[i,'Salario'] = all_jobs_average
                df.at[i,'Media'] = all_jobs_average
            else:
                df.at[i, 'Media'] = jobs_average_dict[df.at[i, 'Termo_Busca']] 
        return df       

    def __remove_stopwords_and_lematize(self, df):
        #nltk.download('stopwords')
        nlp = spacy.load('pt_core_news_sm')
        pt_stp_words = stopwords.words('portuguese')
        for i in range(0, len(df["Descricao"])):
            tokenized_descriptions = nlp(df.at[i, 'Descricao'])
            tokens = [token.text for token in tokenized_descriptions]
            filtered_tokens = [word for word in tokens if word not in pt_stp_words]
            filtered_descrpition = " ".join(filtered_tokens)
            lematized_description = nlp(filtered_descrpition)
            lematized_tokens = [token.lemma_ for token in lematized_description]
            df.at[i, 'Descricao Processada'] = " ".join(lematized_tokens)
        return df



            
            


oi = dataProcessing("data.csv").process_data().to_csv("lematizado.csv")