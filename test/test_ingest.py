from src.data_processing.ingest import clean_and_save_news_data,load_data
import pandas as pd
import os
def test_load_data():
    assert isinstance(load_data(), pd.DataFrame)

def test_clean_and_save_news_data():
    df = load_data()
    path = 'cleaned_news_data.csv'
    clean_and_save_news_data(df,path)
    assert os.path.isfile(path)
#check the file contain right data
#test every