import requests
import os
import logging
import pandas as pd



def load_data(path = "labeled_newscatcher_dataset.csv",url = "https://raw.githubusercontent.com/kotartemiy/topic-labeled-news-dataset/refs/heads/master/labeled_newscatcher_dataset.csv"):
    logger = logging.getLogger(__name__)
    logger.info("ingest_data")

    if os.path.isfile(path):
        logger.info("find data set locally")
    else:
        logger.info("downloading data set from url")
        try:
            r = requests.get(url)
            with open(path, "wb") as f:
                f.write(r.content)
            logger.info("data set downloaded")
        except Exception as e:
            logger.error(f"download data set failed: {e}")
    return pd.read_csv(path,sep=';')


def clean_and_save_news_data(df,path = 'cleaned_news_data.csv'):
    logger = logging.getLogger(__name__)
    logger.info("ingest_data")
    logger.info("cleaning the data")

    df = df[ df['lang'] ==  'en']
    df = df.drop('lang',axis = 1)
    df = df.drop('published_date',axis = 1)
    df = df.drop_duplicates()
    logger.info("data cleaned")
    df.to_csv(path, index=False)
    logger.info("cleaned data saved")