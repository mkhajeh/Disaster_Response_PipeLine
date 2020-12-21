import sys
import numpy as np
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import matplotlib.pyplot as plt


def load_data(messages_filepath, categories_filepath):
    """
    using this function, you can load two initial 'CSV' files and get the final "Merged data frame"
    ---
    INPUT: the path(name) of the two csv files.
    OUTPUT: df (which is the merge of two input files)
    """

    messages=pd.read_csv(messages_filepath)
    categories=pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories, on='id')
    #just to check the first 5 lines of output
    print(df.head())
    return df


def clean_data(df):
    """
    Using this function, the original (before cleaning) df created in load_data function will be cleaned.
    the major cleaning in this function is making sure the category column is cleaned properly and each category will be converted to separate column.
    ----
    INPUT:
    df before cleaning.

    OUTPUT:
    df after cleaning.
    """
    categories =df['categories'].str.split(';', expand=True)
    #just to see the first 5 lines of categories dataframe.
    #this dataframe is temporary and at the end of this stage it will be re-merged with originial df.
    print(categories.head())
    row = categories.iloc[0]
    category_colnames = row.str.split('-').str.get(0)
    #just to make sure the name is aligned with what I am looking for:
    print(category_colnames)
    categories.columns = category_colnames
    print(categories.head())
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str.get(1)
    print(categories.head())


    for column in categories:
        # set each value to be the last character of the string
        #categories[column] = categories[column].str.split('-').str.get(1)

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    print(categories.head())
    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)
    print(df.head())
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    #replce value of 2 with 1 in 'related' column:
    df['related'].replace({2 : 1})
    # check number of duplicates
    print(df.duplicated().sum())
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df





def save_data(df, database_filename):
    '''
    Saving data as a table in db to use in future modelling phase.
    '''

    #save the cleaned dataframe into a db file
    #drop table if exists
    dbpath = 'sqlite:///{}'.format(database_filename)
    table = 'disaster_response'
    engine = create_engine(dbpath)
    df.to_sql(table, engine, if_exists='replace',index=False)



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
