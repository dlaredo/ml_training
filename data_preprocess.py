import numpy as np
import pandas as pd
import logging
import traceback
import warnings
from collections import Counter

if __name__ == '__main__':

    #Configure logger
    data_logger = logging.getLogger('data_logger')
    data_logger.setLevel(logging.INFO)
    data_fh = logging.FileHandler('data.log')
    data_formatter = logging.Formatter(fmt='%(levelname)s:%(threadName)s:%(asctime)s:%(filename)s:%(funcName)s:%(message)s',
                                    datefmt='%m/%d/%Y %H:%M:%S')
    data_fh.setFormatter(data_formatter)
    data_logger.addHandler(data_fh)

    try:

        #Process data according to solution
        training = pd.read_csv("data_raw/train.csv")
        training = training.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

        #Fill missing values
        training.Age = training.Age.fillna(training.Age.median())
        training.Embarked = training.Embarked.fillna('S')

        #Transform categorical into integer
        embark_dummies_titanic  = pd.get_dummies(training['Embarked'])
        sex_dummies_titanic  = pd.get_dummies(training['Sex'])
        pclass_dummies_titanic  = pd.get_dummies(training['Pclass'], prefix="Class")

        #Put data together
        training = training.drop(['Embarked', 'Sex', 'Pclass'], axis=1)
        titanic = training.join([embark_dummies_titanic, sex_dummies_titanic, pclass_dummies_titanic])

        titanic.to_csv('data/train.csv')

    except Exception as e:

        print("Error while processing data. Please check log")
        data_logger.error("Error while processing data")
        data_logger.error(traceback.format_exc())