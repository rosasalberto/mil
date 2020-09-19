import os

from mil.data.datasets.loader import load_data

current_file = os.path.abspath(os.path.dirname(__file__))
def load():
    return load_data(current_file + './csv/web_recommendation_1.csv')