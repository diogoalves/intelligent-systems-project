import os
import pickle
import pandas as pd

from flask import Flask
from flask import request


# 1. Model Loading
# Loads pretrained models from a specified path available in the environment 
#  variable MODEL_PATH and KMEANS_PATH.
file_name = os.environ['MODEL_PATH']
with open(file_name, 'rb') as open_file:
    classifier = pickle.load(open_file)
file_name = os.environ['KMEANS_PATH']
with open(file_name, 'rb') as open_file:
    kmeans = pickle.load(open_file)

# 2. Categorization Endpoint
# Exposes a POST endpoint /v1/categorize that receives a JSON with product data
#  and returns a JSON with their predicted categories.
api = Flask(__name__)
@api.route('/v1/categorize', methods=['POST'])
def categorizebatch():

    # 3. Input validation
    # Returns status 400 (Bad Request) in case of ill-formatted user input without killing the API.
    try:
        body = request.json
    except:
        return { "error": "Entrada mal formatada. A entrada deve ser um JSON."}, 400

    if "products" not in body:
        return { "error": "Entrada mal formatada. Campo 'products' não existe"}, 400

    for product in body['products']:
        for column in ['title', 'concatenated_tags', 'query', 'price', 'weight', 'minimum_quantity']:
            if column not in product:    
                return { "error": f"Entrada mal formatada. Campo '{column}' não existe no Produto {product}"}, 400


    # 4. Data preparation
    # Shape the input to feed the petrained classifier.
    df = pd.DataFrame.from_dict(data=body['products'])
    try:
        data = data_format(df)
    except:
        return { "error": f"Ocorreu um erro irterno durante a etapa de formatação da entrada."}, 500
    
    # Classifying
    try:
        categories = classifier.predict(data)
    except:
        return { "error": f"Ocorreu um erro irterno durante a etapa classificação."}, 500
    
    return { 'categories': categories.tolist()}    


def data_format(X):
    kmeansArray  = kmeans.predict(X[['price', 'weight', 'minimum_quantity']])
    kmeansSeries = pd.Series(kmeansArray, name="kmeans")
    X = pd.concat([X, kmeansSeries], axis=1)
    X['kmeansPriceWeightMinimumQuantity'] = 'grupo' + X['kmeans'].astype(str)
    return X['title'] + ' ' +  X['concatenated_tags'] + ' ' +  X['query'] + ' ' +  X['kmeansPriceWeightMinimumQuantity']