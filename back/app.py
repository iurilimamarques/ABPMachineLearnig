from flask import Flask, request, jsonify
from joblib import load  
from flask_cors import CORS 
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

app = Flask(__name__)
CORS(app) 

large_files_folder = '../large-files'

modelo, ref_cols, target = load(large_files_folder + '/model_nota_MEDIA.pkl')

@app.route('/prever', methods=['POST'])
def prever():
    dados = request.json
    try:
        loaded_encoder = load(large_files_folder + '/encoder.pkl')

        dados = request.json

        if not isinstance(dados, dict):
            return jsonify({'erro': 'Os dados enviados devem estar no formato JSON'}), 400

        dados = {key.upper(): value for key, value in dados.items()}
        entrada_df = pd.DataFrame([dados], columns=ref_cols)
        entrada_df = entrada_df.sort_index(axis=1)
        
        columns_to_convert = entrada_df.select_dtypes(include=['object', 'bool']).columns

        entrada_df[columns_to_convert] = loaded_encoder.transform(entrada_df[columns_to_convert])
        
        previsao = modelo.predict(entrada_df)

        return jsonify({'nota': previsao[0]})
    except Exception as e:
        return jsonify({'erro': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False)
