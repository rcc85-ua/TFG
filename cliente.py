from flask import Flask, jsonify, request, render_template, send_file
from flask_cors import CORS
import io
import requests

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')

def home():
    return "La API Funciona!"

# Ruta de la API
@app.route('/api/saludo', methods=['GET'])
def saludo():
    return jsonify({"mensaje": "Hola desde la API!"})

#RUTA PARA ENVIAR IMAGEN
@app.route('/upload', methods = ['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No se encontró imagen'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No se ha seleccionado ninguna imagen'}), 400
    
    try:
        server_url = 'http://127.0.0.1:5001/upload'
        files = {'image': (file.filename, file.read(), file.mimetype)}

        response = requests.post(server_url, files = files)

        if response.status_code == 200:
            server_data = response.json()
            print(server_data['filename'])
            #Enviar la respuesta con el formato esperado

            return jsonify({
                'message' : 'Imagen enviada y procesada',
                #'server_response': server_data,  # Asegúrate de que esta estructura sea correcta
                'filename': server_data['filename'],
                'image_base64': server_data['image_base64'],
                'json_path': server_data['json_path']
            })
        else:
            return jsonify({'error': 'Error al procesar la imagen en el servidor', 
                            'status_code': response.status_code}), 500
    except Exception as e:
        return jsonify({'error': 'Error de conexión con el servidor', 'details': str(e)}), 500


##############################################################################
    
    #AQUI SE TIENE QUE HACER LO QUE SE HAGA A LA IMAGEN

##############################################################################
    #Saca los datos binarios
    imagen_bytes = io.BytesIO(file.read())
    imagen_bytes.seek(0)
    return send_file(imagen_bytes, mimetype = 'image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, port=5000)