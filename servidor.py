from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import base64
import json
import os


app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "Server Activo para recibir imagenes"

@app.route('/upload', methods=['POST'])
def upload_images():
    if 'image' not in request.files:
        return jsonify({"error": "No se encuentra la imagen"}), 400
    
    image = request.files['image']

    if image.filename == '':
        return jsonify({"error": "No se ha seleccionado ninguna imagen"}), 400
    
    #Nombre de la carpeta, para prox mejoras que se almacene en una carp especifica del usuario
    upload_folder = "processed"
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, image.filename)
    image.save(filepath)

    processed_filepath = process_image(filepath)

    #Codifica la imagen en base 64
    #TODO: Cambiarlo porque al final va a devolver un string
    with open(processed_filepath, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

    #TODO: añadir usuario
    output_json = {
        "filename": image.filename, 
        "image_base64": encoded_image
    }
    #TODO: Cuando usuario implementado modificar esto para que el json se almacene en su cuenta
    json_filepath = os.path.join(upload_folder, f"{os.path.splitext(image.filename)[0]}.json")
    with open(json_filepath, "w") as json_file:
        json.dump(output_json, json_file)

    #envia al cliente esto
    return jsonify({
        "message": "Imagen procesada y guardada",
        "json_path": json_filepath,
        "filename": image.filename,
        "image_base64": encoded_image,
    })


#Función que llamara al modelo y devolverá un string
def process_image(filepath):
    return filepath

if __name__ == '__main__':
    app.run(debug = True, port=5001)