#!/home/sudharsan/myenv/bin/python3
from flask import Flask, request, jsonify,send_from_directory
import requests
import os
from datetime import datetime


from llm import llm
from model import model
from model import model_fruit
from llm import normal_llm

import lat_to_place as location
import database

import language_conversion

import ensem

app = Flask(__name__)

db=None
UPLOAD_DIRECTORY = "/home/sudharsan/projects/sih_model/uploads/"
Model_dir="/home/sudharsan/projects/sih_model/model/tflite/"

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#=============================================[react endpoint]=========================================================== 
@app.route('/api', methods=['POST'])                  #To make vecctor embeddings and get response from LLM
def add_user():
    global db
    try:
        new_user = request.json  
        print(new_user)
        
        if 'url' not in new_user or 'query' not in new_user:
            return jsonify({'error': 'Missing required fields'}), 400
        
        db = llm.create_vector_db(new_user['url'])
        response = llm.get_response(db, new_user['query'])
        
        return jsonify({'message': response}), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

#=============================================[flutter endpoint]=========================================================== 


@app.route('/check_endpoint', methods=['get'])     #To verify the IP address
def check():
    return jsonify({'check':True}),201

@app.route('/web_loader', methods=['post'])         #To load the website and convert it into vector embeddings
def load():
    global db
    data=request.json;

    path=data['file_path']


    db=llm.create_vector_db(path)
    return jsonify({'error':False,'body':"website exists"}),201
    
@app.route('/reply', methods=['POST'])                  #To make vecctor embeddings and get response from LLM
def reply():
    global db
    try:
        new_user = request.json  
        print(new_user)

        selected_lang=new_user['query']

        output_lang=new_user['lang']

        eng=language_conversion.translate_to_english(selected_lang)

        response=llm.get_response(db,eng)

        response=language_conversion.english_to_other(output_lang,response)

        return jsonify({'message': response,'error':False}), 200
    
    except Exception as e:
        print(e)
        return jsonify({'message': 'UnExpected Error occurs','error':True}), 500
#====================================================================================================

@app.route('/upload', methods=['POST'])
def upload_file():
    print('uploading called')
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    type_=request.form.get('type')
    crop_name=request.form.get('crop_name')
    
    lat=float(request.form.get('location[latitude]'))
    lon=float(request.form.get('location[longitude]'))

    place = location.get_district_name(lat,lon)

    print(place)

    predicted_det={}

    predicted_det['district']=place
    predicted_det['type']=type_
    predicted_det['plant-name']=crop_name

    predicted_det['coordinates']=[lat,lon]
    
    if file:
        # Generate a unique filename based on current time
        current_datetime = datetime.now()
        timestamp = current_datetime.strftime('%Y%m%d_%H%M%S_%f')
        filename = f"img_{timestamp}.jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save the file to the uploads folder
        file.save(file_path)

        predicted_det['img']=filename

        # Check if the file exists before processing
        if os.path.exists(file_path):
            try:
                # Process the file (e.g., classification)
               # if(type_.lower()=="leaf"):
                    #result = model.img_classification(file_path)
                #elif(type_.lower()=="fruit"):
                    #result=model_fruit.img_classification(file_path)
                result=ensem.ensem_predict(type_.lower(),crop_name.lower(),file_path)
                predicted_det['Disease-name']=result

                print(predicted_det)

                data_insertion_id=database.add_post_det(predicted_det)

                return jsonify({'diseased': True, 'result': result,'insert_id':data_insertion_id}), 200
            except Exception as e:
                print(e)
                return jsonify({'error': f'Error processing file: {str(e)}'}), 500
        else:
            return jsonify({'error': 'File not found after saving'}), 500

@app.route('/common_reply',methods=['post'])
def normal_reply():
    try:
        new_user = request.json  
        print(new_user)


        selected_lang=new_user['query']

        output_lang=new_user['lang']

        eng=language_conversion.translate_to_english(selected_lang)

        response=normal_llm.start_app(eng)

        response=language_conversion.english_to_other(output_lang,response)

        return jsonify({'message': response,'error':False}), 200
    except Exception as e:
        print(error)
        return jsonify({'message': 'UnExpected Error occurs','error':True}), 500

@app.route('/post_data',methods=['get'])
def post_data():
    try:
        # Fetch the documents
        disease = database.find_post()

        # Serialize documents to handle ObjectId
        disease_serialized = [serialize_document(doc) for doc in disease]

        # Return the serialized data
        return jsonify({'Data': disease_serialized}), 200

    except Exception as e:
        print(e)
        return jsonify({'message': 'Unexpected Error occurs', 'error': True}), 500



@app.route('/post_img/<path:filename>')
def serve_file(filename):
    return send_from_directory(UPLOAD_DIRECTORY, filename)

@app.route('/model_download/<path:filename>')
def model_send_file(filename):
    return send_from_directory(Model_dir, filename)



@app.route('/store_chat', methods=['POST'])
def store_chat():

    data=(request.json)
    database.update_document(data['insertion_id'],'convo',data['convo'])
    return jsonify({'diseased': True}), 200

@app.route('/available_offline_models', methods=['POST'])
def available_models():
    try:
        models={
            
                "apple_leaf": "apple_leaf.tflite",
                "corn_leaf":"corn_leaf.tflite",
                "potato_leaf":"potato_leaf.tflite",
                "grape_leaf":"grape_leaf.tflite",
                "paddy_leaf":"paddy_leaf.tflite" ,

                "apple_fruit":"apple_fruit.tflite",
                "potato_fruit":"potato_fruit.tflite",


                "apple_leaf_label":"apple_leaf.txt",
                "corn_leaf_label":"corn_leaf.txt",
                "potato_leaf_label":"potato_leaf.txt",
                "grape_leaf_label":"grape_leaf.txt",
                "paddy_leaf_label":"paddy_leaf.txt",

                "apple_fruit_label":"apple_fruit.txt",
                "potato_fruit_label":"potato_fruit.txt"          
            }


        return jsonify({'offline_models':models}),200

    except Exception as e:
        print(e)
        return jsonify({'message': 'Unexpected Error occurs', 'error': True}), 500

def serialize_document(doc):
    if '_id' in doc:
        doc['_id'] = str(doc['_id'])
    return doc

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
