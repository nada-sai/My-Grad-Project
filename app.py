import numpy as np
import os
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify

import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# Google Drive file IDs
# model_links = {
#     "pistachio": "1--VPQcNBlI78iL9f5HGay3z_RTZ-6xGX",
#     "corn": "1Axhg5KKrEt3ribKL_YVk5PL-lIdTrf2c",
#     "soya": "1LIH5q2vm5L9ihqYs2kbVo73ewWDBp6pt",
#     "seed": "160_t0uZZknDn19HxTtbYforZIsPqmeyV"
# }
#
#
# # Download models function
# def download_model(model_name):
#     file_id = model_links[model_name]
#     output = f"{model_name}_classifier.h5"
#     if not os.path.exists(output):
#         gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
#     return output
#
#
# # Download all models on startup
# for model in model_links:
#     download_model(model)
#
# pistachio = download_model('pistachio')
# corn = download_model('corn')
# soya = download_model('soya')
# seed = download_model('seed')
#
# pistachio_model = load_model(pistachio)
# corn_model = load_model(corn)
# soya_model = load_model(soya)
# seed_model = load_model(seed)

pistachio_model=load_model('models/pistachio_vgg_classifier.h5')
corn_model=load_model('models/corn_classifier.h5')
soya_model=load_model('models/soya_classifier.h5')
seed_model=load_model('models/seed_classifier.h5')
# Preprocess Functions
def preprocess_pistachio(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = preprocess_input(img)
    return img


def preprocess_corn(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img


def preprocess_soya(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img


def preprocess_seed(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img


# Predict Functions
def predict_pistachio(image_path):
    preprocessed_image = preprocess_pistachio(image_path)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    pred_prob = pistachio_model.predict(preprocessed_image)
    prediction = (pred_prob > 0.5).astype(int)
    if prediction[0][0] == 0:
        return 'Unhealthy'
    else:
        return 'Healthy'


def predict_corn(image_path):
    img = preprocess_corn(image_path)
    img = np.expand_dims(img, axis=0)
    prob = corn_model.predict(img)
    prediction = (prob > 0.5).astype(int)
    if prediction[0][0] == 1:
        return 'Unhealthy'
    else:
        return 'Healthy'


def predict_soya(image_path):
    img = preprocess_soya(image_path)
    img = np.expand_dims(img, axis=0)
    prob = soya_model.predict(img)
    prediction = (prob > 0.5).astype(int)
    if prediction[0][0] == 1:
        return 'Unhealthy'
    else:
        return 'Healthy'


def predict_seed(image_path):
    class_labels = ['Corn', 'Pistachio', 'Soybean']
    img = preprocess_seed(image_path)
    img = np.expand_dims(img, axis=0)
    prediction = seed_model.predict(img)
    predicted_class = np.argmax(prediction)
    return class_labels[predicted_class]


def predict_image(image_path):
    seed_class = predict_seed(image_path)
    print(seed_class)
    if seed_class == 'Corn':
        seed_healthiness=predict_corn(image_path)
        if seed_healthiness=='Healthy':
            return f"This Is A {seed_class} Seed And it's {seed_healthiness}"
        else:
            return f"""This Is A {seed_class} Seed And it's {seed_healthiness}\nCorn seeds unsuitable for consumption or planting due to disease or contamination can be redirected to:\n
- **Animal Feed**: After ensuring that harmful pathogens are eliminated through proper processing, these seeds can serve as feed for livestock.\n
- **Industrial Uses**: Damaged corn seeds can be processed into ethanol for fuel or used in the production of biodegradable plastics.\n
\nFor more details on common corn diseases and their management, refer to (https://www.stineseed.com/blog/part-2-common-corn-and-soybean-diseases/)."""
    elif seed_class == 'Soybean':
        seed_healthiness=predict_soya(image_path)
        if seed_healthiness=='Healthy':
            return f"This Is A {seed_class} Seed And it's {seed_healthiness}"
        else:
            return f"""This Is A {seed_class} Seed And it's {seed_healthiness}\nSoybeans that are not fit for traditional uses can still be valuable in:\n
- **Silage**: Combining soybeans with corn to produce silage offers nutritious livestock feed. A typical mixture is two parts corn to one part soybeans.\n  
- **Green Manure**: Incorporating unhealthy soybeans into soil can enhance organic matter and nutrient content, benefiting subsequent crops.\n
\nFor more information on utilizing soybeans for silage and green manure, visit (https://www.ecofarmingdaily.com/grow-crops/grow-soybeans/soybean-crop-science/hnon-human-food-uses/)._"""
    else:
        seed_healthiness=predict_pistachio(image_path)
        if seed_healthiness=='Healthy':
            return f"This Is A {seed_class} Seed And it's {seed_healthiness}"
        else:
            return f"""This Is A {seed_class} Seed And it's {seed_healthiness}\nPistachios can become contaminated with aflatoxins if not stored properly, posing health risks. However, these compromised nuts can be utilized in non-food applications:\n
- **Cosmetic Industry**: Pistachio oil, rich in vitamin E and fatty acids, is valued in skincare products. Extracting oil from these nuts can provide ingredients for moisturizers and hair care items.\n
- **Biofuel Production**: The oil content in pistachios can be processed into biodiesel, offering a renewable energy source.\n
\nFor more information on the risks associated with improperly stored pistachios, you can visit (https://domesticfits.com/pistachios-benefits/)._"""




@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image file provided."}), 400

    temp_path = "temp_image.jpg"
    file.save(temp_path)

    try:
        result = predict_image(temp_path)
    except Exception as e:
        os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

    os.remove(temp_path)
    return jsonify({"result": result})


if __name__ == '__main__':
    app.run(debug=True)