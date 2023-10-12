import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


def predict_user_image(filename):
    model = load_model('train/train.h5')

    img_path = 'static/images/' + filename
    img = Image.open(img_path).convert('L')
    img=img.resize((8,98))
    user_img = np.asarray(img).astype('float32')
    user_img = user_img / 255.0

    print(user_img.shape)

    user_img = user_img.reshape(1,98,8,1)

    prediction = model.predict(user_img)

    pred_digit = np.argmax(prediction)

    print(f"Predict Num is {pred_digit}")
    return pred_digit


#test = predict_user_image('9.png')


