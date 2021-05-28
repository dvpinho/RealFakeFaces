from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from tensorflow import expand_dims


def get_model(model_path: str):
    """
    Loads a previously trained keras model.

    :param model_path: model path
    :return: model
    """

    return load_model(model_path)


def get_model_summary(model):
    """
    Prints the keras model summary.

    :return: model summary string
    """

    model_summary_string = []
    model.summary(print_fn=lambda x: model_summary_string.append(x), line_length=78)

    return '\n'.join(model_summary_string)


def model_prediction(image_path: str, model):
    """
    Calculates the prediction label and probability given an image and model as inputs.

    :param image_path: path of face image to be fed to the model.
    :param model: pre-trained keras model.
    :return: prediction label, prediction probability for each class.
    """

    img = load_img(image_path, target_size=(300, 300))
    img_array = img_to_array(img)
    img_array = expand_dims(img_array / 255, 0)

    sigmoid_threshold = 0.423
    prediction = model.predict(img_array)[0][0]

    classes = ['Real Face', 'Fake Face']
    label = classes[0] if prediction > sigmoid_threshold else classes[-1]

    classes_probability = _prediction_probability(prediction, label, classes)

    real_face_probability = classes_probability['Real Face']
    fake_face_probability = classes_probability['Fake Face']

    return label, real_face_probability, fake_face_probability


def _prediction_probability(prediction: float, prediction_label: str, prediction_classes: list):
    """
    Calculates the prediction probability of a class given the classifier's output.

    :param prediction: sigmoid output value from the classifier (binary classifier).
    :param prediction_label: classifier prediction label (e.g. 'Real' or 'Fake').
    :param prediction_classes: list of possible classes to predict.
    :return: dictionary with the calculated probability for each class.
    """

    prediction_classes.remove(prediction_label)
    prediction_prob = max(prediction, 1 - prediction)

    probabilities = {
        prediction_label: round(prediction_prob * 100),
        prediction_classes[0]: round((1 - prediction_prob) * 100)
    }

    return probabilities
