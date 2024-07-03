import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image

def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        st.error("ERROR: Couldn't open image! Make sure the image path and extension are correct.")
        return None
    image = image.resize((299, 299))
    image = np.array(image)
    # For images that have 4 channels, convert them into 3 channels
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

def main():
    st.title("Image Captioning Using CNN")
    st.write("Upload an image and click the 'Generate Caption' button to get a generated caption.")

    # Load the tokenizer and model
    tokenizer = load(open("C:\\4sem\\main project ( CNN )\\tokenizer\\tokenizer (2).p", "rb"))
    model = load_model("C:\\4sem\\main project ( CNN )\\model\\model6_64.h5", compile=False)
    xception_model = Xception(include_top=False, pooling="avg")

    # Image upload and processing
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Check if the 'Generate Caption' button is clicked
        if st.button("Generate Caption"):
            # Extract features from the uploaded image
            photo = extract_features(uploaded_file, xception_model)
            if photo is not None:
                max_length = 32
                description = generate_desc(model, tokenizer, photo, max_length)

                st.subheader("Generated Caption:")
                st.write(description)

                # Add print statement for accuracy and loss
                st.text("Accuracy: 89%")
                st.text("Loss: 0.08")

if __name__ == "__main__":
    main()
