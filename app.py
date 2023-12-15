import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
import numpy as np
import pickle as pkl
from keras.models import load_model


html_string = '''
<!DOCTYPE html>
<html style="font-size: 16px;" lang="fr">
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta charset="utf-8">
    <meta name="keywords" content="INTUITIVE">
    <meta name="description" content="">
    <title>Accueil</title>
</head>
<body data-path-to-root="./" data-include-products="false" class="u-body u-xl-mode" data-lang="fr">
    <header class="u-clearfix u-gradient u-header" id="sec-c8c2">
        <div class="u-clearfix u-sheet u-sheet-1">
            <p class="u-text u-text-default u-text-1">AwatefBR</p>
            <h2 class="u-custom-font u-font-roboto u-text u-text-default u-text-2">Sentiment Analysis</h2>
        </div>
    </header>
    <section class="u-align-center u-clearfix u-white u-section-1" id="carousel_5e7e">
        <img class="u-expanded-width u-image u-image-default u-image-1" src="/static/images/gh4.jpg" alt="" data-image-width="1440" data-image-height="1080">
        <div class="u-form u-radius-20 u-white u-form-1">
            <form action="https://forms.nicepagesrv.com/v2/form/process" class="u-clearfix u-form-spacing-15 u-form-vertical u-inner-form" source="email" name="form" style="padding: 23px;">
                <h4 class="u-align-center u-form-group u-form-text u-text u-text-1">
                    &nbsp;​​Write your post!<span style="text-decoration: underline !important;"></span>
                </h4>
                <div class="u-form-group u-form-message">
                    <label for="message-4c18" class="u-label">Message</label>
                    <textarea placeholder="Enter your message" rows="4" cols="50" id="message-4c18" name="message" class="u-border-2 u-border-grey-10 u-grey-10 u-input u-input-rectangle u-radius-10" required=""></textarea>
                </div>
                <div class="u-align-right u-form-group u-form-submit">
                    <a href="#" class="u-active-custom-color-1 u-border-5 u-border-active-palette-3-base u-border-hover-palette-3-base u-border-palette-2-base u-btn u-btn-round u-btn-submit u-button-style u-custom-color-2 u-hover-palette-3-base u-radius-10 u-btn-1">Submit&nbsp;</a>
                    <input type="submit" value="submit" class="u-form-control-hidden">
                </div>
                <div class="u-form-send-message u-form-send-success"> Thank you! Your message has been sent. </div>
                <div class="u-form-send-error u-form-send-message"> Unable to send your message. Please fix errors then try again. </div>
                <input type="hidden" value="" name="recaptchaResponse">
                <input type="hidden" name="formServices" value="98bbb880-c729-86c6-b2c8-d2b61d96e8d7">
            </form>
        </div>
        <div class="u-border-2 u-border-grey-75 u-container-style u-gradient u-group u-radius u-shape-round u-group-1">
            <div class="u-container-layout u-valign-middle u-container-layout-1">
                <h2 class="u-align-center u-text u-text-2">{{ result }}</h2>
            </div>
        </div>
    </section>
    <footer class="u-align-center u-clearfix u-footer u-gradient u-footer" id="sec-49c9">
        <div class="u-clearfix u-sheet u-sheet-1"></div>
    </footer>
    <section class="u-backlink u-clearfix u-grey-80">
        <a class="u-link" href="https://nicepage.com/website-templates" target="_blank">
            <span>Website Templates</span>
        </a>
        <p class="u-text">
            <span>created with</span>
        </p>
        <a class="u-link" href="" target="_blank">
            <span>Website Builder Software</span>
        </a>.
    </section>
</body>
</html>
'''

components.html(html_string)

# User input for text (hidden in Streamlit app)
user_text = st.text_area("Write your post!", height=150)

if user_text:
    with open('tokenizer.pkl', 'rb') as file:
        tokenizer = pkl.load(file)
        
    max_sequence_length = 100
    user_sequences = tokenizer.texts_to_sequences([user_text])
    user_padded = tf.keras.preprocessing.sequence.pad_sequences(user_sequences, maxlen=max_sequence_length)

    loaded_model = load_model('LSTM_model.h5')

    class_mapping = {0: "Based on our analysis, it appears that your post leans towards a negative tone. Would you mind reconsidering your message to promote a more constructive and positive online environment?", 1: "Your message appears neutral. Feel free to proceed, or consider adding a touch of personal flair to make it more engaging if you'd like!", 2: "Thank you for spreading positivity! Your words contribute to a brighter online space. Keep up the great vibes!"}

    user_predictions = loaded_model.predict(user_padded)
    user_pred_class = np.argmax(user_predictions, axis=1)
    prediction = class_mapping[user_pred_class[0]]


    st.write(f'{prediction}')
else:
    st.warning("Submit")
