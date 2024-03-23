# import streamlit as st
# st.write("Hello from Streamlit")

# import streamlit as st
# from transformers import AutoTokenizer, AutoModelForMaskedLM

# def mask_text(input_text):
#     model_name = "bert-base-uncased"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForMaskedLM.from_pretrained(model_name)

#     masked_text = input_text.replace("[MASK]", tokenizer.mask_token)
#     inputs = tokenizer(masked_text, return_tensors="pt")
#     outputs = model(**inputs)
#     logits = outputs.logits
#     masked_index = masked_text.split().index(tokenizer.mask_token)
#     probabilities = logits[0, masked_index].softmax(dim=0)
#     predicted_token = tokenizer.convert_ids_to_tokens(
#         torch.argmax(probabilities).item()
#     )
#     return predicted_token

# def main():
#     st.title("Text Masking App")

#     # Input Textbox
#     st.subheader("Input Text")
#     input_text = st.text_area("Enter text (max 100 words)", max_chars=500)

#     # Mask Button
#     if st.button("Mask"):
#         if input_text:
#             # Mask the text
#             masked_output = mask_text(input_text)

#             # Output Textbox
#             st.subheader("Output Text")
#             st.text_area("Masked Output", masked_output, key="output_text")

# if __name__ == "__main__":
#     main()

# app.py

#**********************3rd round***************************
# import streamlit as st
# import os
# import docx2txt  # For extracting text from Word documents
# import PyPDF2  # For extracting text from PDFs
# import pickle

# # Set a maximum word limit for the textarea
# MAX_WORD_LIMIT = 1000

# def extract_text_from_pdf(file_path):
#     with open(file_path, "rb") as f:
#         pdf_reader = PyPDF2.PdfReader(f)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def extract_text_from_docx(file_path):
#     return docx2txt.process(file_path)

# def main():
#     st.title("NLP Text Analyzer")

#     # Load your pre-trained model
#     model = pickle.load(open('tagged_sentence.pickle', 'rb'))
    
#     # Create a textarea with validation
#     user_input = st.text_area("Enter your text here:", max_chars=MAX_WORD_LIMIT)
#     if len(user_input.split()) > MAX_WORD_LIMIT:
#         st.warning(f"Word limit exceeded. Please keep it under {MAX_WORD_LIMIT} words.")

#     # Create an upload button with validation
#     uploaded_file = st.file_uploader("Upload a PDF or Word document", type=["pdf", "docx"])
#     if uploaded_file:
#         file_extension = os.path.splitext(uploaded_file.name)[1].lower()

#         if file_extension == ".pdf":
#             extracted_text = extract_text_from_pdf(uploaded_file)
#         elif file_extension == ".docx":
#             extracted_text = extract_text_from_docx(uploaded_file)
#         else:
#             st.error("Unsupported file format. Please upload a PDF or Word document.")
#             return

#         st.success(f"Text extracted from {uploaded_file.name}:")
#         st.write(extracted_text)

#     # You can add your NLP model integration here
#     # Create a button to make predictions
#     if st.button('Analyze'):
#         # Preprocess the user_input here (if necessary)
#         # For example, if your model expects a vectorized form of the text:
#         # preprocessed_input = vectorizer.transform([user_input])

#         # Make predictions
#         prediction = model.predict([user_input])

#         # Display predictions
#         st.write(f"Prediction: {prediction}")
    
#     # For demonstration purposes, let's just display the input text
#     st.subheader("Input Text:")
#     st.write(user_input)

# if __name__ == "__main__":
#     main()


#**********************3rd round***************************

# import streamlit as st
# import pickle
# from flair.data import Sentence
# # Assuming FlairRecognizer is defined in FlairRecognizer.py
# from Flair_train_and_test import FlairRecognizer


# def load_flair_recognizer_model(model_path):
#     with open(model_path, "rb") as f:
#         flair_recognizer = pickle.load(f)
#     return flair_recognizer

# def main():
#     st.title("Named Entity Recognition (NER) using Flair")

#     # Load the FlairRecognizer model
#     model_path = "flair_recognizer_model.pickle"  # Update the path accordingly
#     flair_recognizer = load_flair_recognizer_model(model_path)

#     # Text input for user
#     user_input = st.text_input("Enter your sentence:")

#     # Perform NER tagging when button is clicked
#     if st.button("Analyze"):
#         if user_input:
#             # Process user input with FlairRecognizer
#             sentence = Sentence(user_input)
#             results = flair_recognizer.analyze(sentence, entities=None, nlp_artifacts=None)

#             # Display NER tags
#             st.write("Named Entities:")
#             for result in results:
#                 st.write(f"Entity: {result.entity_type}, Value: {result.full_text}")
#         else:
#             st.warning("Please enter a sentence for analysis.")

# if __name__ == "__main__":
#     main()

#**********************4th round***************************

# import streamlit as st
# # from flair.models import SequenceTagger
# # from flair.data import Sentence
# from Flair_train_and_test import FlairRecognizer

# # import streamlit as st
# # # Import the model class or function from your Python file
# # from your_model_file import YourModelClassOrFunction

# def load_model():
#     # Load your model
#     # This could be a path to a model file if your model requires one
#     model = FlairRecognizer.load_model('path/to/your/model/file')
#     return model

# def main():
#     st.title("Your App Title")

#     # Load the model
#     model = load_model()

#     # Get user input
#     user_input = st.text_input("Enter your text here:")

#     if st.button("Analyze"):
#         if user_input:
#             # Use the model to make predictions
#             prediction = model.predict(user_input)
#             # Display the results
#             st.write(prediction)
#         else:
#             st.warning("Please enter some text to analyze.")

# if __name__ == "__main__":
#     main()


#**********************5th round***************************
    

import streamlit as st
from pathlib import Path
from Final_file import FlairRecognizer



# Cache the model loading and prediction function
@st.cache_resource
def cached_predict_ner_tags(text):
    return FlairRecognizer.predict_ner_tags(text)

# Cache the text analysis function
@st.cache_data
def cached_analyze_text(text):
    return FlairRecognizer.analyze_text(text)

def main():
    st.title('PII Masking App')
    st.sidebar.header('Upload Options')
    upload_option = st.sidebar.radio("Choose upload option:", ('Text Input', 'File Upload'))

    # Dropdown menu with four choices
    st.sidebar.header('Masking Options')
    choice = st.sidebar.selectbox('Choose your masking option:', ['Option 1', 'Option 2', 'Option 3', 'Option 4'])

    if upload_option == 'Text Input':
        input_text = st.text_area("Enter text here:")
        if st.button('Analyze'):
            with st.spinner('Wait for it... the model is loading'):
                cached_predict_ner_tags(input_text)
                masked_text = cached_analyze_text(input_text)
            st.text_area("Masked text:", value=masked_text, height=200)
    elif upload_option == 'File Upload':
        uploaded_file = st.file_uploader("Upload a file", type=['txt', 'pdf', 'docx'])
        if uploaded_file is not None:
            file_contents = uploaded_file.read()
            if st.button('Analyze'):
                with st.spinner('Wait for it... the model is loading'):
                    # Display the file contents
                    st.write("File contents:")
                    st.write(file_contents)
                    st.write(file_contents.decode())
                    cached_predict_ner_tags(file_contents.decode())
                    masked_text = cached_analyze_text(file_contents.decode())
                st.text_area("Masked text:", value=masked_text, height=200)

if __name__ == "__main__":
    main()
