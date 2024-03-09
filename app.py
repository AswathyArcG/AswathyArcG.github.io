# import streamlit as st

# st.title('Personal Identification Information Masking')

# st.title('Input')
# st.subheader('Enter text')

# st.title('Output')
# st.subheader('De-identified')

import streamlit as st
from transformers import AutoTokenizer, AutoModelForMaskedLM

def mask_text(input_text):
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    masked_text = input_text.replace("[MASK]", tokenizer.mask_token)
    inputs = tokenizer(masked_text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    masked_index = masked_text.split().index(tokenizer.mask_token)
    probabilities = logits[0, masked_index].softmax(dim=0)
    predicted_token = tokenizer.convert_ids_to_tokens(
        torch.argmax(probabilities).item()
    )
    return predicted_token

def main():
    st.title("Text Masking App")

    # Input Textbox
    st.subheader("Input Text")
    input_text = st.text_area("Enter text (max 100 words)", max_chars=500)

    # Mask Button
    if st.button("Mask"):
        if input_text:
            # Mask the text
            masked_output = mask_text(input_text)

            # Output Textbox
            st.subheader("Output Text")
            st.text_area("Masked Output", masked_output, key="output_text")

if __name__ == "__main__":
    main()
