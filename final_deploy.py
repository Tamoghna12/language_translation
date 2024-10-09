import os
import streamlit as st
from pdf2image import convert_from_path
import easyocr
from google.cloud import translate_v2 as translate
import numpy as np
import torch
import tempfile
import gc
from langdetect import detect, DetectorFactory

# Set a seed for reproducibility in language detection
DetectorFactory.seed = 0

# Set page configuration
st.set_page_config(page_title="Bulk OCR & Translation App", layout="wide")

# Custom CSS for light/dark mode toggle


def apply_custom_css(dark_mode=False):
    if dark_mode:
        st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
            background-color: #333;
            color: white;
        }
        .output-box {
            border: 1px solid #555;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .output-box {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

# Function to set GPU memory strategy


def set_gpu_memory_strategy():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.7)

# Function to process each page


def process_page(page, reader):
    try:
        img_array = np.array(page.convert('RGB'))
        result = reader.readtext(img_array, detail=0)
        extracted_text = " ".join(result)
        return extracted_text
    except Exception as e:
        st.error(f"Error processing page: {str(e)}")
        return None


def translate_text(text, target_language, api_key):
    client = translate.Client(
        target_language=target_language, credentials=None)
    translated_text = client.translate(text, target_language)
    return translated_text['translatedText']


def main():
    # Dark mode toggle
    dark_mode = st.sidebar.checkbox("Enable Dark Mode")
    apply_custom_css(dark_mode)

    st.title("Bulk OCR & Translation App")

    # API Key input
    st.sidebar.subheader("Google Cloud API Key")
    api_key = st.sidebar.text_input(
        "Enter your Google Cloud API Key", type="password")

    # Instructions for creating the API key
    st.sidebar.subheader("Instructions for API Key Creation")
    st.sidebar.markdown("""
    1. Visit [Google Cloud Console](https://console.cloud.google.com/).
    2. Navigate to **APIs & Services** > **Credentials**.
    3. Click **Create Credentials** and select **API Key**.
    4. Enable **Cloud Translation API** for your project.
    5. Copy the API key and paste it in the sidebar input field.
    """)

    if not api_key:
        st.warning(
            "Please enter your Google Cloud API key to enable translation.")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    # Destination language options
    translation_options = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Japanese": "ja",
        "Korean": "ko",
        "Chinese (Simplified)": "zh-CN",
        "Hindi": "hi",
        "Bengali": "bn",
        "Marathi": "mr",
        "Tamil": "ta",
        "Telugu": "te",
        "Gujarati": "gu",
        "Malayalam": "ml",
        "Punjabi": "pa",
        "Urdu": "ur"
    }

    target_language = st.selectbox(
        "Select Target Language", list(translation_options.keys()))

    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.read())
            temp_pdf_path = temp_pdf.name

        # Load pages from the temporary PDF file
        pages = convert_from_path(temp_pdf_path, 100)

        # Initialize EasyOCR reader
        use_gpu = torch.cuda.is_available()
        reader = easyocr.Reader(['en'], gpu=use_gpu, quantize=True)

        all_extracted_text = []
        all_translated_text = []

        # Process pages
        for page in pages:
            extracted_text = process_page(page, reader)
            if extracted_text:
                detected_language = detect(extracted_text)
                st.write(f"Detected Language: {detected_language}")

                # Translate text using Google Cloud API
                translated_text = translate_text(
                    extracted_text, translation_options[target_language], api_key)

                all_extracted_text.append(extracted_text)
                all_translated_text.append(translated_text)

        # Combine all translated text into a single output
        combined_output = "\n\n".join(f"Original Text:\n{extracted_text}\n\nTranslated Text:\n{translated_text}"
                                      for extracted_text, translated_text in zip(all_extracted_text, all_translated_text))

        # Display the output text
        st.markdown("### Translated Output")
        st.markdown(
            f'<div class="output-box">{combined_output}</div>', unsafe_allow_html=True)

        # Download button for the output file
        if combined_output:
            st.markdown("---")
            st.subheader("Download Translated Document")
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
            with open(tmp_file.name, 'w', encoding='utf-8') as f:
                f.write(combined_output)
            with open(tmp_file.name, "rb") as file:
                st.download_button(
                    label="Download Translated Document",
                    data=file,
                    file_name="translated_document.txt",
                    mime="text/plain"
                )
            os.unlink(tmp_file.name)  # Delete the temporary file

        gc.collect()
        if use_gpu:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
