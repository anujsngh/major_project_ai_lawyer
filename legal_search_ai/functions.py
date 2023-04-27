import string
import re
from unicodedata import normalize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import PyPDF2


def preprocess_text(text):
    # Normalization
    text = normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # Remove special characters
    text = re.sub(r'[^0-9a-zA-Z]', ' ', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Convert to lowercase
    tokens = [token.lower() for token in tokens]

    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into text
    preprocessed_text = " ".join(tokens)

    return preprocessed_text


def load_pdf(pdf_file_path, password=None):
    # Open the PDF file in read-binary mode
    with open(pdf_file_path, 'rb') as pdf_file:
    
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Check if the PDF is encrypted
        if pdf_reader.is_encrypted:
            # Decrypt the PDF with the password 'password'
            pdf_reader.decrypt(password)
            
        # Get the number of pages in the PDF file
        num_pages = len(pdf_reader.pages)

        pdf_text = ""
        # Loop through each page in the PDF file
        for page_num in range(num_pages):
            
            # Get the page object for the current page
            page = pdf_reader.pages[page_num]

            # Extract the text from the page
            page_text = page.extract_text()

            # Print the text from the page
            pdf_text += page_text

        return pdf_text
