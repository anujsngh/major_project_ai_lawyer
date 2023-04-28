import string
import re
from unicodedata import normalize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import PyPDF2
from data import *
import os
import logging


def get_section_details(section_num):
    # Set the path to the folder containing the text files
    folder_path = 'data/statute_docs/IT_ACT_2000'

    # Get a list of all the files in the folder
    file_list = os.listdir(folder_path)

    # Filter the list to include only text files
    text_files = [file for file in file_list if file.endswith('.txt')]
    
    for file_name in text_files:
        pattern = r"C\d+_S(\d+[A-Z]*).txt"
        match = re.search(pattern, file_name)
        if match:
            file_section_num = match.group(1)
            
            if (file_section_num == section_num):
                file_path = os.path.join(folder_path, file_name)
                
                # Open the file for reading
                with open(file_path, 'r') as file:

                    # Read all the text from the file
                    file_text = file.read()
                logging.debug(file_text)
                return file_text



def extract_sections(query):
    # Define regex patterns for each format
    pattern_00 = r'(\w+)'
    pattern_0 = r'Section\s+(\w+)'
    pattern_1 = r'Section\s+(\w+)\s+of\s+the\s+\w+'
    pattern_2 = r'(\w+)\s+and\s+(\w+)\s+of\s+the\s+\w+'
    pattern_3 = r'(\w+),\s*(\w+)\s+and\s+(\w+)\s+of\s+the\s+\w+'
    pattern_4 = r'(\w+)\s+to\s+(\w+)\s+of\s+the\s+\w+'

    # Define a list to hold the section numbers from each input string
    section_numbers = []

    # Extract section numbers from input string 1 using pattern 1
    match = re.search(pattern_1, query)
    if match:
        section_numbers.append(match.group(1))

    # Extract section numbers from input string 2 using pattern 2
    match = re.search(pattern_2, query)
    if match:
        section_numbers.append(match.group(1))
        section_numbers.append(match.group(2))

    # Extract section numbers from input string 3 using pattern 3
    match = re.search(pattern_3, query)
    if match:
        section_numbers.append(match.group(1))
        section_numbers.append(match.group(2))
        section_numbers.append(match.group(3))

    # Extract section numbers from input string 4 using pattern 4
    match = re.search(pattern_4, query)
    if match:
        start_section = match.group(1)
        end_section = match.group(2)
        
        # Add all section numbers between start_section and end_section (inclusive)
        section_list = list(section_order_dict.keys())
        start_index = section_list.index(start_section)
        end_index = section_list.index(end_section)
        section_numbers += section_list[start_index:end_index+1]

    if len(section_numbers) == 0:
        # Extract section numbers from input string 1 using pattern 00
        match = re.search(pattern_00, query)
        if match:
            section_numbers.append(match.group(1))

        # Extract section numbers from input string 1 using pattern 0
        match = re.search(pattern_0, query)
        if match:
            section_numbers.append(match.group(1))

    # Return the section numbers extracted from each input string
    logging.debug(section_numbers)
    return list(set(section_numbers))


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



if __name__ == "__main__":
    print(get_section_details("23"))