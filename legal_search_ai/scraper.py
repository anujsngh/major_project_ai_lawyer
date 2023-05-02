# selenium 4
# selenium 4
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from urllib.parse import urljoin
import time
import sys
import logging
import os
import urllib
import pandas as pd
from bs4 import BeautifulSoup
import shutil
import tempfile
import urllib.request
from selenium.webdriver.common.action_chains import ActionChains
from urlretrieve import *
from data import cwd
from playsound import playsound
from plyer import notification


data_dir_path = os.path.join(cwd, "data/case_docs/IT_ACT_2000").replace("/", "\\")
if not os.path.exists(data_dir_path):
    os.makedirs(data_dir_path)


def select_25_cases(driver):
    # Wait for a specific element to appear on the new page
    try:
        dropdown_element = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="example_pdf_length"]/label/select'))
        )
        # We are on a new page, continue with the scraping process
        # ...
    except:
        # We are still on the same page, something went wrong
        print('Error: Unable to navigate to the next page after solving the CAPTCHA')

    # Find the dropdown element and create a Select object
    dropdown = Select(dropdown_element)
    # Select the option by value
    dropdown.select_by_value('25')


def scrap_cases():

    # options = webdriver.ChromeOptions()
    # options.add_argument('--headless')

    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

    # Set a page load timeout of 10 seconds
    driver.set_page_load_timeout(10)

    # Navigate to a webpage
    driver.get("https://judgments.ecourts.gov.in/pdfsearch/")

    # Find the dropdown element and create a Select object
    dropdown_element = driver.find_element(By.ID, 'fcourt_type')
    dropdown = Select(dropdown_element)

    # Select the option by value
    dropdown.select_by_value('2')

    search_elem = driver.find_element(By.ID, 'search_text')
    search_elem.send_keys("The Information Technology Act")

    # # Get the handle of the current window
    # window_handle = driver.current_window_handle
    # # Switch to the window by handle and bring it to front
    # driver.switch_to.window(window_handle)
    # driver.execute_script("window.focus();")

    flag = 1
    s_t = time.time()
    while True:
        # find the input box by id
        input_field = driver.find_element(By.ID, "captcha")
        if flag:
            # Move the cursor to the input field
            # input_field.click()
            input_field.send_keys(Keys.HOME)
            # driver.execute_script("arguments[0].click();", input_field)
            flag = 0
        # get the value of the input box
        value = input_field.get_attribute("value")
        if len(value) == 6:
            # input box is filled
            submit_btn = driver.find_element(By.ID, 'main_search')
            driver.execute_script("arguments[0].click();", submit_btn)
            break
        else:
            # input box is empty
            print("Input box is empty")
            time.sleep(2)
        
        e_t = time.time() - s_t
        if (e_t >= 10) and (e_t%5==0):
            notification.notify(title='Critical Alert!!!', message='Fill Captcha to Resume Scraping')
            playsound('data/alert.wav')

    # # Switch to the window by handle and send it to the background
    # driver.switch_to.window(window_handle)
    # driver.execute_script("window.blur();")
    
    # Pause the script for 10 seconds
    time.sleep(5)

    # define the highest index
    max_index = - 1
    try:
        org_df = pd.read_csv(os.path.join(data_dir_path, "case_pdfs.csv"))
        if not org_df.empty:
            # get the highest index
            max_index = org_df.index.max()
    except:
        print("DF is not locatable.")
    num_of_rows = max_index + 1
    pages_to_traverse = num_of_rows // 25
    start_index = num_of_rows%25
    end_index = 25

    for _ in range(pages_to_traverse):
        select_25_cases(driver)
        # Find the btn element and click to go to next page
        next_btn = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.ID, 'example_pdf_next'))
        )
        driver.execute_script("arguments[0].click();", next_btn)
        time.sleep(5)

    select_25_cases(driver)

    time.sleep(10)

    table = driver.find_element(By.ID, "example_pdf")
    table_html = table.get_attribute('outerHTML')

    soup = BeautifulSoup(table_html, features="lxml")
    table = soup.find("table")

    rows = table.find_all("tr")

    cases = []
    for row in rows:
        cells = row.find_all("td")
        case_data = {}
        for cell in cells[1:]:
            case_data["case_num"], case_data["case_title"] = cell.find("button").text.split(" of ", maxsplit=1)
            case_data["judges"] = cell.find("strong").text.split(": ", maxsplit=1)[1]
            case_data["cnr_num"], case_data["date_of_register"], case_data["date_of_decision"], case_data["disposal_nature"] = cell.find("strong", "caseDetailsTD").text.split(" | ")
            case_data["cnr_num"] = case_data["cnr_num"].split(": ")[1]
            case_data["date_of_register"] = case_data["date_of_register"].split(": ")[1]
            case_data["date_of_decision"] = case_data["date_of_decision"].split(": ")[1]
            case_data["disposal_nature"], case_data["court"] = case_data["disposal_nature"].split("Court : ")
            case_data["disposal_nature"] = case_data["disposal_nature"].split(": ")[1]
        cases.append(case_data)

    # df = pd.read_html(table_html)[0]

    # append the second DataFrame to the first one
    df = pd.DataFrame(cases)

    df = df.dropna(axis=0, how='all')

    df = df.reset_index(drop=True)

    for i in range(start_index, end_index):
        # Wait for a specific element to appear on the new page
        try:
            open_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, f"link_{i}"))
            )
            # We found url of case
        except:
            # We didn't found url, something went wrong
            print('Error: Unable to find url of case')

        try:
            # find the element we want to click on
            open_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, f"link_{i}")))

            # scroll the page to bring the element into view
            driver.execute_script("arguments[0].scrollIntoView();", open_button)

            # click on the element
            driver.execute_script("arguments[0].click();", open_button)
        except:
            print('Error: Button is not clickable!')

        # # Click on the button to open the pop-up
        # open_button.click()

        time.sleep(5)

        object_element = driver.find_element(By.TAG_NAME, "object")
        data_value = object_element.get_attribute("data")

        if "https://judgments.ecourts.gov.in" not in data_value:
            pdf_url = urljoin("https://judgments.ecourts.gov.in", data_value)
        else:
            pdf_url = data_value
        filename = os.path.basename(data_value)

        pdf_path = os.path.join(data_dir_path, filename)

        # Download the PDF
        
        urlretrieve(pdf_url, pdf_path)

        df.loc[i, "pdf"] = filename

        # Click on the button to open the pop-up
        close_button = driver.find_element(By.ID, "modal_close")
        driver.execute_script("arguments[0].click();", close_button)

        # df.to_csv(os.path.join("/data/temp", "case_pdfs.csv"))

    try:
        final_df = pd.concat([org_df, df], ignore_index=True)
        # final_df.dropna(axis=1, how="any")
        final_df.to_csv(os.path.join(data_dir_path, "case_pdfs.csv"))
    except NameError:
        df.to_csv(os.path.join(data_dir_path, "case_pdfs.csv"))
    except Exception as e:
        print(e)
    

    # Close the window and quit the driver
    driver.close()
    driver.quit()

def remove_extra_pdf_files():
    # specify the file path of the PDF file to be deleted
    try:
        df = pd.read_csv(os.path.join(data_dir_path, "case_pdfs.csv"))
    except:
        return
    df_pdf_files = set(df["pdf"].astype(str))
    # print(df_pdf_files)
    dir_pdf_files = [f for f in os.listdir(data_dir_path) if f.endswith('.pdf')]
    # print(dir_pdf_files)
    count = 0
    for dir_pdf_file in dir_pdf_files:
        if dir_pdf_file not in df_pdf_files:
            file_path = os.path.join(data_dir_path, dir_pdf_file)
            # delete the file
            os.remove(file_path)
            count += 1
    print(f"{count} extra pdf files removed.")


if __name__ == "__main__":
    while True:
        remove_extra_pdf_files()
        # # implement a temp storage using a temp/case_pdfs.csv file and add it's content in case of any failure.
        try:
            scrap_cases()
        except:
            logging.debug("Error:", sys.exc_info()[0])
