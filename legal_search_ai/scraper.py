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
import os
import urllib
import pandas as pd
from bs4 import BeautifulSoup
import shutil
import tempfile
import urllib.request
from selenium.webdriver.common.action_chains import ActionChains
from urlretrieve import *


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

# Pause the script for 15 seconds
time.sleep(15)

# Wait for a specific element to appear on the new page
try:
    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="example_pdf_length"]/label/select'))
    )
    # We are on a new page, continue with the scraping process
    # ...
except:
    # We are still on the same page, something went wrong
    print('Error: Unable to navigate to the next page after solving the CAPTCHA')



# Find the dropdown element and create a Select object
dropdown_element = driver.find_element(By.XPATH, '//*[@id="example_pdf_length"]/label/select')
dropdown = Select(dropdown_element)

# Select the option by value
dropdown.select_by_value('25')

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

df = pd.DataFrame(cases)

df = df.dropna(how='all')

for i in range(25):
    # Wait for a specific element to appear on the new page
    try:
        element = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.ID, f"link_{i}"))
        )
        # We found url of case
    except:
        # We didn't found url, something went wrong
        print('Error: Unable to find url of case')

    # check if there are any other elements blocking the element to click on
    blocking_element = driver.find_element(By.ID, 'modal_close')
    if blocking_element.is_displayed():
        # move the mouse to the blocking element to interact with it and remove it
        action = ActionChains(driver)
        action.move_to_element(blocking_element).click().perform()

    # Create ActionChains object
    actions = ActionChains(driver)

    # wait for element to become clickable
    open_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, f"link_{i}")))

    try:
        # Move to the element and perform click action using ActionChains
        actions.move_to_element(open_button).click().perform()
    except:
        driver.execute_script("arguments[0].scrollIntoView();", open_button)
        driver.execute_script("window.scrollBy(0, 1000);")
        # click on element
        open_button.click()

        # Click on the button to open the pop-up
        # open_button = driver.find_element(By.ID, f"link_{i}")
        # open_button.click()

    time.sleep(5)

    object_element = driver.find_element(By.TAG_NAME, "object")
    data_value = object_element.get_attribute("data")

    if "https://judgments.ecourts.gov.in" not in data_value:
        pdf_url = urljoin("https://judgments.ecourts.gov.in", data_value)
    else:
        pdf_url = data_value
    filename = os.path.basename(data_value)

    directory_path = "data/case_docs/IT_ACT_2000"
    pdf_path = os.path.join(directory_path, filename)

    # Click on the button to open the pop-up
    close_button = driver.find_element(By.ID, "modal_close")
    close_button.click()
    
    # Download the PDF
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    urlretrieve(pdf_url, pdf_path)

    df.loc[i, "pdf"] = filename

    time.sleep(5)


df.to_csv("data/case_docs/case_pdfs.csv")


# Close the window and quit the driver
driver.close()
driver.quit()


