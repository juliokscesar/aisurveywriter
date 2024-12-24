import pandas as pd
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from time import sleep
import undetected_chromedriver as uc
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from fake_useragent import UserAgent
from selenium.webdriver.support import expected_conditions as EC
import os

MAIL = os.environ["NBLM_EMAIL"]
PASSWORD = os.environ["NBLM_PASS"]

DRIVER_PATH = "drivers/chromedriver"
# BROWSER = "C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe"

op = uc.ChromeOptions()
op.add_argument(f"user-agent={UserAgent.random}")
op.add_argument("user-data-dir=./")
op.add_experimental_option("detach", True)
op.add_experimental_option("excludeSwitches", ["enable-logging"])

driver = uc.Chrome(
        chrome_options=op, 
        #browser_executable_path=BROWSER, 
        driver_executable_path=DRIVER_PATH
)
print("Accessing NotebookLM main page")
driver.get("https://notebooklm.google/")

sleep(3)
orig_window = driver.current_window_handle

# Click on 'Try NotebookLM'
print("Clicking on 'Try NotebookLM'")
elem = driver.find_element(By.LINK_TEXT, "Try NotebookLM")
elem.click()
sleep(8)

# clicking on it opens second tab to login with google
wait = WebDriverWait(driver, 10)
wait.until(EC.number_of_windows_to_be(2))

print("Changing to login tab")
for window_handle in driver.window_handles:
    if window_handle != orig_window:
        driver.switch_to.window(window_handle)
        break
sleep(5)

# now on login page
# email page
print("Entering email")
elem = driver.find_element(By.XPATH, "//input[@type='email']")
elem.send_keys(MAIL)
elem.send_keys(Keys.ENTER)
sleep(5)

# password page
print("Entering password")
elem = driver.find_element(By.XPATH, "//input[@type='password']")
elem.send_keys(PASSWORD)
elem.send_keys(Keys.ENTER)
sleep(5)

# wait for user login confirmation
if "/challenge/" in driver.current_url:
    print("_"*80)
    print("WAITING FOR USER ACCOUNT 2F AUTHENTICATION. ENTER Y/y/yes TO CONTINUE")
    p = input("> ")
    while 'y' not in p.lower().strip():
        p = input("> ")


# Now on NotebookLM projects page
# create new one
print("Creating new notebook")
elem = driver.find_element(By.XPATH, "//button[contains(@class,'create-new-button')]")
elem.click()
sleep(10)

# first need to hover on input button so that the input element appears
elem = driver.find_element(By.XPATH, "//div[contains(@class, 'dropzone') and contains(@class, 'dropzone-3panel') and contains(@class, 'ng-star-inserted')]//button[contains(@class, 'mat-mdc-icon-button') and contains(@class, 'dropzone-icon-3panel')]")
hover = ActionChains(driver).move_to_element(elem)
hover.perform()
sleep(5)

# send pdf files
pdf_paths = [
    "refexamples/ArigaK2023_Langmuir.pdf",
    "refexamples/FangC_ApplicationsLangmuir.pdf",
    "refexamples/ArigaK2022_PastAndFutureLangmuir.pdf",
    "refexamples/LuC2024_AIScientist.pdf",
]
print("Sending PDFs:", ", ".join(pdf_paths))
elem = driver.find_elements(By.XPATH, "//div[contains(@class, 'dropzone') and contains(@class, 'dropzone-3panel') and contains(@class, 'ng-star-inserted')]//input[@type='file' and @name='Filedata']")
print(elem)
elem[0].send_keys("\n".join([os.path.join(os.getcwd(), pdf) for pdf in pdf_paths]))
sleep(40)
