import pandas as pd
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from time import sleep
import undetected_chromedriver as uc
from selenium.webdriver.support.ui import WebDriverWait
from fake_useragent import UserAgent
from selenium.webdriver.support import expected_conditions as EC
import os


prompt = "Write a poem about physics"
MAIL = os.environ["NBLM_EMAIL"]
PASSWORD = os.environ["NBLM_PASS"]
# print(prompt)
PATH = "chromedriver-linux64/chromedriver"
# BROWSER = "C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe"

op = uc.ChromeOptions()
op.add_argument(f"user-agent={UserAgent.random}")
op.add_argument("user-data-dir=./")
op.add_experimental_option("detach", True)
op.add_experimental_option("excludeSwitches", ["enable-logging"])

driver = uc.Chrome(
        chrome_options=op, 
        #browser_executable_path=BROWSER, 
        driver_executable_path=PATH
)
driver.get("https://notebooklm.google/")

sleep(3)
orig_window = driver.current_window_handle

# Click on 'Try NotebookLM'
elem = driver.find_element(By.LINK_TEXT, "Try NotebookLM")
elem.click()
sleep(5)

# clicking on it opens second tab to login with google
wait = WebDriverWait(driver, 10)
wait.until(EC.number_of_windows_to_be(2))

for window_handle in driver.window_handles:
    if window_handle != orig_window:
        driver.switch_to.window(window_handle)
        break
sleep(5)
print(driver.current_url)

# now on login page
# email page
elem = driver.find_element(By.XPATH, "//input[@type='email']")
elem.send_keys(MAIL)
elem.send_keys(Keys.ENTER)

# password page
elem = driver.find_element(By.XPATH, "//input[@type='password']")
elem.send_keys(PASSWORD)
elem.send_keys(Keys.ENTER)
sleep(5)

# wait for user login confirmation

