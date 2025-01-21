from abc import ABC, abstractmethod
from typing import List
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from time import sleep
import os

class ChatBot(ABC):
    def __init__(self, username: str, password: str, main_url: str, driver: uc.Chrome):
        self._username = username
        self._password = password
        self._main_url = main_url
        self._web_driver = driver

    @abstractmethod
    def login(self) -> bool:
        pass

    @abstractmethod
    def send_prompt(self, prompt: str):
        pass

    def wait(self, sec: int):
        sleep(sec)

CHATGPT_MAINURL = "https://chatgpt.com/"
class ChatGPTBot(ChatBot):
    def __init__(self, username: str, password: str, driver: uc.Chrome):
        super().__init__(username,password,CHATGPT_MAINURL,driver)
        self._prompt_element = None

    def login(self) -> bool:
        self._web_driver.get(self._main_url)
        sleep(5)

        # Click on login button
        elem = self._web_driver.find_element(By.XPATH, '//button[@data-testid="login-button"]')
        elem.click()
        sleep(5)

        # input email
        elem = self._web_driver.find_element(By.ID, "email-input")
        elem.send_keys(self._username)

        # click to continue to password page
        elem = self._web_driver.find_element(By.CLASS_NAME, "continue-btn")
        elem.click()
        sleep(6)

        # input password
        elem = self._web_driver.find_element(By.ID, "password")
        elem.send_keys(self._password)

        # Then click to continue
        wait = WebDriverWait(self._web_driver, 10)
        elem = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "_button-login-password")))
        elem.click()

        # it may ask for verification by the user, so wait for it
        if self._web_driver.current_url.find("login_challenge") != -1:
            print("_"*80)
            print("> CHATGPT ASKING FOR VERIFICATION CODE. SEND Y/y/yes HERE WHEN DONE")
            p = input("> ")
            while "y" not in p.strip().lower():
                p = input("> ")
        sleep(10)

        # if we logged in successfully, keep the prompt input element
        self._prompt_element = self._web_driver.find_element(By.ID, "prompt-textarea")
        return (self._prompt_element is not None)

    def is_logged_in(self):
        return (self._prompt_element is not None)

    def add_files(self, paths: List[str]):
        if self._prompt_element is None:
            print("CHATGPT BOT: not logged in (prompt element is None)")
            return
        
        elem = self._web_driver.find_element(By.XPATH, "//input[@type='file']")
        elem.send_keys("\n".join(
            [os.path.abspath(os.path.join(os.getcwd(), path)) for path in paths]
        ))
        print("Sent files to ChatGPT. Now sleeping for 20 seconds...")
        sleep(20)


    def send_prompt(self, prompt: str):
        if self._prompt_element is None:
            print("CHATGPT BOT: prompt element is None")
            return
        self._prompt_element.send_keys(prompt)
        self._prompt_element.send_keys(Keys.ENTER)

    def get_last_response(self, in_code_block=False) -> str:
        if in_code_block:
            response_elem = self._web_driver.find_elements(By.TAG_NAME, "code")[-1]
            return response_elem.text

        response_elem = self._web_driver.find_elements(By.XPATH, "//div[@data-message-author-role='assistant']")
        print(response_elem)
        return response_elem[-1].text


NBLM_MAIN_URL = "https://notebooklm.google/"
class NotebookLMBot(ChatBot):
    def __init__(self, user: str, password: str, driver: uc.Chrome, src_paths: List[str]):
        super().__init__(user, password, NBLM_MAIN_URL, driver)
        self._src_paths = src_paths.copy()
        self._prompt_element = None
    
    def login(self) -> bool:
        self._web_driver.get(self._main_url)
        sleep(3)

        # keep track of original tab
        orig_window = self._web_driver.current_window_handle

        # Click on "Try NotebookLM"
        print("Clicking on 'Try NotebookLM'")
        elem = self._web_driver.find_element(By.LINK_TEXT, "Try NotebookLM")
        elem.click()
        
        # wait for the login tab to open and switch
        WebDriverWait(self._web_driver, 10).until(EC.number_of_windows_to_be(2))
        print("Changing to login tab")
        for window_handle in self._web_driver.window_handles:
            if window_handle != orig_window:
                self._web_driver.switch_to.window(window_handle)
                break
        sleep(3)

        # Now on login page
        # Enter email
        print("Entering email")
        elem = self._web_driver.find_element(By.XPATH, "//input[@type='email']")
        elem.send_keys(self._username)
        elem.send_keys(Keys.ENTER)
        sleep(5)

        # Enter password
        print("Entering password")
        elem = self._web_driver.find_element(By.XPATH, "//input[@type='password']")
        elem.send_keys(self._password)
        elem.send_keys(Keys.ENTER)
        sleep(5)

        # wait for user authentication if detected
        if "/challenge/" in self._web_driver.current_url:
            print("*"*80)
            print("WAITING FOR USER ACCOUNT 2F AUTHENTICATION. ENTER [y]es TO CONTINUE")
            print("*"*80)
            p = input("> ")
            while 'y' not in p.lower().strip():
                p = input("> ")

        # Now on NotebookLM projects page
        print("Creating new notebook")
        elem = self._web_driver.find_element(By.XPATH, "//button[contains(@class,'create-new-button')]")
        elem.click()
        sleep(7)

        self.add_sources(src_paths=self._src_paths, sleep_for=40)

        # get input textarea
        self._prompt_element = self._web_driver.find_element(By.XPATH, "//textarea[contains(@class, 'query-box-input')]")
        return (self._prompt_element is not None)

    def is_logged_in(self):
        return (self._prompt_element is not None)

    def add_sources(self, src_paths: List[str], sleep_for: int = 40):
        # first need to hover on input button so that the input element appears
        elem = self._web_driver.find_element(By.XPATH, "//div[contains(@class, 'dropzone') and contains(@class, 'dropzone-3panel') and contains(@class, 'ng-star-inserted')]//button[contains(@class, 'mat-mdc-icon-button') and contains(@class, 'dropzone-icon-3panel')]")
        hover = ActionChains(self._web_driver).move_to_element(elem)
        hover.perform()
        sleep(5)

        # Send sources
        print("Sending sources:", ", ".join(src_paths))
        elem = self._web_driver.find_elements(By.XPATH, "//div[contains(@class, 'dropzone') and contains(@class, 'dropzone-3panel') and contains(@class, 'ng-star-inserted')]//input[@type='file' and @name='Filedata']")
        elem[0].send_keys("\n".join(
            [os.path.abspath(os.path.join(os.getcwd(), src)) for src in src_paths]
        ))
        print(f"Sleeping for {sleep_for} seconds to wait for sources to load...")
        sleep(sleep_for)
        self._src_paths.extend(src_paths)

    def send_prompt(self, prompt: str, sleep_for: int = 30):
        if self._prompt_element is None:
            print("NotebookLMBot prompt element is None")
            return
        self._prompt_element = self._web_driver.find_element(By.XPATH, "//textarea[contains(@class, 'query-box-input')]")

        # since we probabily have newline characters, it's better to enter block by block
        for block in prompt.split("\n"):
            self._prompt_element.send_keys(block)
            ActionChains(self._web_driver).key_down(Keys.SHIFT).key_down(Keys.ENTER).key_up(Keys.SHIFT).key_up(Keys.ENTER).perform()
        self._prompt_element.send_keys(Keys.ENTER)

        if sleep_for:
            print(f"Sleeping for {sleep_for} seconds waiting for response...")
            sleep(sleep_for)

    def get_last_response(self) -> str:
        elem = self._web_driver.find_elements(By.TAG_NAME, "chat-message")[-1]
        if elem is None:
            return ""
        response = elem.get_property("outerText")

        # response comes with some text from other elements, so just cut it
        response = response[:response.find("\nkeep_pin")]
        return response
    
    def append_sources(self, src_paths: List[str], sleep_for: int = 30):
        elem = self._web_driver.find_element(By.XPATH, "//button[contains(@class, 'add-source-button')]")
        elem.click()
        sleep(4)

        self.add_sources(src_paths, sleep_for)
