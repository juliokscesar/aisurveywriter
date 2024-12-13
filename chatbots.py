from abc import ABC, abstractmethod
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep

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

