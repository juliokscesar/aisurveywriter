from typing import Union
import os
import chatbots as cb
import undetected_chromedriver as uc
from fake_useragent import UserAgent

def init_driver(browser_path: Union[str,None] = None, driver_path: Union[str,None] = None) -> uc.Chrome:
    op = uc.ChromeOptions()
    op.add_argument(f"user-agent={UserAgent.random}")
    op.add_argument("user-data-dir=./")
    op.add_experimental_option("detach", True)
    op.add_experimental_option("excludeSwitches", ["enable-logging"])
    driver = uc.Chrome(
            chrome_options=op,
            browser_executable_path=browser_path,
            driver_executable_path=driver_path
    )
    return driver

def main():
    DRIVER_PATH = "drivers/chromedriver"
    driver = init_driver(driver_path=DRIVER_PATH)

    EMAIL = os.environ["GPTMAIL"]
    PASS = os.environ["GPTPASS"]

    bot = cb.ChatGPTBot(EMAIL, PASS, driver)
    if not bot.login():
        print("Deu ruim")
    
    bot.wait(5)
    
    print("_"*80)
    print("START SENDING PROMPTS NOW. SEND '!stop' TO STOP")
    print("_"*80)
    while True:
        prompt = input("> ")
        if "!stop" in prompt.lower().strip():
            break
        bot.send_prompt(prompt)
        bot.wait(10)
        print(bot.get_last_response(in_code_block=True))

if __name__ == "__main__":
    main()

