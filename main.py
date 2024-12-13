from typing import Union
import os
import chatbots as cb
import undetected_chromedriver as uc
from fake_useragent import UserAgent

import yaml

def read_yaml(path: str):
    content = {}
    with open(path, "r") as f:
        content = yaml.safe_load(f)
    return content


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

def direct_chat(bot: cb.ChatGPTBot):
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


def main():
    DRIVER_PATH = "drivers/chromedriver"
    driver = init_driver(driver_path=DRIVER_PATH)

    EMAIL = os.environ["GPTMAIL"]
    PASS = os.environ["GPTPASS"]

    # Bot setup
    bot = cb.ChatGPTBot(EMAIL, PASS, driver)
    if not bot.login():
        print("Couldn't log in")
        return

    # Tell bot to only give responses in latex format
    bot.send_prompt("For now on, every section I ask you to write, you must provide the answer in LaTeX and in a code block. There's no need to include the preamble of the document. Only include the '\\section...' on forward. Remember that if you want to include figures, draw them with the 'tickz' package if possible.")
    bot.wait(10)

    sections = read_yaml("paper_instructions.yaml")
    base_fmt = sections["base_prompt_format"]
    subject = sections["subject"]

    dois = sections["ref_DOIS"]
    bot.send_prompt("This is the list of article DOIs which you must access, read and store all the information about them: {', '.join(dois)}")
    bot.wait(15)

    for section in sections["sections"]:
        struct = "The required structure for the section {section['title']} is:\n{section['description']}"
        prompt = base_fmt.format(section["title"], subject, section["description"])
        bot.send_prompt(prompt)
        bot.wait(40)


if __name__ == "__main__":
    main()

