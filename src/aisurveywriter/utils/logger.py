from time import sleep

from langchain_core.messages import AIMessage

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def global_log(*msgs):
    logging.info(*msgs)

def named_log(obj, *msgs):
    logging.info(f"({obj.__class__.__name__})", *msgs)

def countdown_log(msg: str, sec: int):
    for t in range(1, sec+1):
        sleep(1)
        print(f"\r{msg} {t} s", end='')
    print()

def metadata_log(obj, time_elapsed: int, airesponse: AIMessage):
    named_log(obj, f"time elapsed: {time_elapsed} s | usage metadata: {airesponse.usage_metadata}")

def cooldown_log(obj, cooldown: int):
    countdown_log(f"({obj.__class__.__name__}) cooldown (request limitations):", cooldown)