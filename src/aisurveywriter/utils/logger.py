from time import sleep

def named_log(obj, *msgs):
    print(f"({obj.__class__.__name__})", *msgs)

def countdown_log(msg: str, sec: int):
    print()
    for t in range(1, sec+1):
        sleep(1)
        print(f"\r{msg} {t} s", end='')
    print()