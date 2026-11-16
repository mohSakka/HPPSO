"""Resume frozen GA replay, retrying transient Windows result-file locks."""
from experiments import cec2011_ga_fitness_reuse as replay
from experiments.final_benchmark import write_result as original_write
import time


def safe_write(path,data):
    for attempt in range(200):
        try:return original_write(path,data)
        except PermissionError:
            if attempt==199:raise
            time.sleep(.05)


replay.write_result=safe_write
replay.base.write_result=safe_write


def main():
    replay.main()


if __name__=='__main__':main()
