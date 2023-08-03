from __future__ import print_function

from utils.nodelet import init
print("initializing ros node...", end="")
init()
print("Done!")

# from scenes.pass_light_straight import run
# from scenes.lifter_straight import run
from scenes.cruise_simple import run

if __name__ == "__main__":
    run()