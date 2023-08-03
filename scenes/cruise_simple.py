from dsl.actions import move_forward, turn_left, turn_right
from dsl.perceptions import is_destination_reached, lane_assist
from utils.lane_detect import LEFT_TOKEN, RIGHT_TOKEN

MAX_ADJUST = 5

def run():
    print("start running")

    while not is_destination_reached():

        for i in range(MAX_ADJUST):
            lane = lane_assist()
            print("lane assist: {}".format(lane))

            if lane == LEFT_TOKEN:
                print("turn left")
                turn_left()
            elif lane == RIGHT_TOKEN:
                print("turn right")
                turn_right()
            else:
                break
        
        print("move forward")
        move_forward()

    print("destination reached - done!")


if __name__ == "__main__":
    run()
