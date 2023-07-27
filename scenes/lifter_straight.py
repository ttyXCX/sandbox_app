from dsl.actions import move_forward, wait, lifter_up, lifter_down
from dsl.perceptions import scan_lifter, is_destination_reached, is_lifter_exist


def run():
    print("start running")

    while not is_destination_reached():
        print("scan_lifter")
        scan_lifter()

        if not is_lifter_exist():
            print("moving forward - no lifter")
            move_forward()
        else:
            lifter_up()
            # 升降杆升起后经过三秒落下，需快速通过
            move_forward()
            move_forward()
            move_forward()

    print("destination reached - done!")


if __name__ == "__main__":
    run()