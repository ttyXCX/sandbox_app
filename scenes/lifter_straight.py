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
            print("detected lifter - raise lifter")
            while is_lifter_exist():
                lifter_up()
                scan_lifter()
            # 升降杆升起后经过三秒落下，需快速通过
            print("lifter up - moving forward")
            move_forward()
            move_forward()
            move_forward()
            print("passed lifter - down lifter")
            for _ in range(100):
                lifter_down()

    print("destination reached - done!")


if __name__ == "__main__":
    run()