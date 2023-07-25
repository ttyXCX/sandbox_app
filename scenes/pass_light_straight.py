from dsl.actions import move_forward, wait
from dsl.perceptions import scan, is_destination_reached, is_traffic_light_exist, is_traffic_light_green, is_distance_safe


def run():
    print("start running")

    while not is_destination_reached():
        print("scan")
        scan()

        if not is_traffic_light_exist():
            print("moving forward - no light")
            move_forward()
        else:
            if not is_traffic_light_green() and not is_distance_safe():
                print("waiting - green={}, safe={}".format(is_traffic_light_green(), is_distance_safe()))
                wait()
            else:
                print("moving forward - green={}, safe={}".format(is_traffic_light_green(), is_distance_safe()))
                move_forward()

    print("destination reached - done!")


if __name__ == "__main__":
    run()