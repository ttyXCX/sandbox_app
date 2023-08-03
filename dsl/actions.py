import rospy
import time
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt8

from utils.lifter_control import LifterController
from utils.nodelet import mv_pub

PUB_INTERVAL = 0.5
ACTION_INTERVAL = 1
MOVE_DIST = 0.2
TURN_ANG = 0.1

# rospy.init_node("move_base_controller", anonymous=True)
# mv_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=5)

lift_ctr = LifterController()


def __assign_Twist(lx=.0, ly=.0, lz=.0,
                   ax=.0, ay=.0, az=.0):
    msg = Twist()

    msg.linear.x = lx
    msg.linear.y = ly
    msg.linear.z = lz

    msg.angular.x = ax
    msg.angular.y = ay
    msg.angular.z = az

    return msg


def move_forward():
    msg = __assign_Twist(lx=MOVE_DIST)
    mv_pub.publish(msg)
    time.sleep(PUB_INTERVAL)


def move_backward():
    msg = __assign_Twist(lx=-MOVE_DIST)
    mv_pub.publish(msg)
    time.sleep(PUB_INTERVAL)


def turn_left():
    msg = __assign_Twist(az=TURN_ANG)
    mv_pub.publish(msg)

    time.sleep(PUB_INTERVAL)
    mv_pub.publish(msg)


def turn_right():
    msg = __assign_Twist(az=-TURN_ANG)
    mv_pub.publish(msg)

    time.sleep(PUB_INTERVAL)
    mv_pub.publish(msg)


def wait():
    time.sleep(ACTION_INTERVAL)


def __assign_UInt8(data=1):
    '''Deprecated function'''
    msg = UInt8()
    msg.data = data
    return msg


def lifter_up():
    lift_ctr.up()


def lifter_down():
    lift_ctr.down()


def lifter_hold():
    lift_ctr.hold()


if __name__ == "__main__":
    while (True):
        op = raw_input("input operation:")

        if op == "exit":
            break
        elif op == "m":
            move_forward()
        elif op == "b":
            move_backward()
        elif op == "l":
            turn_left()
        elif op == "r":
            turn_right()
        elif op == "lu":
            lifter_up()
        elif op == "ld":
            lifter_down()
        elif op == "lh":
            lifter_hold()
        else:
            print("{}Invalid operation '{}'{}".format("\033[0;31;47m", op, "\033[0m"))
