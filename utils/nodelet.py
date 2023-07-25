import rospy
from geometry_msgs.msg import Twist

mv_pub = None

def init():
    rospy.init_node("move_base_controller", anonymous=True)
    global mv_pub
    mv_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=5)


def get_mv_pub():
    return mv_pub