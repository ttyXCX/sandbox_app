import rospy
import time
from geometry_msgs.msg import Twist

if __name__ == "__main__":
    rospy.init_node("move_node")
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=2)
    print("init done")

    msg = Twist()
    msg.linear.x = 0.
    msg.linear.y = .0
    msg.linear.z = .0
    msg.angular.x = .0
    msg.angular.y = .0
    msg.angular.z = 0.1

    # EPOCH = 1
    # for i in range(EPOCH):

    pub.publish(msg)
    time.sleep(0.5)
    pub.publish(msg)

    # counter = 0
    # while not rospy.is_shutdown():
    #     pub.publish(msg)
    #     print("msg published")
    #     time.sleep(0.5)
        
    #     counter += 1
    #     if counter == 2:
    #         break
