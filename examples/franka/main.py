import os

from cv_bridge import CvBridge
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState

os.environ["ROS_MASTER_URI"] = "http://192.168.5.7:11311"
os.environ["ROS_IP"] = "192.168.5.8"
print("ROS_MASTER_URI is set to:", os.environ["ROS_MASTER_URI"])


class PI0:
    def __init__(self, local_host):
        self.policy = websocket_client_policy.WebsocketClientPolicy(host=local_host, port=8000)
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.instruction = "Stack the cups in a counterclockwise direction."

    def update_observation_window(self, img, wrist_img, joints, gripper):
        self.observation_window = {
            "joints": np.array(joints, dtype=np.float32),
            "gripper": np.array(gripper, dtype=np.float32),
            "image": image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224)),
            "wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, 224, 224)),
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]


class panda:
    def __init__(self):
        self.img = None
        self.wrist_img = None
        self.bridge = CvBridge()
        self.joints = None
        self.gripper = None
        rospy.init_node("eval_pi0", anonymous=True)
        # 三个订阅者
        rospy.Subscriber("/ob_camera_01/color/image_raw", Image, self.img_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(
            "/ob_camera_02/color/image_raw", Image, self.wrist_img_callback, queue_size=1000, tcp_nodelay=True
        )
        rospy.Subscriber("/joint_states", JointState, self.joint_callback, queue_size=1000, tcp_nodelay=True)
        # 两个发布者
        self.joint_pub = rospy.Publisher("/io_teleop/joint_cmd", JointState, queue_size=10)
        self.gripper_pub = rospy.Publisher("/io_teleop/target_gripper_status", JointState, queue_size=10)

    def img_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.img = img

    def wrist_img_callback(self, msg):
        wrist_img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.wrist_img = wrist_img

    def joint_callback(self, msg):
        joints = msg.position[:7]
        gripper = 1 - 2 * msg.position[7] / 0.08
        self.joints = joints
        self.gripper = gripper

    def joint_pub(self, joints):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]
        msg.position = joints[:7]
        self.joint_pub.publish(msg)

    def gripper_pub(self, joints):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.position = [0.0, 0.0]
        msg.name = ["joint_1", "joint_2"]
        if joints[7] >= 0.5:
            msg.position[0] = 1.0
        else:
            msg.position[0] = 0.0
        self.gripper_pub.publish(msg)


def main():
    model = PI0("192.168.2.32")
    robot = panda()

    rate = rospy.Rate(15)

    while not rospy.is_shutdown():
        if (
            robot.img is not None
            and robot.wrist_img is not None
            and robot.joints is not None
            and robot.gripper is not None
        ):
            model.update_observation_window(robot.img, robot.wrist_img, robot.joints, robot.gripper)
            robot.img = None
            robot.wrist_img = None
            robot.joints = None
            robot.gripper = None
            actions = model.get_action()
            assert actions.shape == (50, 8)
            actions = actions[:12]
            for i in range(12):
                robot.joint_pub(actions[i])
                robot.gripper_pub(actions[i])
                rate.sleep()
        rate.sleep()
