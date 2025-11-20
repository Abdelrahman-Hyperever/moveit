#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FIXED: Analog control never blocks - runs in separate thread
"""

from __future__ import print_function

import threading
import time
import numpy
import rospy
import tf

from moveit_ros_planning_interface._moveit_robot_interface import RobotInterface
from std_msgs.msg import Empty, String
from sensor_msgs.msg import Joy
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import InteractiveMarkerInit


def signedSquare(val):
    return val * abs(val)


class JoyStatus:
    def __init__(self):
        self.center = False
        self.select = False
        self.start = False
        self.L3 = False
        self.R3 = False
        self.square = False
        self.up = False
        self.down = False
        self.left = False
        self.right = False
        self.triangle = False
        self.cross = False
        self.circle = False
        self.L1 = False
        self.R1 = False
        self.L2 = False
        self.R2 = False
        self.left_analog_x = 0.0
        self.left_analog_y = 0.0
        self.right_analog_x = 0.0
        self.right_analog_y = 0.0


class LogitechF710Status(JoyStatus):
    def __init__(self, msg):
        JoyStatus.__init__(self)
        
        self.cross = msg.buttons[1] == 1 if len(msg.buttons) > 1 else False
        self.circle = msg.buttons[2] == 1 if len(msg.buttons) > 2 else False
        self.square = msg.buttons[0] == 1 if len(msg.buttons) > 0 else False
        self.triangle = msg.buttons[3] == 1 if len(msg.buttons) > 3 else False
        
        self.L1 = msg.buttons[4] == 1 if len(msg.buttons) > 4 else False
        self.R1 = msg.buttons[5] == 1 if len(msg.buttons) > 5 else False
        self.L2 = msg.buttons[6] == 1 if len(msg.buttons) > 6 else False
        self.R2 = msg.buttons[7] == 1 if len(msg.buttons) > 7 else False
        
        self.select = msg.buttons[8] == 1 if len(msg.buttons) > 8 else False
        self.start = msg.buttons[9] == 1 if len(msg.buttons) > 9 else False
        self.L3 = msg.buttons[10] == 1 if len(msg.buttons) > 10 else False
        self.R3 = msg.buttons[11] == 1 if len(msg.buttons) > 11 else False
        
        if len(msg.axes) > 5:
            self.left = msg.axes[4] > 0.5
            self.right = msg.axes[4] < -0.5
            self.up = msg.axes[5] > 0.5
            self.down = msg.axes[5] < -0.5
        
        self.left_analog_x = msg.axes[0] if len(msg.axes) > 0 else 0.0
        self.left_analog_y = msg.axes[1] if len(msg.axes) > 1 else 0.0
        self.right_analog_x = msg.axes[2] if len(msg.axes) > 2 else 0.0
        self.right_analog_y = msg.axes[3] if len(msg.axes) > 3 else 0.0


class StatusHistory:
    def __init__(self, max_length=10):
        self.max_length = max_length
        self.buffer = []
        self.lock = threading.Lock()

    def add(self, status):
        with self.lock:
            self.buffer.append(status)
            if len(self.buffer) > self.max_length:
                self.buffer.pop(0)

    def all(self, proc):
        with self.lock:
            for status in self.buffer:
                if not proc(status):
                    return False
            return True

    def latest(self):
        with self.lock:
            return self.buffer[-1] if self.buffer else None

    def new(self, status, attr):
        with self.lock:
            if not self.buffer:
                return getattr(status, attr)
            return getattr(status, attr) and not getattr(self.buffer[-1], attr)


class MoveitJoy:
    def __init__(self):
        # Core state
        self.tf_listener = tf.TransformListener()
        self.history = StatusHistory(max_length=10)
        
        # Thread-safe current joystick state
        self.current_joy_status = None
        self.joy_lock = threading.Lock()
        
        # Pose state
        self.current_pose = PoseStamped()
        self.current_pose.pose.orientation.w = 1.0
        self.pose_lock = threading.Lock()
        
        # Marker state
        self.initial_poses = {}
        self.marker_lock = threading.Lock()
        self.initialize_poses = False
        
        # Planning groups
        self.planning_groups_tips = {}
        self.current_planning_group_index = 0
        self.current_eef_index = 0
        self.initialized = False
        
        # Control flags
        self.movement_stopped = False
        self.running = True
        
        # Button debounce
        self.last_button_time = {}
        
        # Parameters
        self.position_scale = rospy.get_param('~position_scale', 150.0)
        self.rotation_scale = rospy.get_param('~rotation_scale', 0.002)
        self.left_deadzone = rospy.get_param('~left_stick_deadzone', 0.05)
        self.right_deadzone = rospy.get_param('~right_stick_deadzone', 0.1)
        self.z_speed = rospy.get_param('~z_movement_speed', 0.003)
        self.update_rate = rospy.get_param('~update_rate', 30.0)
        self.accel_factor = rospy.get_param('~acceleration_factor', 1.5)
        self.debounce_time = rospy.get_param('~button_debounce_time', 0.3)
        
        # Initialize robot interface
        self.parseSRDF()
        
        # Publishers
        self.plan_group_pub = rospy.Publisher(
            "/rviz/moveit/select_planning_group", String, queue_size=5, latch=True)
        self.joy_pose_pub = rospy.Publisher("/joy_pose", PoseStamped, queue_size=1)
        self.plan_pub = rospy.Publisher("/rviz/moveit/plan", Empty, queue_size=5)
        self.execute_pub = rospy.Publisher("/rviz/moveit/execute", Empty, queue_size=5)
        self.update_start_state_pub = rospy.Publisher(
            "/rviz/moveit/update_start_state", Empty, queue_size=5)
        self.stop_pub = rospy.Publisher("/rviz/moveit/stop", Empty, queue_size=5)
        
        # Subscribers
        self.marker_sub = rospy.Subscriber(
            "/rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/update_full",
            InteractiveMarkerInit, self.markerCB, queue_size=1)
        
        self.joy_sub = rospy.Subscriber("/joy", Joy, self.joyCB, queue_size=1)
        
        # Initialize
        self.updatePlanningGroup(0)
        self.updatePoseTopic(0, wait=False)
        
        # Start analog control thread (CRITICAL FIX)
        self.analog_thread = threading.Thread(target=self.analogControlLoop)
        self.analog_thread.daemon = True
        self.analog_thread.start()
        
        rospy.loginfo("✓ MoveitJoy initialized with separate analog thread")

    def parseSRDF(self):
        ri = RobotInterface("/robot_description")
        planning_groups = {}
        
        for g in ri.get_group_names():
            self.planning_groups_tips[g] = ri.get_group_joint_tips(g)
            if len(self.planning_groups_tips[g]) > 0:
                planning_groups[g] = [
                    "/rviz/moveit/move_marker/goal_" + l
                    for l in self.planning_groups_tips[g]
                ]
        
        self.planning_groups = planning_groups
        self.planning_groups_keys = list(planning_groups.keys())
        self.frame_id = ri.get_planning_frame()
        
        rospy.loginfo("Planning groups: %s", self.planning_groups_keys)

    def updatePlanningGroup(self, next_index):
        if next_index >= len(self.planning_groups_keys):
            self.current_planning_group_index = 0
        elif next_index < 0:
            self.current_planning_group_index = len(self.planning_groups_keys) - 1
        else:
            self.current_planning_group_index = next_index
        
        group = self.planning_groups_keys[self.current_planning_group_index]
        rospy.loginfo("Planning group: %s", group)
        self.plan_group_pub.publish(group)

    def updatePoseTopic(self, next_index, wait=True):
        group = self.planning_groups_keys[self.current_planning_group_index]
        topics = self.planning_groups[group]
        
        if next_index >= len(topics):
            self.current_eef_index = 0
        elif next_index < 0:
            self.current_eef_index = len(topics) - 1
        else:
            self.current_eef_index = next_index
        
        next_topic = topics[self.current_eef_index]
        eef_name = self.planning_groups_tips[group][self.current_eef_index]
        
        rospy.loginfo("End effector: %s", eef_name)
        
        # Recreate publisher (CRITICAL FIX)
        self.pose_pub = rospy.Publisher(next_topic, PoseStamped, queue_size=5)
        rospy.sleep(0.1)  # Let publisher connect
        
        if wait:
            self.waitForInitialPose(next_topic)
        
        self.current_pose_topic = next_topic

    def markerCB(self, msg):
        with self.marker_lock:
            if not self.initialize_poses:
                return
            
            self.initial_poses = {}
            for marker in msg.markers:
                if marker.name.startswith("EE:goal_"):
                    try:
                        if marker.header.frame_id != self.frame_id:
                            ps = PoseStamped(header=marker.header, pose=marker.pose)
                            transformed = self.tf_listener.transformPose(self.frame_id, ps)
                            self.initial_poses[marker.name[3:]] = transformed.pose
                        else:
                            self.initial_poses[marker.name[3:]] = marker.pose
                    except Exception as e:
                        rospy.logwarn_throttle(5.0, "Marker TF error: %s", e)

    def waitForInitialPose(self, topic, timeout=30):
        topic_suffix = topic.split("/")[-1]
        
        for i in range(timeout):
            if rospy.is_shutdown():
                return False
            
            with self.marker_lock:
                self.initialize_poses = True
                if topic_suffix in self.initial_poses:
                    with self.pose_lock:
                        self.current_pose = PoseStamped(pose=self.initial_poses[topic_suffix])
                        self.current_pose.header.frame_id = self.frame_id
                    self.initialize_poses = False
                    rospy.loginfo("✓ Got initial pose for %s", topic_suffix)
                    return True
            
            if i % 10 == 0:
                rospy.loginfo("Waiting for %s... (%d/%d)", topic_suffix, i, timeout)
            
            rospy.sleep(0.1)
        
        rospy.logwarn("Timeout waiting for %s", topic_suffix)
        return False

    def joyCB(self, msg):
        """Just store the latest joystick state - don't process here"""
        axes = len(msg.axes)
        buttons = len(msg.buttons)
        
        if (axes == 6 or axes == 8) and buttons == 12:
            status = LogitechF710Status(msg)
            
            with self.joy_lock:
                self.current_joy_status = status
            
            self.history.add(status)
            
            # Handle buttons in callback (fast)
            self.handleButtons(status)
        else:
            rospy.logwarn_throttle(5.0, "Unknown controller: axes=%d, buttons=%d", axes, buttons)

    def handleButtons(self, status):
        """Process button presses immediately"""
        now = rospy.Time.now()
        
        # Initialize if needed
        if not self.initialized:
            self.initializePlanningGroup()
            return
        
        # Emergency stop
        if self.history.new(status, "select"):
            if self.checkDebounce("select", now):
                self.toggleStop()
            return
        
        if self.movement_stopped:
            if self.history.new(status, "start"):
                if self.checkDebounce("start", now):
                    self.toggleStop()
            return
        
        # Planning group change
        if self.history.new(status, "start"):
            if self.checkDebounce("start", now):
                self.updatePlanningGroup(self.current_planning_group_index - 1)
                self.current_eef_index = 0
                self.updatePoseTopic(self.current_eef_index)
                rospy.loginfo("📋 Plan")
                self.plan_pub.publish(Empty())
            return
        
        # End effector change
        if self.history.new(status, "triangle"):
            if self.checkDebounce("triangle", now):
                self.updatePoseTopic(self.current_eef_index + 1)
            return
        
        if self.history.new(status, "cross"):
            if self.checkDebounce("cross", now):
                self.updatePoseTopic(self.current_eef_index - 1)
            return
        
        # Plan
        if self.history.new(status, "square"):
            if self.checkDebounce("square", now):
                rospy.loginfo("📋 Plan")
                self.plan_pub.publish(Empty())
            return
        
        # Execute
        if self.history.new(status, "circle"):
            if self.checkDebounce("circle", now):
                rospy.loginfo("▶ Execute")
                self.execute_pub.publish(Empty())
            return

    def checkDebounce(self, button, now):
        """Check if enough time has passed since last button press"""
        if button not in self.last_button_time:
            self.last_button_time[button] = now
            return True
        
        if (now - self.last_button_time[button]).to_sec() > self.debounce_time:
            self.last_button_time[button] = now
            return True
        
        return False

    def toggleStop(self):
        self.movement_stopped = not self.movement_stopped
        if self.movement_stopped:
            rospy.loginfo("🛑 STOP")
            self.stop_pub.publish(Empty())
        else:
            rospy.loginfo("🔓 GO")

    def initializePlanningGroup(self):
        """Try to initialize with any available planning group"""
        for attempt in range(len(self.planning_groups_keys)):
            self.updatePlanningGroup(self.current_planning_group_index)
            group = self.planning_groups_keys[self.current_planning_group_index]
            topics = self.planning_groups[group]
            
            if self.waitForInitialPose(topics[0], timeout=30):
                rospy.loginfo("✓ Initialized with group: %s", group)
                self.initialized = True
                self.updatePoseTopic(0)
                return
            
            rospy.logwarn("Failed to init group %s, trying next...", group)
            self.current_planning_group_index += 1
            if self.current_planning_group_index >= len(self.planning_groups_keys):
                self.current_planning_group_index = 0

    def computePoseFromJoy(self, pre_pose, status):
        """Compute new pose from joystick input"""
        new_pose = PoseStamped()
        new_pose.header.frame_id = self.frame_id
        new_pose.header.stamp = rospy.Time(0.0)
        
        # Position
        x_diff = 0.0
        y_diff = 0.0
        z_diff = 0.0
        
        if abs(status.left_analog_y) > self.left_deadzone:
            x_diff = signedSquare(status.left_analog_y) / self.position_scale
        
        if abs(status.left_analog_x) > self.left_deadzone:
            y_diff = signedSquare(status.left_analog_x) / self.position_scale
        
        if status.L2:
            z_diff = self.z_speed
        elif status.R2:
            z_diff = -self.z_speed
        
        if abs(status.right_analog_y) > self.right_deadzone:
            z_diff += signedSquare(status.right_analog_y) / (self.position_scale * 1.5)
        
        # Apply acceleration
        if self.history.all(lambda s: s.L2) or self.history.all(lambda s: s.R2):
            z_diff *= self.accel_factor
        
        # Transform to world frame
        local_move = numpy.array([x_diff, y_diff, z_diff, 1.0])
        q = numpy.array([
            pre_pose.pose.orientation.x,
            pre_pose.pose.orientation.y,
            pre_pose.pose.orientation.z,
            pre_pose.pose.orientation.w
        ])
        
        q = q / numpy.linalg.norm(q)
        xyz_move = numpy.dot(tf.transformations.quaternion_matrix(q), local_move)
        
        new_pose.pose.position.x = pre_pose.pose.position.x + xyz_move[0]
        new_pose.pose.position.y = pre_pose.pose.position.y + xyz_move[1]
        new_pose.pose.position.z = pre_pose.pose.position.z + xyz_move[2]
        
        # Orientation
        roll = pitch = yaw = 0.0
        dt = self.rotation_scale
        
        if abs(status.right_analog_x) > self.right_deadzone:
            yaw = dt * signedSquare(status.right_analog_x)
        
        if status.L1:
            yaw += dt * (self.accel_factor if self.history.all(lambda s: s.L1) else 1.0)
        elif status.R1:
            yaw -= dt * (self.accel_factor if self.history.all(lambda s: s.R1) else 1.0)
        
        if status.up:
            pitch += dt * (self.accel_factor if self.history.all(lambda s: s.up) else 1.0)
        elif status.down:
            pitch -= dt * (self.accel_factor if self.history.all(lambda s: s.down) else 1.0)
        
        if status.right:
            roll += dt * (self.accel_factor if self.history.all(lambda s: s.right) else 1.0)
        elif status.left:
            roll -= dt * (self.accel_factor if self.history.all(lambda s: s.left) else 1.0)
        
        diff_q = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        new_q = tf.transformations.quaternion_multiply(q, diff_q)
        new_q = new_q / numpy.linalg.norm(new_q)
        
        new_pose.pose.orientation.x = new_q[0]
        new_pose.pose.orientation.y = new_q[1]
        new_pose.pose.orientation.z = new_q[2]
        new_pose.pose.orientation.w = new_q[3]
        
        return new_pose

    def analogControlLoop(self):
        """
        CRITICAL: This runs in a separate thread and NEVER blocks
        """
        rate = rospy.Rate(self.update_rate)
        counter = 0
        
        rospy.loginfo("✓ Analog control thread started")
        
        while not rospy.is_shutdown() and self.running:
            try:
                # Get current joystick state (thread-safe)
                with self.joy_lock:
                    status = self.current_joy_status
                
                if status is None or not self.initialized or self.movement_stopped:
                    rate.sleep()
                    continue
                
                # Get current pose (thread-safe)
                with self.pose_lock:
                    pre_pose = PoseStamped()
                    pre_pose.pose = self.current_pose.pose
                    pre_pose.header.frame_id = self.frame_id
                
                # Compute new pose
                new_pose = self.computePoseFromJoy(pre_pose, status)
                
                # Publish (thread-safe)
                self.pose_pub.publish(new_pose)
                self.joy_pose_pub.publish(new_pose)
                
                # Update current pose (thread-safe)
                with self.pose_lock:
                    self.current_pose = new_pose
                
                # Update marker state (thread-safe)
                with self.marker_lock:
                    topic_suffix = self.current_pose_topic.split("/")[-1]
                    self.initial_poses[topic_suffix] = new_pose.pose
                
                # Periodic start state update
                counter += 1
                if counter % 10 == 0:
                    self.update_start_state_pub.publish(Empty())
                
            except Exception as e:
                rospy.logerr("Analog control error: %s", e)
            
            rate.sleep()
        
        rospy.loginfo("Analog control thread stopped")

    def shutdown(self):
        """Clean shutdown"""
        rospy.loginfo("Shutting down...")
        self.running = False
        if self.analog_thread.is_alive():
            self.analog_thread.join(timeout=1.0)


if __name__ == '__main__':
    rospy.init_node('moveit_joy')
    
    try:
        node = MoveitJoy()
        rospy.on_shutdown(node.shutdown)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
