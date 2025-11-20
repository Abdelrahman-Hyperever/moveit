#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fixed version with proper planning/execution state tracking
"""

from __future__ import print_function

import xml.dom.minidom
from operator import add
import sys
import threading
from moveit_ros_planning_interface._moveit_robot_interface import RobotInterface

import rospy
import roslib
import numpy
import time
import tf
from std_msgs.msg import Empty, String
from sensor_msgs.msg import Joy
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import InteractiveMarkerInit
from moveit_msgs.msg import DisplayTrajectory, ExecuteTrajectoryActionResult
from actionlib_msgs.msg import GoalStatusArray


def signedSquare(val):
    if val > 0:
        sign = 1
    else:
        sign = -1
    return val * val * sign


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
        
        # Face buttons
        self.cross = msg.buttons[1] == 1 if len(msg.buttons) > 1 else False
        self.circle = msg.buttons[2] == 1 if len(msg.buttons) > 2 else False
        self.square = msg.buttons[0] == 1 if len(msg.buttons) > 0 else False
        self.triangle = msg.buttons[3] == 1 if len(msg.buttons) > 3 else False
        
        # Shoulder buttons
        self.L1 = msg.buttons[4] == 1 if len(msg.buttons) > 4 else False
        self.R1 = msg.buttons[5] == 1 if len(msg.buttons) > 5 else False
        
        # Triggers
        self.L2 = msg.buttons[6] == 1 if len(msg.buttons) > 6 else False
        self.R2 = msg.buttons[7] == 1 if len(msg.buttons) > 7 else False
        
        # Back/Start buttons
        self.select = msg.buttons[8] == 1 if len(msg.buttons) > 8 else False
        self.start = msg.buttons[9] == 1 if len(msg.buttons) > 9 else False
        
        # Stick buttons
        self.L3 = msg.buttons[10] == 1 if len(msg.buttons) > 10 else False
        self.R3 = msg.buttons[11] == 1 if len(msg.buttons) > 11 else False
        
        self.center = False
        
        # D-pad
        if len(msg.axes) > 5:
            self.left = msg.axes[4] > 0.5 if len(msg.axes) > 4 else False
            self.right = msg.axes[4] < -0.5 if len(msg.axes) > 4 else False
            self.up = msg.axes[5] > 0.5 if len(msg.axes) > 5 else False
            self.down = msg.axes[5] < -0.5 if len(msg.axes) > 5 else False
        else:
            self.up = msg.buttons[12] == 1 if len(msg.buttons) > 12 else False
            self.down = msg.buttons[13] == 1 if len(msg.buttons) > 13 else False
            self.left = msg.buttons[14] == 1 if len(msg.buttons) > 14 else False
            self.right = msg.buttons[15] == 1 if len(msg.buttons) > 15 else False
        
        # Analog sticks
        self.left_analog_x = msg.axes[0] if len(msg.axes) > 0 else 0.0
        self.left_analog_y = msg.axes[1] if len(msg.axes) > 1 else 0.0
        self.right_analog_x = msg.axes[2] if len(msg.axes) > 2 else 0.0
        self.right_analog_y = msg.axes[3] if len(msg.axes) > 3 else 0.0
        
        self.orig_msg = msg


class StatusHistory:
    def __init__(self, max_length=10):
        self.max_length = max_length
        self.buffer = []

    def add(self, status):
        self.buffer.append(status)
        if len(self.buffer) > self.max_length:
            self.buffer = self.buffer[1 : self.max_length + 1]

    def all(self, proc):
        for status in self.buffer:
            if not proc(status):
                return False
        return True

    def latest(self):
        if len(self.buffer) > 0:
            return self.buffer[-1]
        else:
            return None

    def length(self):
        return len(self.buffer)

    def new(self, status, attr):
        if len(self.buffer) == 0:
            return getattr(status, attr)
        else:
            return getattr(status, attr) and not getattr(self.latest(), attr)


class MoveitJoy:
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
        for name in planning_groups.keys():
            print(name, planning_groups[name])
        self.planning_groups = planning_groups
        self.planning_groups_keys = list(planning_groups.keys())
        self.frame_id = ri.get_planning_frame()

    def __init__(self):
        self.initial_poses = {}
        self.planning_groups_tips = {}
        self.tf_listener = tf.TransformListener()
        self.marker_lock = threading.Lock()
        self.prev_time = rospy.Time.now()
        self.counter = 0
        self.history = StatusHistory(max_length=10)
        self.pre_pose = PoseStamped()
        self.pre_pose.pose.orientation.w = 1
        self.current_planning_group_index = 0
        self.current_eef_index = 0
        self.initialize_poses = False
        self.initialized = False
        self.movement_stopped = False
        
        # State tracking with thread-safe lock
        self.state_lock = threading.Lock()
        self.is_planning = False
        self.is_executing = False
        self.last_button_time = rospy.Time.now()
        self.last_plan_time = rospy.Time.now()
        self.last_execute_time = rospy.Time.now()
        
        # Parameters
        self.debug_joystick = rospy.get_param('~debug_joystick', False)
        self.position_scale = rospy.get_param('~position_scale', 150.0)
        self.rotation_scale = rospy.get_param('~rotation_scale', 0.002)
        self.left_stick_deadzone = rospy.get_param('~left_stick_deadzone', 0.05)
        self.right_stick_deadzone = rospy.get_param('~right_stick_deadzone', 0.1)
        self.z_movement_speed = rospy.get_param('~z_movement_speed', 0.003)
        self.update_rate = rospy.get_param('~update_rate', 30.0)
        self.acceleration_factor = rospy.get_param('~acceleration_factor', 1.5)
        self.smooth_factor = rospy.get_param('~smooth_factor', 0.3)
        
        # Timeout parameters
        self.button_debounce_time = rospy.get_param('~button_debounce_time', 0.3)
        self.marker_timeout = rospy.get_param('~marker_timeout', 2.0)
        self.lock_timeout = rospy.get_param('~lock_timeout', 0.1)
        self.planning_timeout = rospy.get_param('~planning_timeout', 15.0)  # Max time for planning
        self.execution_timeout = rospy.get_param('~execution_timeout', 60.0)  # Max time for execution
        
        self.last_commanded_pose = None
        
        self.parseSRDF()
        
        # Publishers
        self.plan_group_pub = rospy.Publisher(
            "/rviz/moveit/select_planning_group", String, queue_size=5, latch=True
        )
        self.joy_pose_pub = rospy.Publisher("/joy_pose", PoseStamped, queue_size=1)
        self.plan_pub = rospy.Publisher("/rviz/moveit/plan", Empty, queue_size=5)
        self.execute_pub = rospy.Publisher("/rviz/moveit/execute", Empty, queue_size=5)
        self.update_start_state_pub = rospy.Publisher(
            "/rviz/moveit/update_start_state", Empty, queue_size=5
        )
        self.update_goal_state_pub = rospy.Publisher(
            "/rviz/moveit/update_goal_state", Empty, queue_size=5
        )
        self.stop_pub = rospy.Publisher("/rviz/moveit/stop", Empty, queue_size=5)
        
        # Subscribers for state feedback
        self.trajectory_sub = rospy.Subscriber(
            "/move_group/display_planned_path",
            DisplayTrajectory,
            self.trajectoryCallback,
            queue_size=1
        )
        
        self.execute_result_sub = rospy.Subscriber(
            "/execute_trajectory/result",
            ExecuteTrajectoryActionResult,
            self.executeResultCallback,
            queue_size=1
        )
        
        self.move_group_status_sub = rospy.Subscriber(
            "/move_group/status",
            GoalStatusArray,
            self.moveGroupStatusCallback,
            queue_size=1
        )
        
        self.interactive_marker_sub = rospy.Subscriber(
            "/rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/update_full",
            InteractiveMarkerInit,
            self.markerCB,
            queue_size=1,
            buff_size=2**20
        )
        
        self.sub = rospy.Subscriber("/joy", Joy, self.joyCB, queue_size=1)
        
        # Watchdog timer
        self.watchdog_timer = rospy.Timer(rospy.Duration(1.0), self.watchdogCB)
        self.last_joy_time = rospy.Time.now()
        
        # Initialize planning group
        self.updatePlanningGroup(0)
        self.updatePoseTopic(0, False)
        
        rospy.loginfo("MoveitJoy initialized with state feedback")

    # ========================================================================
    # STATE FEEDBACK CALLBACKS
    # ========================================================================
    
    def trajectoryCallback(self, msg):
        """Called when a trajectory is planned"""
        with self.state_lock:
            if self.is_planning:
                rospy.loginfo("Planning completed successfully")
                self.is_planning = False
                self.last_plan_time = rospy.Time.now()
    
    def executeResultCallback(self, msg):
        """Called when execution completes"""
        with self.state_lock:
            if self.is_executing:
                status = msg.status.status
                if status == 3:  # SUCCEEDED
                    rospy.loginfo("Execution completed successfully")
                elif status == 4:  # ABORTED
                    rospy.logwarn(" Execution aborted")
                elif status == 5:  # REJECTED
                    rospy.logwarn(" Execution rejected")
                else:
                    rospy.loginfo("Execution finished with status: %d", status)
                
                self.is_executing = False
                self.last_execute_time = rospy.Time.now()
    
    def moveGroupStatusCallback(self, msg):
        """Monitor move_group action status"""
        if len(msg.status_list) == 0:
            return
        
        with self.state_lock:
            latest_status = msg.status_list[-1]
            status_code = latest_status.status
            
            # Status codes:
            # 0 = PENDING, 1 = ACTIVE, 2 = PREEMPTED, 3 = SUCCEEDED
            # 4 = ABORTED, 5 = REJECTED, 6 = PREEMPTING, 7 = RECALLING
            # 8 = RECALLED, 9 = LOST
            
            if status_code in [3, 4, 5, 2, 8, 9]:  # Terminal states
                if self.is_planning:
                    rospy.logdebug("Planning terminal state: %d", status_code)
                    self.is_planning = False
                
                if self.is_executing:
                    rospy.logdebug("Execution terminal state: %d", status_code)
                    self.is_executing = False

    # ========================================================================
    # WATCHDOG
    # ========================================================================
    
    def watchdogCB(self, event):
        """Watchdog to detect stuck states and auto-recover"""
        now = rospy.Time.now()
        
        # Check joystick connection
        if (now - self.last_joy_time).to_sec() > 5.0:
            rospy.logwarn_throttle(10.0, "No joystick input for 5 seconds")
        
        with self.state_lock:
            # Check if planning is stuck
            if self.is_planning:
                time_since_plan = (now - self.last_plan_time).to_sec()
                if time_since_plan > self.planning_timeout:
                    rospy.logwarn("Planning timeout (%.1fs) - resetting flag", time_since_plan)
                    self.is_planning = False
            
            # Check if execution is stuck
            if self.is_executing:
                time_since_execute = (now - self.last_execute_time).to_sec()
                if time_since_execute > self.execution_timeout:
                    rospy.logwarn("Execution timeout (%.1fs) - resetting flag", time_since_execute)
                    self.is_executing = False

    # ========================================================================
    # PLANNING GROUP MANAGEMENT
    # ========================================================================

    def updatePlanningGroup(self, next_index):
        if next_index >= len(self.planning_groups_keys):
            self.current_planning_group_index = 0
        elif next_index < 0:
            self.current_planning_group_index = len(self.planning_groups_keys) - 1
        else:
            self.current_planning_group_index = next_index
        
        try:
            next_planning_group = self.planning_groups_keys[self.current_planning_group_index]
        except IndexError:
            msg = "Check if you started movegroups. Exiting."
            rospy.logfatal(msg)
            raise rospy.ROSInitException(msg)
        
        rospy.loginfo("Changed planning group to: %s", next_planning_group)
        self.plan_group_pub.publish(next_planning_group)

    def updatePoseTopic(self, next_index, wait=True):
        planning_group = self.planning_groups_keys[self.current_planning_group_index]
        topics = self.planning_groups[planning_group]
        
        if next_index >= len(topics):
            self.current_eef_index = 0
        elif next_index < 0:
            self.current_eef_index = len(topics) - 1
        else:
            self.current_eef_index = next_index
        
        next_topic = topics[self.current_eef_index]
        
        rospy.loginfo(
            "Changed end effector to: %s",
            self.planning_groups_tips[planning_group][self.current_eef_index]
        )
        
        self.pose_pub = rospy.Publisher(next_topic, PoseStamped, queue_size=5)
        
        if wait:
            self.waitForInitialPose(next_topic)
        
        self.current_pose_topic = next_topic

    # ========================================================================
    # MARKER MANAGEMENT
    # ========================================================================

    def markerCB(self, msg):
        acquired = self.marker_lock.acquire(blocking=True, timeout=self.lock_timeout)
        if not acquired:
            return
            
        try:
            if not self.initialize_poses:
                return
            
            self.initial_poses = {}
            for marker in msg.markers:
                if marker.name.startswith("EE:goal_"):
                    if marker.header.frame_id != self.frame_id:
                        ps = PoseStamped(header=marker.header, pose=marker.pose)
                        try:
                            transformed_pose = self.tf_listener.transformPose(self.frame_id, ps)
                            self.initial_poses[marker.name[3:]] = transformed_pose.pose
                        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                            rospy.logwarn_throttle(5.0, "TF error: %s", e)
                    else:
                        self.initial_poses[marker.name[3:]] = marker.pose
        finally:
            self.marker_lock.release()

    def waitForInitialPose(self, next_topic, timeout=None):
        counter = 0
        timeout_counter = timeout if timeout else 30
        
        while not rospy.is_shutdown():
            counter += 1
            if counter >= timeout_counter:
                rospy.logwarn("Timeout waiting for initial pose")
                return False
                
            acquired = self.marker_lock.acquire(blocking=True, timeout=self.lock_timeout)
            if not acquired:
                rospy.sleep(0.1)
                continue
                
            try:
                self.initialize_poses = True
                topic_suffix = next_topic.split("/")[-1]
                
                if topic_suffix in self.initial_poses:
                    self.pre_pose = PoseStamped(pose=self.initial_poses[topic_suffix])
                    self.initialize_poses = False
                    return True
                else:
                    if counter % 10 == 0:
                        rospy.loginfo("Waiting for pose topic '%s' (%d/%d)", 
                                     topic_suffix, counter, timeout_counter)
            finally:
                self.marker_lock.release()
                
            rospy.sleep(0.1)
        
        return False

    # ========================================================================
    # JOYSTICK CALLBACK
    # ========================================================================

    def joyCB(self, msg):
        self.last_joy_time = rospy.Time.now()
        
        axes_amount = len(msg.axes)
        buttons_amount = len(msg.buttons)
        
        if (axes_amount == 6 and buttons_amount == 12) or (axes_amount == 8 and buttons_amount == 12):
            status = LogitechF710Status(msg)
        else:
            rospy.logwarn_throttle(5.0, 
                "Unknown joystick: axes=%d, buttons=%d", axes_amount, buttons_amount)
            return
            
        self.run(status)
        self.history.add(status)

    # ========================================================================
    # POSE COMPUTATION
    # ========================================================================

    def computePoseFromJoy(self, pre_pose, status):
        new_pose = PoseStamped()
        new_pose.header.frame_id = self.frame_id
        new_pose.header.stamp = rospy.Time(0.0)
        
        # Position control
        scale = self.position_scale
        deadzone = self.left_stick_deadzone
        
        x_diff = signedSquare(status.left_analog_y) / scale if abs(status.left_analog_y) > deadzone else 0.0
        y_diff = signedSquare(status.left_analog_x) / scale if abs(status.left_analog_x) > deadzone else 0.0
        
        z_diff = 0.0
        if status.L2:
            z_diff = self.z_movement_speed
        elif status.R2:
            z_diff = -self.z_movement_speed
        
        right_deadzone = self.right_stick_deadzone
        if abs(status.right_analog_y) > right_deadzone:
            z_diff += signedSquare(status.right_analog_y) / (scale * 1.5)
        
        z_scale = self.acceleration_factor if (self.history.all(lambda s: s.L2) or self.history.all(lambda s: s.R2)) else 1.0
        
        local_move = numpy.array((x_diff, y_diff, z_diff * z_scale, 1.0))
        q = numpy.array((
            pre_pose.pose.orientation.x,
            pre_pose.pose.orientation.y,
            pre_pose.pose.orientation.z,
            pre_pose.pose.orientation.w,
        ))
        
        q_norm = numpy.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
            
        xyz_move = numpy.dot(tf.transformations.quaternion_matrix(q), local_move)
        new_pose.pose.position.x = pre_pose.pose.position.x + xyz_move[0]
        new_pose.pose.position.y = pre_pose.pose.position.y + xyz_move[1]
        new_pose.pose.position.z = pre_pose.pose.position.z + xyz_move[2]
        
        # Orientation control
        roll = pitch = yaw = 0.0
        DTHETA = self.rotation_scale
        
        if abs(status.right_analog_x) > right_deadzone and abs(status.right_analog_y) < right_deadzone:
            yaw = DTHETA * signedSquare(status.right_analog_x)
        
        if status.L1:
            yaw += DTHETA * (self.acceleration_factor if self.history.all(lambda s: s.L1) else 1.0)
        elif status.R1:
            yaw -= DTHETA * (self.acceleration_factor if self.history.all(lambda s: s.R1) else 1.0)
        
        if status.up:
            pitch += DTHETA * (self.acceleration_factor if self.history.all(lambda s: s.up) else 1.0)
        elif status.down:
            pitch -= DTHETA * (self.acceleration_factor if self.history.all(lambda s: s.down) else 1.0)
        
        if status.right:
            roll += DTHETA * (self.acceleration_factor if self.history.all(lambda s: s.right) else 1.0)
        elif status.left:
            roll -= DTHETA * (self.acceleration_factor if self.history.all(lambda s: s.left) else 1.0)
        
        diff_q = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        new_q = tf.transformations.quaternion_multiply(q, diff_q)
        
        new_q_norm = numpy.linalg.norm(new_q)
        if new_q_norm > 0:
            new_q = new_q / new_q_norm
            
        new_pose.pose.orientation.x = new_q[0]
        new_pose.pose.orientation.y = new_q[1]
        new_pose.pose.orientation.z = new_q[2]
        new_pose.pose.orientation.w = new_q[3]
        
        # Smoothing
        if self.last_commanded_pose is not None:
            smooth = self.smooth_factor
            new_pose.pose.position.x = smooth * self.last_commanded_pose.pose.position.x + (1-smooth) * new_pose.pose.position.x
            new_pose.pose.position.y = smooth * self.last_commanded_pose.pose.position.y + (1-smooth) * new_pose.pose.position.y
            new_pose.pose.position.z = smooth * self.last_commanded_pose.pose.position.z + (1-smooth) * new_pose.pose.position.z
            
            last_q = numpy.array([
                self.last_commanded_pose.pose.orientation.x,
                self.last_commanded_pose.pose.orientation.y,
                self.last_commanded_pose.pose.orientation.z,
                self.last_commanded_pose.pose.orientation.w
            ])
            
            current_q = numpy.array([
                new_pose.pose.orientation.x,
                new_pose.pose.orientation.y,
                new_pose.pose.orientation.z,
                new_pose.pose.orientation.w
            ])
            
            if numpy.dot(last_q, current_q) < 0:
                current_q = -current_q
                
            smooth_q = tf.transformations.quaternion_slerp(last_q, current_q, 1-smooth)
            
            new_pose.pose.orientation.x = smooth_q[0]
            new_pose.pose.orientation.y = smooth_q[1]
            new_pose.pose.orientation.z = smooth_q[2]
            new_pose.pose.orientation.w = smooth_q[3]
        
        self.last_commanded_pose = new_pose
        return new_pose

    # ========================================================================
    # MAIN CONTROL LOOP
    # ========================================================================

    def toggleStop(self):
        if self.movement_stopped:
            rospy.loginfo("🔓 Movement unlocked")
            self.movement_stopped = False
        else:
            rospy.loginfo("🛑 EMERGENCY STOP - Movement locked")
            self.movement_stopped = True
            self.stop_pub.publish(Empty())

    def run(self, status):
        now = rospy.Time.now()
        
        # Initialize if needed
        if not self.initialized:
            while True:
                self.updatePlanningGroup(self.current_planning_group_index)
                planning_group = self.planning_groups_keys[self.current_planning_group_index]
                topics = self.planning_groups[planning_group]
                next_topic = topics[self.current_eef_index]
                
                if not self.waitForInitialPose(next_topic, timeout=30):
                    rospy.logwarn("Unable to initialize group %s. Trying next.", planning_group)
                else:
                    rospy.loginfo("Initialized planning group: %s", planning_group)
                    self.initialized = True
                    self.updatePoseTopic(self.current_eef_index)
                    return
                
                self.current_planning_group_index += 1
                if self.current_planning_group_index >= len(self.planning_groups_keys):
                    self.current_planning_group_index = 0
        
        # Emergency stop toggle
        if self.history.new(status, "select"):
            if (now - self.last_button_time).to_sec() > self.button_debounce_time:
                self.toggleStop()
                self.last_button_time = now
            return
        
        if self.movement_stopped:
            if self.history.new(status, "start"):
                if (now - self.last_button_time).to_sec() > self.button_debounce_time:
                    self.toggleStop()
                    self.last_button_time = now
            return
        
        # Button controls with debounce
        if self.history.new(status, "start"):
            if (now - self.last_button_time).to_sec() > self.button_debounce_time:
                self.updatePlanningGroup(self.current_planning_group_index - 1)
                self.current_eef_index = 0
                self.updatePoseTopic(self.current_eef_index)
                rospy.loginfo(" Planning after group change")
                self.plan_pub.publish(Empty())
                with self.state_lock:
                    self.is_planning = True
                    self.last_plan_time = now
                self.last_button_time = now
            return
            
        elif self.history.new(status, "triangle"):
            if (now - self.last_button_time).to_sec() > self.button_debounce_time:
                self.updatePoseTopic(self.current_eef_index + 1)
                self.last_button_time = now
            return
            
        elif self.history.new(status, "cross"):
            if (now - self.last_button_time).to_sec() > self.button_debounce_time:
                self.updatePoseTopic(self.current_eef_index - 1)
                self.last_button_time = now
            return
            
        elif self.history.new(status, "square"):
            if (now - self.last_button_time).to_sec() > self.button_debounce_time:
                rospy.loginfo(" Plan requested")
                self.plan_pub.publish(Empty())
                with self.state_lock:
                    self.is_planning = True
                    self.last_plan_time = now
                self.last_button_time = now
            return
            
        elif self.history.new(status, "circle"):
            if (now - self.last_button_time).to_sec() > self.button_debounce_time:
                rospy.loginfo(" Execute requested")
                self.execute_pub.publish(Empty())
                with self.state_lock:
                    self.is_executing = True
                    self.last_execute_time = now
                self.last_button_time = now
            return
        
        # Analog control - non-blocking
        acquired = self.marker_lock.acquire(blocking=False)
        if not acquired:
            return
            
        try:
            pre_pose = self.pre_pose
            new_pose = self.computePoseFromJoy(pre_pose, status)
            
            if (now - self.prev_time).to_sec() > 1.0 / self.update_rate:
                self.pose_pub.publish(new_pose)
                self.joy_pose_pub.publish(new_pose)
                self.prev_time = now
                
            self.counter += 1
            if self.counter % 10 == 0:
                self.update_start_state_pub.publish(Empty())
                
            self.pre_pose = new_pose
            self.initial_poses[self.current_pose_topic.split("/")[-1]] = new_pose.pose
            
        finally:
            self.marker_lock.release()


if __name__ == '__main__':
    rospy.init_node('moveit_joy')
    try:
        node = MoveitJoy()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
