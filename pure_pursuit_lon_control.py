import rclpy
from rclpy.node import Node
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from scipy.linalg import solve_continuous_are
import time
import threading
import os
import sys

sys.path.insert(0, os.path.abspath('./assetto_corsa_gym'))

try:
    from common_msgs.msg import VehicleState
    from ac_actuation_msgs.msg import SteeringReport
except ImportError:
    workspace_root = os.path.abspath(os.path.dirname(__file__))
    for python_version in ['python3.8', 'python3.10', 'python3.11', 'python3.12']:
        local_site_packages = os.path.join(workspace_root, 'install', 'common_msgs', 'lib', python_version, 'site-packages')
        if os.path.exists(local_site_packages):
            sys.path.append(local_site_packages)
        
        local_site_packages_actuation = os.path.join(workspace_root, 'install', 'ac_actuation_msgs', 'lib', python_version, 'site-packages')
        if os.path.exists(local_site_packages_actuation):
            sys.path.append(local_site_packages_actuation)
            
    try:
        from common_msgs.msg import VehicleState
        from ac_actuation_msgs.msg import SteeringReport
    except ImportError:
        print("Error: Could not import VehicleState or SteeringReport. Make sure you have sourced the workspace setup file.")
        sys.exit(1)

from AssettoCorsaEnv.vjoy_linux import vJoy

class AssettoCorsaAutonomousAgent(Node):
    def __init__(self):
        super().__init__('ac_autonomous_agent')

        self.log_file = None 
        self.csv_log_file = None 
        self.last_wait_log_time = 0.0

        self.trajectory_path = "ac_offline_train_paths_gt3.yml" 
        self.csv_path = "assetto_corsa_gym/AssettoCorsaConfigs/tracks/ks_laguna_seca-racing_line_3d_full.csv"
        
        self.wheelbase = 2.9718
        self.min_lookahead = 2.5
        self.max_lookahead = 30.0
        self.lookahead_gain = 0.32
        self.max_steer_angle = 0.19
        
        self.steer_filter_alpha = 1.0
        self.max_steer_rate = 6.0
        
        self.brake_lookahead_gain = 3.0 

        self.use_lqr = True
        self.K_lqr_accel = None
        self.K_lqr_brake = None
        self.compute_lqr_gains()

        self.kp_speed = 0.18
        self.ki_speed = 0.002  
        self.kd_speed = 0.01 
        
        self.k_accel = 0.4 
        self.kp_brake = 0.6
        self.ki_brake = 0.03  
        
        self.k_brake_accel = 0.04 
        self.integral_speed_error = 0.0
        self.prev_speed_error = 0.0
        self.max_integral = 10.0  
        
        self.filtered_target_speed = 0.0
        self.prev_throttle = 0.0
        self.prev_brake = 0.0
        self.max_throttle_rate = 5.0
        self.max_brake_rate = 10.0
        
        self.throttle_filter_alpha = 0.5
        self.filtered_throttle = 0.0
        
        self.brake_filter_alpha = 0.7
        self.filtered_brake = 0.0
        
        self.speed_deadband = 1.0 / 3.6 
        self.max_lateral_accel = 30.0
        self.min_speed = 30.0
        
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_speed = 0.0
        self.current_steer_actual = 0.0
        self.state_received = False
        
        self.filtered_steer = 0.0
        self.prev_steer_cmd = 0.0
        self.last_control_time = None


        self.log_to_file(f"Loading trajectory from {self.csv_path}...")
        try:
            self.df = pd.read_csv(self.csv_path)
            self.path_points = self.df[['pos_x', 'pos_y']].values
            self.path_tree = KDTree(self.path_points)
            self.log_to_file(f"Loaded {len(self.df)} trajectory points.")
        except Exception as e:
            self.log_to_file(f"Failed to load trajectory: {e}")
            raise e

        self.log_to_file("Initializing vJoy...")
        self.joy = vJoy()
        self.joy.open()
        self.log_to_file("vJoy initialized.")

        self.sub_state = self.create_subscription(
            VehicleState,
            '/vehicle_state',
            self.vehicle_state_callback,
            10
        )
        
        self.sub_steer = self.create_subscription(
            SteeringReport,
            '/ac_actuation/steering_report',
            self.steering_callback,
            10
        )

        self.pub_state_with_target = self.create_publisher(
             VehicleState,
             '/vehicle_target_state',
             10
        )

        self.create_timer(0.04, self.control_loop)

    def log_to_file(self, message):
        return

    def vehicle_state_callback(self, msg):
        self.current_x = msg.world_position.position.x
        self.current_y = msg.world_position.position.y
        self.current_yaw = msg.yaw 
        self.current_speed = msg.speed_kmh / 3.6
        self.state_received = True
        self.last_state_msg = msg

    def steering_callback(self, msg):
        self.current_steer_actual = msg.steering_wheel_angle

    def get_target_point(self, gain_override=None):
        gain = self.lookahead_gain if gain_override is None else gain_override
        
        lookahead_distance = self.min_lookahead + gain * self.current_speed
        lookahead_distance = min(lookahead_distance, self.max_lookahead)
        
        speed_lookahead_distance = lookahead_distance * 3.0
        
        dist, idx = self.path_tree.query([self.current_x, self.current_y])
        

        current_idx = idx
        path_len = len(self.path_points)
        
        steering_target = None
        min_speed_in_window = 100.0
        
        for i in range(path_len):
            next_idx = (current_idx + i) % path_len
            dx = self.path_points[next_idx][0] - self.current_x
            dy = self.path_points[next_idx][1] - self.current_y
            d = np.sqrt(dx*dx + dy*dy)
            
            if d < speed_lookahead_distance:
                pt_speed = self.df.iloc[next_idx]['target_speed']
                if pt_speed < min_speed_in_window:
                    min_speed_in_window = pt_speed
            
            if steering_target is None and d > lookahead_distance:
                steering_target = self.df.iloc[next_idx]
            
            if d > speed_lookahead_distance and steering_target is not None:
                break
        
        if steering_target is None:
            steering_target = self.df.iloc[current_idx]
            
        return steering_target, min_speed_in_window

    def compute_lqr_gains(self):
        a = 0.5
        
        Q = np.diag([0.5, 0.01]) 
        
        R = np.array([[20.0]])    
        
        R_brake = np.array([[60.0]])

        b_accel = -5.0 
        
        A = np.array([[-a, 0], [1, 0]])
        B_acc = np.array([[b_accel], [0]])
        
        try:
            P_acc = solve_continuous_are(A, B_acc, Q, R)
            K_acc = np.linalg.inv(R) @ B_acc.T @ P_acc
            self.K_lqr_accel = K_acc.flatten()
            self.log_to_file(f"LQR Accel Gains: {self.K_lqr_accel}")
        except Exception as e:
            self.K_lqr_accel = None

        b_brake = 30.0 
        B_brk = np.array([[b_brake], [0]])
        
        try:
            P_brk = solve_continuous_are(A, B_brk, Q, R_brake)
            K_brk = np.linalg.inv(R_brake) @ B_brk.T @ P_brk
            self.K_lqr_brake = K_brk.flatten()
            self.log_to_file(f"LQR Brake Gains: {self.K_lqr_brake}")
        except Exception as e:
            self.K_lqr_brake = None

    def control_loop(self):
        if not self.state_received:
            current_time = time.time()
            if current_time - self.last_wait_log_time > 2.0:
                self.log_to_file("Waiting for vehicle state...")
                self.last_wait_log_time = current_time
            return

        steering_target, speed_target_raw = self.get_target_point()
        
        target_x = steering_target['pos_x']
        target_y = steering_target['pos_y']
        
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        
        local_x = dx * np.cos(self.current_yaw) + dy * np.sin(self.current_yaw)
        local_y = -dx * np.sin(self.current_yaw) + dy * np.cos(self.current_yaw)
        
        ld_squared = local_x**2 + local_y**2
        if ld_squared > 0.01:
            curvature = 2.0 * (-local_y) / ld_squared
            raw_steer = np.arctan(self.wheelbase * curvature)
            raw_steer = raw_steer / self.max_steer_angle
        else:
            raw_steer = 0.0
        
        raw_steer = max(-1.0, min(1.0, raw_steer))
        

        current_time = time.time()
        if self.last_control_time is None:
            dt = 0.02
        else:
            dt = current_time - self.last_control_time
            dt = max(0.001, min(0.1, dt))
        self.last_control_time = current_time
        
        self.filtered_steer = self.steer_filter_alpha * raw_steer + (1 - self.steer_filter_alpha) * self.filtered_steer
        

        max_delta = self.max_steer_rate * dt
        steer_delta = self.filtered_steer - self.prev_steer_cmd
        steer_delta = max(-max_delta, min(max_delta, steer_delta))
        steer_cmd = self.prev_steer_cmd + steer_delta
        self.prev_steer_cmd = steer_cmd
        
        steer_cmd = max(-1.0, min(1.0, steer_cmd))

        abs_steer = abs(self.filtered_steer)
        if abs_steer > 0.1:
            approx_curvature = abs(np.tan(abs_steer * self.max_steer_angle) / self.wheelbase)
            if approx_curvature > 0.001:
                curvature_limited_speed = np.sqrt(self.max_lateral_accel / approx_curvature)
                curvature_limited_speed = max(self.min_speed, curvature_limited_speed)
            else:
                curvature_limited_speed = 100.0
        else:
            curvature_limited_speed = 100.0
        
        target_speed = min(speed_target_raw, curvature_limited_speed, 100.0)
        

        if self.filtered_target_speed == 0.0:
            self.filtered_target_speed = target_speed
        
        self.filtered_target_speed = 0.5 * target_speed + 0.5 * self.filtered_target_speed
        
        speed_error = self.filtered_target_speed - self.current_speed
        
        self.integral_speed_error += speed_error * dt
        self.integral_speed_error = max(-self.max_integral, min(self.max_integral, self.integral_speed_error))
        
        derivative_speed_error = (speed_error - self.prev_speed_error) / dt
        self.prev_speed_error = speed_error

        brake_target_point, brake_target_speed_raw = self.get_target_point(gain_override=self.brake_lookahead_gain)
        
        brake_dist = max(1.0, np.linalg.norm([brake_target_point['pos_x']-self.current_x, brake_target_point['pos_y']-self.current_y]))
        target_accel = (target_speed**2 - self.current_speed**2) / (2 * brake_dist)
        throttle_ff = max(0.0, min(1.0, self.k_accel * target_accel))
        
        brake_spd_lim = min(brake_target_speed_raw, curvature_limited_speed, 100.0)
        target_decel = (brake_spd_lim**2 - self.current_speed**2) / (2 * brake_dist)
        brake_ff = max(0.0, min(1.0, -self.k_brake_accel * target_decel)) if target_decel < 0 else 0.0

        throttle_cmd = 0.0
        brake_cmd = 0.0
        
        if speed_error > self.speed_deadband:
            if self.use_lqr and self.K_lqr_accel is not None:
                kp = abs(self.K_lqr_accel[0])
                ki = abs(self.K_lqr_accel[1])
            else:
                kp = self.kp_speed
                ki = self.ki_speed

            pid_output = kp * speed_error + ki * self.integral_speed_error + self.kd_speed * derivative_speed_error
            throttle_cmd = throttle_ff + pid_output
            
            steer_penalty = abs(self.filtered_steer) * 1.5 
            max_throttle_steer = max(0.2, 1.0 - steer_penalty)
            if self.current_speed < 80.0/3.6: 
                max_throttle_speed = 0.6 + (0.4 * (self.current_speed / (80.0/3.6)))
            else:
                max_throttle_speed = 1.0
            
            stability_limit = min(max_throttle_steer, max_throttle_speed)
            throttle_cmd = max(0.0, min(stability_limit, throttle_cmd))
        
        elif speed_error < -self.speed_deadband:
            if self.use_lqr and self.K_lqr_brake is not None:
                kp = abs(self.K_lqr_brake[0])
                ki = abs(self.K_lqr_brake[1])
            else:
                kp = self.kp_brake
                ki = self.ki_brake

            brake_pid = -(kp * speed_error + ki * self.integral_speed_error)
            
            brake_cmd = max(0.0, min(1.0, brake_ff + brake_pid))
            
            if abs(self.filtered_steer) > 0.25:
                brake_cmd *= 0.9
        
        else:
            throttle_cmd = 0.0
            brake_cmd = 0.0
            self.integral_speed_error *= 0.95
            
        max_thr_delta = 8.0 * dt
        thr_delta = throttle_cmd - self.prev_throttle
        thr_delta = max(-max_thr_delta, min(max_thr_delta, thr_delta))
        throttle_cmd = self.prev_throttle + thr_delta
        self.prev_throttle = throttle_cmd
        
        self.filtered_throttle = self.throttle_filter_alpha * throttle_cmd + (1 - self.throttle_filter_alpha) * self.filtered_throttle
        throttle_cmd = self.filtered_throttle
        
        max_brk_delta = self.max_brake_rate * dt
        brk_delta = brake_cmd - self.prev_brake
        brk_delta = max(-max_brk_delta, min(max_brk_delta, brk_delta))
        brake_cmd = self.prev_brake + brk_delta
        self.prev_brake = brake_cmd
        
        self.filtered_brake = self.brake_filter_alpha * brake_cmd + (1 - self.brake_filter_alpha) * self.filtered_brake
        brake_cmd = self.filtered_brake

        steer_int = int((steer_cmd + 1.0) * 16384)
        steer_int = max(0, min(32768, steer_int))

        throttle_int = int(throttle_cmd * 32768)
        throttle_int = max(0, min(32768, throttle_int))

        brake_int = int(brake_cmd * 32768)
        brake_int = max(0, min(32768, brake_int))

        pos = self.joy.generateJoystickPosition(
            wAxisX=steer_int,
            wAxisY=throttle_int,
            wAxisZ=brake_int
        )
        self.joy.update(pos)
        
        self.log_to_file(f"TgtSpd: {target_speed:.1f} | CurSpd: {self.current_speed:.1f} | Steer: {steer_cmd:.2f} | Thr: {throttle_cmd:.2f} | Brk: {brake_cmd:.2f} | LocalY: {abs(local_y):.1f}")
        
        if hasattr(self, 'last_state_msg'):
            out_msg = self.last_state_msg
            out_msg.target_speed = float(target_speed)
            self.pub_state_with_target.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    agent = AssettoCorsaAutonomousAgent()
    
    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        pass
    finally:
        if agent.joy.uinput:
             pos = agent.joy.generateJoystickPosition(wAxisX=16384, wAxisY=0, wAxisZ=32768)
             agent.joy.update(pos)
             agent.joy.close()
        if hasattr(agent, 'log_file') and agent.log_file:
            agent.log_file.close()
        if hasattr(agent, 'csv_log_file') and agent.csv_log_file:
            agent.csv_log_file.close()
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()