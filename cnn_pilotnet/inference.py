import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistWithCovarianceStamped
import message_filters

# FSDS의 커스텀 제어 메시지
from fs_msgs.msg import ControlCommand 

import math
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms

MAX_SPEED = 10.0  # 모델 출력 디코딩용 상한 (m/s), train.py와 동일해야 함

# 차량 파라미터 (1/r 변환용)
WHEELBASE = 1.55          # 축거 (m)
MAX_STEER_ANGLE = 0.5    # 최대 조향각 (rad, ~28.6°)
MAX_INV_RADIUS = math.tan(MAX_STEER_ANGLE) / WHEELBASE  # 최대 역회전반경

# ==========================================
# PID 제어기
# ==========================================
class PIDController:
    def __init__(self, kp, ki, kd, integral_limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.prev_error = None
        self.prev_time = None

    def compute(self, error, current_time_sec):
        """PID 출력 계산. 양수=가속, 음수=감속"""
        if self.prev_time is None:
            self.prev_time = current_time_sec
            self.prev_error = error
            return max(-1.0, min(1.0, self.kp * error))

        dt = current_time_sec - self.prev_time
        if dt <= 0:
            return max(-1.0, min(1.0, self.kp * error))

        # Integral (anti-windup)
        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))

        # Derivative
        derivative = (error - self.prev_error) / dt

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = max(-1.0, min(1.0, output))

        self.prev_error = error
        self.prev_time = current_time_sec
        return output

    def reset(self):
        self.integral = 0.0
        self.prev_error = None
        self.prev_time = None

# ==========================================
# 1. 모델 아키텍처 (train.py와 완전히 동일해야 함)
# ================================================================
class PilotNet(nn.Module):
    def __init__(self, in_channels=3):
        super(PilotNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(1152, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(50, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 2),  # Steering, Speed(정규화)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        # Steering: tanh → [-1, 1]
        steering = torch.tanh(x[:, 0:1])
        # Speed(정규화): sigmoid → [0, 1]
        speed_norm = torch.sigmoid(x[:, 1:2])
        return torch.cat([steering, speed_norm], dim=1)

    def get_visual_backprop(self, x):
        """Visual Backpropagation: 모델이 의존하는 이미지 영역을 마스크 형태로 추출"""
        feature_maps = []
        out = x
        for layer in self.features:
            out = layer(out)
            if isinstance(layer, nn.ReLU): 
                # ReLU 활성화 후의 피처 맵을 저장
                feature_maps.append(out)
                if len(feature_maps) == 5: # 마지막 Conv-ReLU까지만 (총 5개)
                    break 
                
        # 1. 마지막 레이어의 피처맵을 채널 방향으로 평균
        mask = feature_maps[-1].mean(dim=1, keepdim=True)
        
        # 2. 역방향으로 진행하며 업샘플링 후 곱셈
        for i in range(len(feature_maps)-2, -1, -1):
            prev_map = feature_maps[i].mean(dim=1, keepdim=True)
            # 이전 피처맵 크기에 맞게 업샘플링
            mask = nn.functional.interpolate(mask, size=prev_map.shape[2:], mode='bilinear', align_corners=False)
            # Point-wise 곱셈
            mask = mask * prev_map
            
        # 3. 원본 입력 크기 (66x200)로 최종 업샘플링
        mask = nn.functional.interpolate(mask, size=(66, 200), mode='bilinear', align_corners=False)
        
        # 4. 0~1 사이로 정규화
        mask = mask - mask.min()
        mask = mask / (mask.max() + 1e-8)
        
        return mask.squeeze().cpu().numpy()

# ==========================================
# 2. ROS 2 추론 노드
# ==========================================
class E2EInferenceNode(Node):
    def __init__(self):
        super().__init__('e2e_inference_node')

        # 장치 설정 및 모델 로드
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PilotNet(in_channels=6).to(self.device)
        
        # 저장된 가중치 불러오기
        model_path = '/home/ace/fsds_dataset/pretrained_model/pilotnet_model_20260520_223740.pth'
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() # 평가(추론) 모드로 전환
        self.get_logger().info(f"모델 로드 완료: {model_path} ({self.device})")

        # 이미지 변환 도구 세팅 (train.py와 동일한 전처리, YUV 입력)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((66, 200)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 현재 속도 (GSS로부터 업데이트)
        self.current_speed = 0.0

        # PID 제어기 (목표 속도 → throttle/brake 변환)
        self.pid = PIDController(kp=0.20, ki=0.01, kd=0.02)

        # Publisher (제어 명령 전송)
        self.cmd_publisher = self.create_publisher(ControlCommand, '/fsds/control_command', 10)
        
        # 스티어링 스무딩 (EMA 필터) 용 변수
        self.ema_steering = 0.0
        self.ema_alpha = 0.2  # 새로운 예측값을 20%만 반영, 이전 값을 80% 유지

        # cam2 이미지 버퍼
        self.cam2_image = None

        # Subscriber (카메라 이미지 수신 - TimeSynchronizer로 cam1, cam2 정확한 프레임 동기화 매칭)
        self.cam1_sub = message_filters.Subscriber(self, Image, '/fsds/cam1/image_color')
        self.cam2_sub = message_filters.Subscriber(self, Image, '/fsds/cam2/image_color')
        self.ts = message_filters.ApproximateTimeSynchronizer([self.cam1_sub, self.cam2_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.sync_cam_callback)

        # Subscriber (GSS: 현재 속도 피드백)
        self.gss_sub = self.create_subscription(
            TwistWithCovarianceStamped, '/fsds/gss', self.gss_callback, 10)

    @staticmethod
    def ros_image_to_bgr(msg):
        """ROS Image 메시지를 OpenCV BGR 이미지로 변환"""
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding == 'bgra8':
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img  # bgr8

    def gss_callback(self, msg):
        """현재 속도 업데이트 (Ground Speed Sensor)"""
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_speed = math.sqrt(vx * vx + vy * vy)

    def sync_cam_callback(self, cam1_msg, cam2_msg):
        """TimeSynchronizer로 동기화된 cam1+cam2 프레임으로 추론"""
        try:
            cam1_bgr = self.ros_image_to_bgr(cam1_msg)
            self.cam2_image = self.ros_image_to_bgr(cam2_msg)

        except Exception as e:
            self.get_logger().error(f"cam1 이미지 변환 오류: {e}")
            return

        # 이미지 Crop (Top 0.52, Bottom 0.7)
        h, w = cam1_bgr.shape[:2]
        top = int(h * 0.52)
        bottom = int(h * 0.70)
        
        cam1_cropped = cam1_bgr[top:bottom, :]
        cam2_cropped = self.cam2_image[top:bottom, :]

        # 1. BGR → YUV 변환 (논문과 동일, train.py와 일치)
        cam1_yuv = cv2.cvtColor(cam1_cropped, cv2.COLOR_BGR2YUV)
        cam2_yuv = cv2.cvtColor(cam2_cropped, cv2.COLOR_BGR2YUV)

        # 2. 각 카메라 이미지를 텐서로 변환 후 6채널로 결합
        cam1_tensor = self.transform(cam1_yuv)
        cam2_tensor = self.transform(cam2_yuv)
        input_tensor = torch.cat([cam1_tensor, cam2_tensor], dim=0).unsqueeze(0).to(self.device)

        # 3. 모델 추론
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # Visual Backpropagation 마스크 생성
            vbp_mask = self.model.get_visual_backprop(input_tensor)

        # 결과값 추출 (1/r: 역회전반경)
        inv_radius_norm = float(output[0][0])           # [-1, 1]
        inv_radius = inv_radius_norm * MAX_INV_RADIUS    # 실제 1/r (1/m)
        speed_norm_pred = float(output[0][1])            # [0, 1]
        target_speed = speed_norm_pred * MAX_SPEED       # m/s

        # 4. 1/r → 조향 명령 변환 (Ackermann)
        steering_angle = math.atan(inv_radius * WHEELBASE)
        steering_cmd = steering_angle / MAX_STEER_ANGLE
        steering_cmd = max(-1.0, min(1.0, steering_cmd))

        # 5. PID 제어: 목표 속도 → throttle/brake
        speed_error = target_speed - self.current_speed
        now_sec = self.get_clock().now().nanoseconds / 1e9
        pid_output = self.pid.compute(speed_error, now_sec)

        cmd_msg = ControlCommand()
        cmd_msg.steering = steering_cmd

        if pid_output >= 0.0:
            cmd_msg.throttle = pid_output
            cmd_msg.brake = 0.0
        else:
            cmd_msg.throttle = 0.0
            cmd_msg.brake = min(-pid_output, 1.0)

        self.cmd_publisher.publish(cmd_msg)

        # 터미널에 현재 예측값 출력
        self.get_logger().info(
            f'1/r: {inv_radius:.4f} | Steer: {steering_cmd:.3f} | '
            f'Target: {target_speed:.1f} m/s | '
            f'Current: {self.current_speed:.1f} m/s | '
            f'Throttle: {cmd_msg.throttle:.3f} | Brake: {cmd_msg.brake:.3f}'
        )

        # Visual Backpropagation 시각화 결과창 띄우기
        try:
            # 크롭된 이미지 크기 66x200에 맞게 리사이즈
            cam1_resized = cv2.resize(cam1_cropped, (200, 66))
            
            # 모델 출력 마스크를 히트맵으로 적용
            heatmap = np.uint8(255 * vbp_mask)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # 크롭된 이미지(캠1)와 히트맵 합성
            overlay = cv2.addWeighted(cam1_resized, 0.5, heatmap_color, 0.5, 0)
            
            # 창 크기를 보기 좋게 확대 (원본 3배)
            overlay_show = cv2.resize(overlay, (600, 198))
            
            cv2.imshow('Visual Backpropagation', overlay_show)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warn(f'Visual Backprop 시각화 오류: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = E2EInferenceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("자율주행을 종료합니다.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()