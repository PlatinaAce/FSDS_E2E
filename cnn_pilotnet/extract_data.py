import os
import csv
import math
import cv2
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import get_typestore, Stores
from rosbags.typesys.store import Nodetype

EARTH_RADIUS = 6378137.0  # WGS84 (m)

# 차량 파라미터 (1/r 변환용)
WHEELBASE = 1.55          # 축거 (m)
MAX_STEER_ANGLE = 0.5    # 최대 조향각 (rad, ~28.6°)
MAX_INV_RADIUS = math.tan(MAX_STEER_ANGLE) / WHEELBASE  # 최대 역회전반경

def compute_speed_from_gps(prev_lat, prev_lon, curr_lat, curr_lon, dt):
    """연속 GPS 좌표로부터 지면 속도(m/s) 계산"""
    if dt <= 0:
        return 0.0
    lat1 = math.radians(prev_lat)
    lat2 = math.radians(curr_lat)
    dlon = math.radians(curr_lon - prev_lon)
    dlat = lat2 - lat1
    dx = EARTH_RADIUS * math.cos((lat1 + lat2) / 2.0) * dlon
    dy = EARTH_RADIUS * dlat
    return math.sqrt(dx * dx + dy * dy) / dt

def main():
    # 설정 경로 (본인의 환경에 맞게 수정하세요)
    bag_path = '/home/ace/fsds_dataset/rosbag/bag_20260313_203644' # 녹화한 bag 파일 폴더 경로
    bag_name = os.path.basename(bag_path)
    output_dir = os.path.join('/home/ace/fsds_dataset/training_data', bag_name)  # 이미지가 저장될 폴더
    csv_filename = os.path.join(output_dir, 'dataset.csv')

    # 필요한 토픽 이름 설정
    topic_cam1 = '/fsds/cam1/image_color'
    topic_cam2 = '/fsds/cam2/image_color'
    topic_control = '/fsds/control_command'
    topic_gps = '/fsds/gps'

    # TypeStore 생성 (rosbags 0.11+ API)
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # fs_msgs 커스텀 메시지 타입 등록
    typestore.register({
        'fs_msgs/msg/ControlCommand': (
            [],
            [
                ('header', (Nodetype.NAME, 'std_msgs/msg/Header')),
                ('throttle', (Nodetype.BASE, ('float64', 0))),
                ('steering', (Nodetype.BASE, ('float64', 0))),
                ('brake', (Nodetype.BASE, ('float64', 0))),
            ],
        ),
    })

    # 이미지 저장 폴더 생성 (cam1, cam2 별도)
    cam1_dir = os.path.join(output_dir, 'images', 'cam1')
    cam2_dir = os.path.join(output_dir, 'images', 'cam2')
    os.makedirs(cam1_dir, exist_ok=True)
    os.makedirs(cam2_dir, exist_ok=True)

    # 데이터 저장을 위한 변수들
    latest_control = None
    latest_cam2 = None    # 최신 cam2 이미지 버퍼
    prev_gps = None      # 이전 GPS 메시지 (속도 계산용)
    latest_speed = 0.0   # GPS로부터 계산한 최신 속도 (m/s)
    image_count = 0

    print(f"Bag 파일 읽기를 시작합니다: {bag_path}")

    # Bag 파일 열기
    with Reader(bag_path) as reader:
        # bag 안에 있는 토픽 목록 출력
        topics_in_bag = [c.topic for c in reader.connections]
        print(f"Bag에 있는 토픽: {topics_in_bag}")

        if topic_cam1 not in topics_in_bag:
            print(f"[오류] 카메라 토픽 '{topic_cam1}'이(가) bag에 없습니다!")
            print(f"  사용 가능한 토픽: {topics_in_bag}")
            return
        if topic_cam2 not in topics_in_bag:
            print(f"[경고] 카메라 토픽 '{topic_cam2}'이(가) bag에 없습니다!")
            print(f"  cam2 없이 진행할 수 없습니다.")
            return

        # csv 파일 준비
        with open(csv_filename, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # CSV 헤더 작성 (cam1 이미지, cam2 이미지, 역회전반경(1/r), 속도(m/s))
            csv_writer.writerow(['cam1_image', 'cam2_image', 'inverse_radius', 'speed'])

            # 메시지를 시간 순서대로 하나씩 읽기
            for connection, timestamp, rawdata in reader.messages():
                # 1. 제어 명령 토픽일 경우: 최신 조향값 → 1/r로 변환
                if connection.topic == topic_control:
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    # steering [-1,1] → 실제 각도 → 1/r → 정규화
                    steer_angle = msg.steering * MAX_STEER_ANGLE
                    inv_r = math.tan(steer_angle) / WHEELBASE
                    inv_r_norm = max(-1.0, min(1.0, inv_r / MAX_INV_RADIUS))
                    latest_control = {'inverse_radius': inv_r_norm}

                # 2. GPS 토픽일 경우: 연속 좌표로 속도 계산
                elif connection.topic == topic_gps:
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    if prev_gps is not None:
                        dt = (timestamp - prev_gps['time']) / 1e9  # ns → s
                        latest_speed = compute_speed_from_gps(
                            prev_gps['lat'], prev_gps['lon'],
                            msg.latitude, msg.longitude, dt)
                    prev_gps = {'lat': msg.latitude, 'lon': msg.longitude, 'time': timestamp}

                # 3. cam2 토픽일 경우: 최신 cam2 이미지 버퍼링
                elif connection.topic == topic_cam2:
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    img_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                    if img_data.shape[2] == 4:
                        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
                    latest_cam2 = img_data

                # 4. cam1 토픽일 경우: cam1+cam2 이미지 저장 및 라벨 매칭
                elif connection.topic == topic_cam1:
                    if latest_control is None or latest_cam2 is None:
                        continue  # 제어값 또는 cam2가 아직 없으면 건너뜀

                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

                    # cam1 이미지 변환
                    cam1_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                    if cam1_data.shape[2] == 4:
                        cam1_data = cv2.cvtColor(cam1_data, cv2.COLOR_BGRA2BGR)

                    # 이미지 파일로 저장
                    img_filename = f"{image_count:06d}.jpg"
                    cv2.imwrite(os.path.join(cam1_dir, img_filename), cam1_data)
                    cv2.imwrite(os.path.join(cam2_dir, img_filename), latest_cam2)

                    # CSV에 행 추가 (cam1, cam2, inverse_radius, speed)
                    csv_writer.writerow([img_filename, img_filename,
                                         latest_control['inverse_radius'], latest_speed])

                    image_count += 1
                    if image_count % 500 == 0:
                        print(f"{image_count}장의 이미지 쌍 추출 완료...")

    print(f"\n데이터 추출이 완료되었습니다!")
    print(f"총 {image_count}개의 데이터 쌍이 {output_dir} 에 저장되었습니다.")

if __name__ == '__main__':
    main()