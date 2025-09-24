#!/usr/bin/env python3
"""
簡易アイトラッキングアプリケーション
OpenCV + MediaPipeを使用したクロスプラットフォーム版（Windows/macOS/Linux対応）
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
import json
import os
import sys
import platform
from datetime import datetime
from collections import deque

class SimpleEyeTracker:
    def __init__(self):
        # プラットフォーム情報を取得
        self.platform_name = platform.system()
        print(f"Running on {self.platform_name} ({platform.release()})")
        
        # MediaPipe設定（高速化のため精度を調整）
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,  # 検出感度を下げて速度向上
            min_tracking_confidence=0.5   # 追跡感度を下げて速度向上
        )
        
        # 人体検出用のFace Detection（高速化のため精度を調整）
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5  # 検出感度を下げて速度向上
        )
        
        # カメラ設定（プラットフォーム対応）
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: カメラが開けません。別のカメラインデックスを試してください。")
                # Windows/Linux: DirectShow/V4L2バックエンドを試行
                if self.platform_name == 'Windows':
                    self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                elif self.platform_name == 'Linux':
                    self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 60)  # フレームレート向上
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # バッファサイズ最小化
            
        except Exception as e:
            print(f"カメラ初期化エラー: {e}")
            sys.exit(1)
        
        # 目の重要な特徴点インデックス
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # 虹彩の特徴点インデックス（精度向上のため追加特徴点を含む）
        self.LEFT_IRIS_INDICES = [474, 475, 476, 477, 468, 469, 470, 471, 472]
        self.RIGHT_IRIS_INDICES = [469, 470, 471, 472, 473, 474, 475, 476, 477]
        
        # 手の検出用MediaPipe（高速化のため最低限の設定）
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,  # 検出閾値を下げて高速化
            min_tracking_confidence=0.3,   # 追跡閾値を下げて高速化
            model_complexity=0              # 軽量モデルを使用
        )
        
        # ポーズ検出用MediaPipe（高速化のため最低限の設定）
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=0  # 軽量モデルを使用
        )
        
        self.person_detected = False
        self.current_gaze_point = (0, 0)
        self.eye_center_point = (0, 0)
        
        # 小窓設定
        self.minimap_width = 220
        self.minimap_height = 140
        self.minimap_x = 0  # 動的に設定
        self.minimap_y = 0  # 動的に設定
        
        # ヒートマップデータ管理
        self.heatmap_data = np.zeros((self.minimap_height, self.minimap_width), dtype=np.float64)
        self.gaze_history_buffer = deque(maxlen=1000)  # 最新1000ポイントを保持
        self.heatmap_decay_rate = 0.995  # ヒートマップの減衰率
        self.heatmap_intensity = 2.0  # ヒートマップの強度
        self.show_heatmap = True
        
        # ログ管理
        self.gaze_log = []
        self.session_start_time = datetime.now()
        self.log_interval = 0.1  # 0.1秒ごとにログ記録
        self.last_log_time = 0
        
        # スムージング用のバッファとパラメータ（より滑らかに調整）
        self.gaze_history = []
        self.gaze_history_size = 6  # 過去6フレームに削減して応答性向上
        self.smoothing_factor = 0.25  # スムージングを軽減して画面下部への反応性向上
        self.smoothed_gaze_point = (0, 0)
        self.smoothed_eye_center = (0, 0)
        
        # アダプティブスムージング（滑らかさ重視）
        self.adaptive_smoothing = True
        self.motion_threshold = 200  # 閾値を下げて画面下部への動きを素早く検出
        self.prev_gaze_point = (0, 0)
        self.high_speed_mode = False
        
        # ゲーム機能設定
        self.game_mode = True
        self.targets = []
        self.target_states = {}
        self.gaze_timer = {}
        self.gaze_threshold = 90  # 視線がターゲットに近いと判定する距離（数字認識用に微調整）
        self.gaze_duration = 0.8  # 色変更に必要な視線停留時間（数字認識用に短縮）
        self.next_target_number = 1  # 次にクリックすべき数字
        self.game_cleared = False  # ゲームクリア状態
        self.clear_message_timer = 0  # CLEARメッセージ表示用タイマー
        self.target_completion_times = {}  # 各ターゲットの完了時間を保存
        self.target_individual_timers = {}  # 各ターゲット個別のタイマー
        self.target_individual_states = {}  # 各ターゲット個別の状態
        self.individual_next_target = 1  # 個別カウント用の次のターゲット番号
        self.setup_game_targets()
        
    def get_eye_center(self, landmarks, eye_indices):
        eye_points = []
        for idx in eye_indices:
            if idx < len(landmarks):
                eye_points.append([landmarks[idx].x, landmarks[idx].y])
        
        if eye_points:
            eye_center = np.mean(eye_points, axis=0)
            return eye_center
        return None
    
    def get_iris_center(self, landmarks, iris_indices):
        iris_points = []
        for idx in iris_indices:
            if idx < len(landmarks):
                iris_points.append([landmarks[idx].x, landmarks[idx].y])
        
        if iris_points:
            iris_center = np.mean(iris_points, axis=0)
            return iris_center
        return None
    
    def calculate_gaze_direction(self, left_eye_center, right_eye_center, left_iris_center, right_iris_center):
        if (left_eye_center is None or right_eye_center is None or 
            left_iris_center is None or right_iris_center is None):
            return None
            
        left_gaze_vector = left_iris_center - left_eye_center
        right_gaze_vector = right_iris_center - right_eye_center
        
        avg_gaze_vector = (left_gaze_vector + right_gaze_vector) / 2
        
        # PC画面上の数字認識に最適化された感度設定
        # X軸（水平）の感度を数字認識に最適化
        gaze_x = avg_gaze_vector[0] * 1200  # 水平感度を向上させて細かい数字の区別を可能に
        
        # Y軸（垂直）の感度を大幅に向上（画面下部の精度向上）
        gaze_y = avg_gaze_vector[1] * 1400  # 垂直感度を大幅向上（1000→1400）画面下部対応
        
        # 数字認識用の線形補正（より直接的な応答）
        gaze_x_sign = 1 if gaze_x >= 0 else -1
        gaze_y_sign = 1 if gaze_y >= 0 else -1
        
        # 数字認識のため線形性を重視（指数補正を軽減）
        gaze_x = gaze_x_sign * math.pow(abs(gaze_x), 0.9)  # より線形な応答で精密制御
        gaze_y = gaze_y_sign * math.pow(abs(gaze_y), 0.9)
        
        return (gaze_x, gaze_y)
    
    def smooth_gaze_data(self, gaze_direction, eye_center_screen):
        """視線データをスムージングして急激な動きを抑制（アダプティブスムージング）"""
        if gaze_direction is None or eye_center_screen is None:
            return None, None
        
        # 現在のデータを履歴に追加
        current_data = {
            'gaze': gaze_direction,
            'eye_center': eye_center_screen
        }
        
        self.gaze_history.append(current_data)
        
        # 履歴サイズを制限
        if len(self.gaze_history) > self.gaze_history_size:
            self.gaze_history.pop(0)
        
        # 急激な動きの検出
        if self.adaptive_smoothing and self.prev_gaze_point != (0, 0):
            gaze_change = np.sqrt((gaze_direction[0] - self.prev_gaze_point[0])**2 + 
                                 (gaze_direction[1] - self.prev_gaze_point[1])**2)
            
            if gaze_change > self.motion_threshold:
                # 急激な動き（特に画面下部への移動）に対応
                self.high_speed_mode = True
                effective_smoothing = self.smoothing_factor * 0.4  # スムージングを大幅軽減（0.7→0.4）
                effective_history_size = max(2, self.gaze_history_size // 3)  # 履歴サイズをさらに削減
            else:
                # 通常の動きの場合は強いスムージング
                self.high_speed_mode = False
                effective_smoothing = self.smoothing_factor
                effective_history_size = self.gaze_history_size
        else:
            effective_smoothing = self.smoothing_factor
            effective_history_size = self.gaze_history_size
        
        # 移動平均を計算（履歴サイズを動的に調整）
        if len(self.gaze_history) >= 2:
            history_to_use = self.gaze_history[-effective_history_size:]
            avg_gaze_x = sum(data['gaze'][0] for data in history_to_use) / len(history_to_use)
            avg_gaze_y = sum(data['gaze'][1] for data in history_to_use) / len(history_to_use)
            avg_eye_center_x = sum(data['eye_center'][0] for data in history_to_use) / len(history_to_use)
            avg_eye_center_y = sum(data['eye_center'][1] for data in history_to_use) / len(history_to_use)
            
            # 指数移動平均を適用してさらにスムージング
            self.smoothed_gaze_point = (
                effective_smoothing * self.smoothed_gaze_point[0] + (1 - effective_smoothing) * avg_gaze_x,
                effective_smoothing * self.smoothed_gaze_point[1] + (1 - effective_smoothing) * avg_gaze_y
            )
            
            self.smoothed_eye_center = (
                int(effective_smoothing * self.smoothed_eye_center[0] + (1 - effective_smoothing) * avg_eye_center_x),
                int(effective_smoothing * self.smoothed_eye_center[1] + (1 - effective_smoothing) * avg_eye_center_y)
            )
        else:
            # 最初のフレームではそのまま使用
            self.smoothed_gaze_point = gaze_direction
            self.smoothed_eye_center = eye_center_screen
        
        # 現在の視線を保存
        self.prev_gaze_point = gaze_direction
        
        return self.smoothed_gaze_point, self.smoothed_eye_center
    
    def draw_eye_contours(self, frame, landmarks, w, h):
        # 高速化のため、フレームをスキップして描画
        if hasattr(self, 'draw_counter'):
            self.draw_counter += 1
        else:
            self.draw_counter = 0
        
        # 2フレームに1回だけ描画
        if self.draw_counter % 2 != 0:
            return
        
        # 左目の輪郭を描画
        left_eye_points = []
        for idx in self.LEFT_EYE_INDICES:
            if idx < len(landmarks):
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                left_eye_points.append((x, y))
        
        if left_eye_points:
            cv2.polylines(frame, [np.array(left_eye_points)], True, (255, 0, 0), 1)  # 線の太さを減らす
        
        # 右目の輪郭を描画
        right_eye_points = []
        for idx in self.RIGHT_EYE_INDICES:
            if idx < len(landmarks):
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                right_eye_points.append((x, y))
        
        if right_eye_points:
            cv2.polylines(frame, [np.array(right_eye_points)], True, (255, 0, 0), 1)  # 線の太さを減らす
    
    def draw_iris_points(self, frame, landmarks, w, h):
        # 左目の虹彩を描画
        for idx in self.LEFT_IRIS_INDICES:
            if idx < len(landmarks):
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
        
        # 右目の虹彩を描画
        for idx in self.RIGHT_IRIS_INDICES:
            if idx < len(landmarks):
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
    
    def detect_hands_and_pose(self, frame_rgb):
        # 手の検出
        hands_results = self.hands.process(frame_rgb)
        hand_landmarks = []
        
        if hands_results.multi_hand_landmarks:
            for hand_landmark in hands_results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmark.landmark:
                    landmarks.append((lm.x, lm.y))
                hand_landmarks.append(landmarks)
        
        # ポーズ検出
        pose_results = self.pose.process(frame_rgb)
        pose_landmarks = []
        
        if pose_results.pose_landmarks:
            for lm in pose_results.pose_landmarks.landmark:
                pose_landmarks.append((lm.x, lm.y))
        
        return hand_landmarks, pose_landmarks
    
    def draw_hands_and_pose(self, frame, hand_landmarks, pose_landmarks):
        h, w, _ = frame.shape
        
        # 手の描画
        for hand in hand_landmarks:
            for i, (x, y) in enumerate(hand):
                px, py = int(x * w), int(y * h)
                cv2.circle(frame, (px, py), 5, (255, 0, 255), -1)
                if i == 8:  # 人差し指の先端
                    cv2.circle(frame, (px, py), 10, (0, 255, 255), 2)
                    cv2.putText(frame, 'Index', (px + 10, py - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 重要なポーズポイントの描画
        if pose_landmarks:
            # 鼻
            nose_x, nose_y = int(pose_landmarks[0][0] * w), int(pose_landmarks[0][1] * h)
            cv2.circle(frame, (nose_x, nose_y), 8, (0, 0, 255), -1)
            cv2.putText(frame, 'Nose', (nose_x + 10, nose_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 肩
            if len(pose_landmarks) > 11:
                left_shoulder_x = int(pose_landmarks[11][0] * w)
                left_shoulder_y = int(pose_landmarks[11][1] * h)
                cv2.circle(frame, (left_shoulder_x, left_shoulder_y), 6, (255, 165, 0), -1)
            
            if len(pose_landmarks) > 12:
                right_shoulder_x = int(pose_landmarks[12][0] * w)
                right_shoulder_y = int(pose_landmarks[12][1] * h)
                cv2.circle(frame, (right_shoulder_x, right_shoulder_y), 6, (255, 165, 0), -1)
    
    def calculate_gaze_target(self, gaze_direction, eye_center_screen, hand_landmarks, pose_landmarks, frame_shape):
        if not gaze_direction:
            return None, None
        
        h, w, _ = frame_shape
        
        # 視線の延長線を計算（数字認識精度向上のためスケール調整）
        gaze_scale = 180  # 数字認識のためスケールを拡大
        gaze_end_x = int(eye_center_screen[0] + gaze_direction[0] * gaze_scale)
        gaze_end_y = int(eye_center_screen[1] + gaze_direction[1] * gaze_scale)
        
        # 画面境界内にクランプ（画面下部の有効領域を拡大）
        gaze_end_x = max(20, min(w - 20, gaze_end_x))  # 左右マージンを縮小（50→20）
        gaze_end_y = max(10, min(h - 10, gaze_end_y))  # 上下マージンを大幅縮小（50→10）画面下部対応
        
        closest_target = None
        closest_distance = float('inf')
        target_type = None
        
        # 手との距離をチェック
        for hand in hand_landmarks:
            for i, (x, y) in enumerate(hand):
                px, py = int(x * w), int(y * h)
                distance = np.sqrt((gaze_end_x - px)**2 + (gaze_end_y - py)**2)
                
                if distance < closest_distance and distance < 100:  # 100ピクセル以内
                    closest_distance = distance
                    closest_target = (px, py)
                    if i == 8:  # 人差し指の先端
                        target_type = "人差し指"
                    elif i == 4:  # 親指の先端
                        target_type = "親指"
                    else:
                        target_type = "手"
        
        # ポーズランドマークとの距離をチェック
        if pose_landmarks:
            # 鼻
            nose_x, nose_y = int(pose_landmarks[0][0] * w), int(pose_landmarks[0][1] * h)
            distance = np.sqrt((gaze_end_x - nose_x)**2 + (gaze_end_y - nose_y)**2)
            if distance < closest_distance and distance < 80:
                closest_distance = distance
                closest_target = (nose_x, nose_y)
                target_type = "鼻"
            
            # 肩
            if len(pose_landmarks) > 11:
                left_shoulder_x = int(pose_landmarks[11][0] * w)
                left_shoulder_y = int(pose_landmarks[11][1] * h)
                distance = np.sqrt((gaze_end_x - left_shoulder_x)**2 + (gaze_end_y - left_shoulder_y)**2)
                if distance < closest_distance and distance < 80:
                    closest_distance = distance
                    closest_target = (left_shoulder_x, left_shoulder_y)
                    target_type = "左肩"
            
            if len(pose_landmarks) > 12:
                right_shoulder_x = int(pose_landmarks[12][0] * w)
                right_shoulder_y = int(pose_landmarks[12][1] * h)
                distance = np.sqrt((gaze_end_x - right_shoulder_x)**2 + (gaze_end_y - right_shoulder_y)**2)
                if distance < closest_distance and distance < 80:
                    closest_distance = distance
                    closest_target = (right_shoulder_x, right_shoulder_y)
                    target_type = "右肩"
        
        return closest_target, target_type
    
    def setup_game_targets(self):
        """ゲーム用の数字ターゲットを設定"""
        self.targets = [
            {"number": 1, "x": 150, "y": 150, "color": (0, 0, 255), "original_color": (0, 0, 255)},  # 左上
            {"number": 2, "x": 1100, "y": 100, "color": (0, 0, 255), "original_color": (0, 0, 255)}   # 右上端
        ]
        
        for target in self.targets:
            self.target_states[target["number"]] = "normal"  # normal, gazing, gazed
            self.gaze_timer[target["number"]] = 0
            self.target_completion_times[target["number"]] = 0  # 完了時間を初期化
            self.target_individual_timers[target["number"]] = 0  # 個別タイマーを初期化
            self.target_individual_states[target["number"]] = "normal"  # normal, gazing, completed
    
    def toggle_game_mode(self):
        """ゲームモードの切り替え"""
        self.game_mode = not self.game_mode
        if self.game_mode:
            # ゲームモード開始時にターゲットをリセット
            self.next_target_number = 1  # 1から始める
            self.game_cleared = False  # ゲームクリア状態をリセット
            self.clear_message_timer = 0  # CLEARメッセージタイマーをリセット
            self.target_completion_times = {}  # 完了時間をリセット
            self.target_individual_timers = {}  # 個別タイマーをリセット
            self.target_individual_states = {}  # 個別状態をリセット
            self.individual_next_target = 1  # 個別カウント用の次のターゲット番号をリセット
            self.setup_game_targets()
            print("ゲームモード開始! 数字1から順番に見つめてください")
        else:
            print("通常モードに戻りました")
    
    def draw_game_targets(self, frame):
        """ゲーム用ターゲットを描画"""
        if not self.game_mode:
            return
        
        # ゲームクリア時のCLEARメッセージ表示
        if self.game_cleared:
            h, w, _ = frame.shape
            current_time = time.time()
            
            # CLEARメッセージを画面中央に大きく表示
            clear_text = "CLEAR!"
            font_scale = 4.0
            thickness = 8
            text_size = cv2.getTextSize(clear_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            
            # 点滅効果（0.5秒間隔）
            blink_interval = 0.5
            time_since_clear = current_time - self.clear_message_timer
            if int(time_since_clear / blink_interval) % 2 == 0:
                # 黒い背景で文字を目立たせる
                cv2.rectangle(frame, (text_x - 20, text_y - text_size[1] - 20), 
                             (text_x + text_size[0] + 20, text_y + 20), (0, 0, 0), -1)
                # 金色でCLEARを表示
                cv2.putText(frame, clear_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 215, 255), thickness)
        
        for target in self.targets:
            # 大きな円を描画（ターゲットをさらに大きく）
            cv2.circle(frame, (target["x"], target["y"]), 70, target["color"], -1)  # 50→70に増大
            cv2.circle(frame, (target["x"], target["y"]), 70, (255, 255, 255), 3)  # 白い枠（太く）
            
            # ターゲット状態による追加表示
            if self.target_states[target["number"]] == "gazing":
                # 狙っている時のエフェクト
                cv2.circle(frame, (target["x"], target["y"]), 80, (255, 255, 0), 3)  # 黄色のリングも大きく
            
            # 数字をさらに大きく描画
            font_scale = 2.0  # 文字をさらに大きく（1.5→2.0）
            thickness = 4     # 文字をさらに太く（3→4）
            text = str(target["number"])
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = target["x"] - text_size[0] // 2
            text_y = target["y"] + text_size[1] // 2
            
            cv2.putText(frame, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            # 個別カウント中または完了後の時間を数字の上に1つだけ表示
            if (target["number"] in self.target_individual_states and 
                (self.target_individual_states[target["number"]] == "gazing" or 
                 self.target_individual_states[target["number"]] == "completed") and
                target["number"] in self.target_completion_times):
                
                display_time = self.target_completion_times[target["number"]]
                # 現在の計測時間を表示
                time_text = f"{display_time:.1f}s"
                time_text_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                time_text_x = target["x"] - time_text_size[0] // 2
                time_text_y = target["y"] - 80
                
                # 背景色を状態に応じて変更
                if self.target_individual_states[target["number"]] == "completed":
                    border_color = (0, 255, 0)  # 緑色（完了）
                else:
                    border_color = (255, 255, 0)  # 黄色（計測中）
                
                # 時間の背景を描画
                cv2.rectangle(frame, (time_text_x - 5, time_text_y - 20), 
                             (time_text_x + time_text_size[0] + 5, time_text_y + 5), (0, 0, 0), -1)
                cv2.rectangle(frame, (time_text_x - 5, time_text_y - 20), 
                             (time_text_x + time_text_size[0] + 5, time_text_y + 5), border_color, 2)
                
                # 時間テキストを表示
                cv2.putText(frame, time_text, (time_text_x, time_text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def check_gaze_line_intersection(self, eye_center, gaze_end, target_x, target_y, target_radius):
        """視線の線がターゲットの円と交差しているかチェック"""
        # 線分と円の交差判定
        # 点から線分への最短距離を計算
        
        # ベクトル計算
        line_vec_x = gaze_end[0] - eye_center[0]
        line_vec_y = gaze_end[1] - eye_center[1]
        point_vec_x = target_x - eye_center[0]
        point_vec_y = target_y - eye_center[1]
        
        # 線分の長さの二乗
        line_len_sq = line_vec_x * line_vec_x + line_vec_y * line_vec_y
        
        if line_len_sq == 0:
            # 線分の長さが0の場合は点との距離
            distance = math.sqrt(point_vec_x * point_vec_x + point_vec_y * point_vec_y)
            return distance <= target_radius
        
        # 線分上の最近点のパラメータt (0 <= t <= 1)
        t = max(0, min(1, (point_vec_x * line_vec_x + point_vec_y * line_vec_y) / line_len_sq))
        
        # 線分上の最近点
        closest_x = eye_center[0] + t * line_vec_x
        closest_y = eye_center[1] + t * line_vec_y
        
        # 最近点からターゲット中心までの距離
        distance = math.sqrt((closest_x - target_x) ** 2 + (closest_y - target_y) ** 2)
        
        return distance <= target_radius
    
    def update_heatmap(self, gaze_x, gaze_y, screen_width, screen_height):
        """ヒートマップデータを更新"""
        # メイン画面座標を小窓座標に変換
        minimap_x = int((gaze_x / screen_width) * self.minimap_width)
        minimap_y = int((gaze_y / screen_height) * self.minimap_height)
        
        # 境界内に制限
        minimap_x = max(0, min(self.minimap_width-1, minimap_x))
        minimap_y = max(0, min(self.minimap_height-1, minimap_y))
        
        # ガウシアンブラーを適用してスムーズなヒートマップを作成
        radius = 8
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                px = minimap_x + dx
                py = minimap_y + dy
                
                if 0 <= px < self.minimap_width and 0 <= py < self.minimap_height:
                    distance = np.sqrt(dx*dx + dy*dy)
                    if distance <= radius:
                        # ガウシアン分布による重み付け
                        weight = np.exp(-(distance*distance) / (2 * (radius/3)**2))
                        self.heatmap_data[py, px] += self.heatmap_intensity * weight
        
        # 履歴バッファに追加
        current_time = time.time()
        self.gaze_history_buffer.append({
            'timestamp': current_time,
            'x': gaze_x,
            'y': gaze_y,
            'minimap_x': minimap_x,
            'minimap_y': minimap_y
        })
    
    def decay_heatmap(self):
        """ヒートマップの減衰処理"""
        self.heatmap_data *= self.heatmap_decay_rate
        # 最小値を設定してゼロに近い値をクリア
        self.heatmap_data[self.heatmap_data < 0.01] = 0
    
    def log_gaze_data(self, gaze_x, gaze_y, eye_center_x, eye_center_y):
        """視線データをログに記録"""
        current_time = time.time()
        
        if current_time - self.last_log_time >= self.log_interval:
            timestamp = datetime.now()
            session_time = (timestamp - self.session_start_time).total_seconds()
            
            log_entry = {
                'timestamp': timestamp.isoformat(),
                'session_time': round(session_time, 3),
                'gaze_x': int(gaze_x),
                'gaze_y': int(gaze_y),
                'eye_center_x': int(eye_center_x),
                'eye_center_y': int(eye_center_y)
            }
            
            self.gaze_log.append(log_entry)
            self.last_log_time = current_time
    
    def save_gaze_log(self, filename=None):
        """視線ログをJSONファイルに保存（プラットフォーム対応）"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gaze_log_{timestamp}.json"
        
        # プラットフォーム情報を追加
        log_data = {
            'session_info': {
                'start_time': self.session_start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_entries': len(self.gaze_log),
                'platform': {
                    'system': self.platform_name,
                    'release': platform.release(),
                    'python_version': platform.python_version()
                },
                'screen_resolution': {
                    'width': 1280,
                    'height': 720
                }
            },
            'gaze_data': self.gaze_log
        }
        
        try:
            # ディレクトリの存在確認と作成
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            print(f"視線ログが保存されました: {filename}")
            return filename
        except Exception as e:
            print(f"ログ保存エラー: {e}")
            return None
    
    def toggle_heatmap_display(self):
        """ヒートマップ表示の切り替え"""
        self.show_heatmap = not self.show_heatmap
        print(f"ヒートマップ表示: {'ON' if self.show_heatmap else 'OFF'}")
    
    def clear_heatmap(self):
        """ヒートマップをクリア"""
        self.heatmap_data.fill(0)
        self.gaze_history_buffer.clear()
        print("ヒートマップをクリアしました")

    def check_gaze_on_targets(self, eye_center, gaze_end_x, gaze_end_y):
        """視線の矢印の動線がターゲットに交差しているかチェック（厳密な順番制約付き）"""
        if not self.game_mode:
            return
        
        current_time = time.time()
        
        for target in self.targets:
            # 視線の線がターゲットと交差しているかチェック
            is_intersecting = self.check_gaze_line_intersection(
                eye_center, (gaze_end_x, gaze_end_y), 
                target["x"], target["y"], self.gaze_threshold
            )
            
            if is_intersecting:
                # 厳密な順番制約：次にクリックすべき数字と完全に一致する場合のみ処理
                if target["number"] == self.next_target_number:
                    if self.target_states[target["number"]] == "normal":
                        self.gaze_timer[target["number"]] = current_time
                        self.target_states[target["number"]] = "gazing"
                    elif self.target_states[target["number"]] == "gazing":
                        elapsed_time = current_time - self.gaze_timer[target["number"]]
                        if elapsed_time >= self.gaze_duration and self.target_states[target["number"]] != "gazed":
                            # 十分な時間矢印が通過した - 色を変更（順番が正しい場合のみ）
                            target["color"] = (0, 255, 0)  # 緑色に変更
                            self.target_states[target["number"]] = "gazed"
                            print(f"ターゲット {target['number']} がクリアされました！（{elapsed_time:.1f}秒）")
                            # 次のターゲット番号を設定
                            self.next_target_number += 1
                            if self.next_target_number > 2:
                                self.game_cleared = True
                                self.clear_message_timer = time.time()
                                print("おめでとうございます！全てのターゲットをクリアしました！")
                        
                        # 1秒以降も継続して時間を更新
                        if elapsed_time >= 1.0:
                            self.target_completion_times[target["number"]] = elapsed_time
                
                # 各ターゲットの個別カウント処理（順番制約付き）
                if (target["number"] == self.individual_next_target and 
                    target["number"] in self.target_individual_states):
                    if self.target_individual_states[target["number"]] == "normal":
                        # 個別タイマー開始
                        self.target_individual_timers[target["number"]] = current_time
                        self.target_individual_states[target["number"]] = "gazing"
                    elif self.target_individual_states[target["number"]] == "gazing":
                        # 個別時間計測中
                        individual_elapsed = current_time - self.target_individual_timers[target["number"]]
                        if individual_elapsed >= 1.0 and self.target_individual_states[target["number"]] != "completed":
                            # 1秒に到達した時点で完了状態にする
                            self.target_individual_states[target["number"]] = "completed"
                            print(f"ターゲット {target['number']} の個別計測完了！（{individual_elapsed:.1f}秒）")
                            # 次の個別ターゲット番号を設定
                            self.individual_next_target += 1
                        # 時間を継続更新（3秒前後問わず）
                        self.target_completion_times[target["number"]] = individual_elapsed
                    elif self.target_individual_states[target["number"]] == "completed":
                        # 完了状態でも見続けている場合は時間を更新し続ける
                        individual_elapsed = current_time - self.target_individual_timers[target["number"]]
                        self.target_completion_times[target["number"]] = individual_elapsed
                    # completed状態の場合は何もしない（再度見ても反応しない）
                elif target["number"] != self.next_target_number:
                    # 順番が違う数字を見た場合：何も起こらない
                    # gazing状態をリセットして間違った数字の進行を防ぐ
                    if target["number"] in self.target_states and self.target_states[target["number"]] == "gazing":
                        self.target_states[target["number"]] = "normal"
            else:
                # ターゲットから視線の矢印が外れた
                if target["number"] in self.target_states and self.target_states[target["number"]] == "gazing":
                    self.target_states[target["number"]] = "normal"
                
                # 個別カウントもリセット（completed状態でない場合のみ）
                if (target["number"] in self.target_individual_states and 
                    self.target_individual_states[target["number"]] == "gazing"):
                    self.target_individual_states[target["number"]] = "normal"
    
    def get_gaze_screen_position(self, gaze_direction, eye_center_screen):
        """視線方向を画面座標に変換"""
        if not gaze_direction or not eye_center_screen:
            return None
        
        # 視線方向を画面座標にマッピング
        gaze_scale = 0.3  # スケールを調整
        gaze_x = int(eye_center_screen[0] + gaze_direction[0] * gaze_scale)
        gaze_y = int(eye_center_screen[1] + gaze_direction[1] * gaze_scale)
        
        return (gaze_x, gaze_y)
    
    def draw_minimap(self, frame, eye_center_screen, gaze_direction):
        """右下に小窓を描画して視線位置とヒートマップを表示"""
        h, w, _ = frame.shape
        
        # 小窓の位置を右下に設定
        self.minimap_x = w - self.minimap_width - 20
        self.minimap_y = h - self.minimap_height - 20
        
        # 小窓の背景を描画（ダークテーマ）
        minimap_bg = frame[self.minimap_y:self.minimap_y + self.minimap_height, 
                          self.minimap_x:self.minimap_x + self.minimap_width].copy()
        overlay = minimap_bg.copy()
        cv2.rectangle(overlay, (0, 0), (self.minimap_width, self.minimap_height), (30, 30, 30), -1)
        minimap_bg = cv2.addWeighted(minimap_bg, 0.2, overlay, 0.8, 0)
        
        # ヒートマップを描画（表示がONの場合）
        if self.show_heatmap and np.max(self.heatmap_data) > 0:
            # ヒートマップの減衰処理
            self.decay_heatmap()
            
            # ヒートマップを正規化してカラーマップに変換
            normalized_heatmap = np.clip(self.heatmap_data / np.max(self.heatmap_data), 0, 1)
            
            # カラーマップを適用（HOT colormap）
            heatmap_colored = np.zeros((self.minimap_height, self.minimap_width, 3), dtype=np.uint8)
            
            for y in range(self.minimap_height):
                for x in range(self.minimap_width):
                    intensity = normalized_heatmap[y, x]
                    if intensity > 0.01:  # 閾値以上の場合のみ描画
                        # HOTカラーマップ（黒→赤→オレンジ→黄→白）
                        if intensity < 0.25:
                            # 黒から赤へ
                            r = int(255 * (intensity / 0.25))
                            g = 0
                            b = 0
                        elif intensity < 0.5:
                            # 赤からオレンジへ
                            r = 255
                            g = int(255 * ((intensity - 0.25) / 0.25))
                            b = 0
                        elif intensity < 0.75:
                            # オレンジから黄へ
                            r = 255
                            g = 255
                            b = int(255 * ((intensity - 0.5) / 0.25))
                        else:
                            # 黄から白へ
                            r = 255
                            g = 255
                            b = 255
                        
                        heatmap_colored[y, x] = [b, g, r]  # BGR形式
            
            # ヒートマップを背景に重ね合わせ（透明度70%）
            heatmap_mask = np.any(heatmap_colored > 0, axis=2)
            alpha = 0.7
            minimap_bg[heatmap_mask] = (1 - alpha) * minimap_bg[heatmap_mask] + alpha * heatmap_colored[heatmap_mask]
        
        # 外枠を描画（白い太枠）
        cv2.rectangle(minimap_bg, (0, 0), (self.minimap_width-1, self.minimap_height-1), (255, 255, 255), 3)
        
        # 内側の細い枠を描画
        cv2.rectangle(minimap_bg, (2, 2), (self.minimap_width-3, self.minimap_height-3), (150, 150, 150), 1)
        
        # グリッドを描画（画面を分割表示、より細かく）
        for i in range(1, 8):
            x = (i * self.minimap_width) // 8
            color = (120, 120, 120) if i % 2 == 0 else (80, 80, 80)
            cv2.line(minimap_bg, (x, 0), (x, self.minimap_height), color, 1)
        for j in range(1, 5):
            y = (j * self.minimap_height) // 5
            color = (120, 120, 120) if j % 2 == 0 else (80, 80, 80)
            cv2.line(minimap_bg, (0, y), (self.minimap_width, y), color, 1)
        
        # タイトルバーを描画
        cv2.rectangle(minimap_bg, (0, 0), (self.minimap_width, 20), (60, 60, 60), -1)
        title_text = "Gaze Heatmap" if self.show_heatmap else "Gaze Tracking"
        cv2.putText(minimap_bg, title_text, (8, 14), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        if eye_center_screen and gaze_direction:
            # 視線の終点位置を計算（統一スケール）
            gaze_scale = 180
            gaze_end_x = eye_center_screen[0] + gaze_direction[0] * gaze_scale
            gaze_end_y = eye_center_screen[1] + gaze_direction[1] * gaze_scale
            
            # 画面境界内にクランプ
            gaze_end_x = max(50, min(w - 50, gaze_end_x))
            gaze_end_y = max(50, min(h - 50, gaze_end_y))
            
            # メイン画面座標を小窓座標に変換
            minimap_gaze_x = int((gaze_end_x / w) * self.minimap_width)
            minimap_gaze_y = int((gaze_end_y / h) * self.minimap_height)
            
            # 境界内に制限
            minimap_gaze_x = max(5, min(self.minimap_width-5, minimap_gaze_x))
            minimap_gaze_y = max(25, min(self.minimap_height-5, minimap_gaze_y))
            
            # 視線位置を赤い点で表示（より目立つように）
            cv2.circle(minimap_bg, (minimap_gaze_x, minimap_gaze_y), 6, (0, 0, 255), -1)
            cv2.circle(minimap_bg, (minimap_gaze_x, minimap_gaze_y), 8, (0, 150, 255), 2)
            cv2.circle(minimap_bg, (minimap_gaze_x, minimap_gaze_y), 12, (255, 255, 255), 1)
            
            # 座標情報を下部に表示（改良版）
            coord_text = f"X:{int(gaze_end_x)} Y:{int(gaze_end_y)}"
            text_bg_x = 5
            text_bg_y = self.minimap_height - 15
            # 背景を少し透明にする
            cv2.rectangle(minimap_bg, (text_bg_x-3, text_bg_y-12), 
                         (text_bg_x + 150, text_bg_y + 5), (0, 0, 0), -1)
            cv2.rectangle(minimap_bg, (text_bg_x-2, text_bg_y-11), 
                         (text_bg_x + 149, text_bg_y + 4), (50, 50, 50), 1)
            cv2.putText(minimap_bg, coord_text, (text_bg_x, text_bg_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # 小窓をメインフレームに合成
        frame[self.minimap_y:self.minimap_y + self.minimap_height, 
              self.minimap_x:self.minimap_x + self.minimap_width] = minimap_bg
    
    def draw_gaze_line(self, frame, eye_center_screen, gaze_direction, hand_landmarks, pose_landmarks):
        if not gaze_direction:
            return
        
        h, w, _ = frame.shape
        
        # 視線の対象を計算
        target_point, target_type = self.calculate_gaze_target(
            gaze_direction, eye_center_screen, hand_landmarks, pose_landmarks, frame.shape
        )
        
        # 視線の終点位置を計算（統一スケール）
        gaze_scale = 180  # 数字認識のため統一スケール
        gaze_end_x = int(eye_center_screen[0] + gaze_direction[0] * gaze_scale)
        gaze_end_y = int(eye_center_screen[1] + gaze_direction[1] * gaze_scale)
        
        # 画面境界内にクランプ（画面下部の有効領域を拡大）
        gaze_end_x = max(20, min(w - 20, gaze_end_x))  # 左右マージンを縮小（50→20）
        gaze_end_y = max(10, min(h - 10, gaze_end_y))  # 上下マージンを大幅縮小（50→10）画面下部対応
        
        # ヒートマップスタイルの視線描画
        self.draw_gaze_heatmap(frame, eye_center_screen, (gaze_end_x, gaze_end_y))
        
        if target_point and target_type:
            # 対象の周りに円を描画
            cv2.circle(frame, target_point, 15, (0, 255, 0), 3)
            
            # 対象の名前を表示
            cv2.putText(frame, f'見ている: {target_type}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 距離を表示
            distance = np.sqrt((eye_center_screen[0] - target_point[0])**2 + 
                             (eye_center_screen[1] - target_point[1])**2)
            cv2.putText(frame, f'距離: {distance:.0f}px', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, '視線方向', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # 視線の数値情報を表示（高速モード表示を追加）
        mode_text = " [HIGH SPEED]" if self.high_speed_mode else ""
        cv2.putText(frame, f'Gaze: ({gaze_direction[0]:.1f}, {gaze_direction[1]:.1f}){mode_text}', 
                   (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)  # 線の太さを減らす
    
    def draw_gaze_heatmap(self, frame, eye_center, gaze_point):
        """軽量化されたヒートマップスタイルで視線を描画"""
        # 軽量化：段階的な円で簡単なヒートマップ効果を作成
        steps = 8  # ステップ数を大幅削減
        for i in range(steps):
            t = i / (steps - 1)
            x = int(eye_center[0] + t * (gaze_point[0] - eye_center[0]))
            y = int(eye_center[1] + t * (gaze_point[1] - eye_center[1]))
            
            # 各点に単純な円を描画（ガウシアン計算を回避）
            intensity = 0.7 * (1.0 - t * 0.4)  # 先端に向かって薄くなる
            radius = int(15 - t * 5)  # 半径も小さくなる
            
            # 簡単なカラーマッピング（計算量削減）
            if intensity > 0.5:
                color = (0, int(255 * intensity), 255)  # 黄色系
            elif intensity > 0.3:
                color = (0, int(255 * intensity), int(255 * intensity * 1.5))  # オレンジ系
            else:
                color = (0, 0, int(255 * intensity * 2))  # 赤系
            
            # 透明度付きの円を描画
            overlay = frame.copy()
            cv2.circle(overlay, (x, y), radius, color, -1)
            cv2.addWeighted(frame, 1 - intensity * 0.5, overlay, intensity * 0.5, 0, frame)
        
        # 視線の終点に明るい円を描画
        cv2.circle(frame, gaze_point, 6, (255, 255, 255), -1)
        cv2.circle(frame, gaze_point, 8, (0, 255, 255), 2)
    
    def run(self):
        print(f"視線追跡+ゲーム機能付きアイトラッキング開始 (クロスプラットフォーム版)")
        print(f"Platform: {self.platform_name} ({platform.release()})")
        print(f"Python: {platform.python_version()}")
        print("========== キーボードショートカット ==========")
        print("'q'/'Q'/ESCキー: 終了（自動でログ保存）")
        print("'g'キー: ゲームモード切り替え")
        print("'h'キー: ヒートマップ表示ON/OFF")
        print("'c'キー: ヒートマップクリア")
        print("'s'キー: 視線ログ手動保存")
        print("=======================================")
        print("緑の枠: 人物検出")
        print("青い線: 目の輪郭")
        print("黄色い点: 瞳孔")
        print("緑の矢印: 視線の導線")
        print("右下小窓: リアルタイムヒートマップ表示")
        print("ゲームモード: 赤い数字ターゲットに1.0秒視線を当てると緑色に変化")
        print("[HIGH SPEED]: 急激な眼球運動検出モード")
        
        fps_counter = 0
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # カメラ映像を水平方向に反転（ミラー効果）
            frame = cv2.flip(frame, 1)
            
            # FPSカウンター
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = 30 / elapsed_time
                start_time = time.time()
                print(f"FPS: {fps:.1f}")
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            
            # 変数を初期化
            smoothed_gaze = None
            smoothed_eye_center = None
            
            # 手とポーズの検出（フレームをスキップして高速化）
            # 3フレームに1回だけ手とポーズを検出
            if hasattr(self, 'detection_counter'):
                self.detection_counter += 1
            else:
                self.detection_counter = 0
            
            if self.detection_counter % 3 == 0:
                hand_landmarks, pose_landmarks = self.detect_hands_and_pose(frame_rgb)
                # 検出結果をキャッシュ
                self.cached_hand_landmarks = hand_landmarks
                self.cached_pose_landmarks = pose_landmarks
            else:
                # キャッシュされた結果を使用
                hand_landmarks = getattr(self, 'cached_hand_landmarks', [])
                pose_landmarks = getattr(self, 'cached_pose_landmarks', [])
            
            # 人物検出
            detection_results = self.face_detection.process(frame_rgb)
            self.person_detected = False
            
            if detection_results.detections:
                self.person_detected = True
                
                # 顔の境界ボックスを描画
                for detection in detection_results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(frame, f'Person: {detection.score[0]:.2f}', 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 顔メッシュ検出
                mesh_results = self.face_mesh.process(frame_rgb)
                
                if mesh_results.multi_face_landmarks:
                    landmarks = mesh_results.multi_face_landmarks[0].landmark
                    
                    # 目の輪郭を描画
                    self.draw_eye_contours(frame, landmarks, w, h)
                    
                    # 虹彩を描画
                    self.draw_iris_points(frame, landmarks, w, h)
                    
                    # 視線方向を計算（高速化のため計算を最適化）
                    left_eye_center = self.get_eye_center(landmarks, self.LEFT_EYE_INDICES)
                    right_eye_center = self.get_eye_center(landmarks, self.RIGHT_EYE_INDICES)
                    left_iris_center = self.get_iris_center(landmarks, self.LEFT_IRIS_INDICES)
                    right_iris_center = self.get_iris_center(landmarks, self.RIGHT_IRIS_INDICES)
                    
                    # 視線方向計算を最適化（急激な動きをサポート）
                    gaze_direction = self.calculate_gaze_direction(
                        left_eye_center, right_eye_center, 
                        left_iris_center, right_iris_center
                    )
                    
                    # 目の中心点を画面座標で計算
                    if left_eye_center is not None and right_eye_center is not None:
                        eye_center_normalized = (left_eye_center + right_eye_center) / 2
                        eye_center_screen = (int(eye_center_normalized[0] * w), 
                                           int(eye_center_normalized[1] * h))
                        
                        # 視線データをスムージング
                        smoothed_gaze, smoothed_eye_center = self.smooth_gaze_data(gaze_direction, eye_center_screen)
                        
                        if smoothed_eye_center:
                            self.eye_center_point = smoothed_eye_center
                            
                            # スムージングされた目の中心に小さな円を描画
                            cv2.circle(frame, smoothed_eye_center, 8, (255, 255, 255), 2)
                            cv2.circle(frame, smoothed_eye_center, 4, (0, 0, 0), -1)
                        
                        # 手とポーズのランドマークを描画
                        self.draw_hands_and_pose(frame, hand_landmarks, pose_landmarks)
                        
                        # ゲームモードのターゲットを描画
                        self.draw_game_targets(frame)
                        
                        # 視線の導線を常に描画
                        if smoothed_gaze and smoothed_eye_center:
                            self.draw_gaze_line(frame, smoothed_eye_center, smoothed_gaze, 
                                              hand_landmarks, pose_landmarks)
                            
                            # 視線の終点位置を計算（統一スケール）
                            gaze_scale = 180
                            gaze_end_x = int(smoothed_eye_center[0] + smoothed_gaze[0] * gaze_scale)
                            gaze_end_y = int(smoothed_eye_center[1] + smoothed_gaze[1] * gaze_scale)
                            
                            # 画面境界内にクランプ
                            gaze_end_x = max(50, min(w - 50, gaze_end_x))
                            gaze_end_y = max(50, min(h - 50, gaze_end_y))
                            
                            # ヒートマップデータを更新
                            self.update_heatmap(gaze_end_x, gaze_end_y, w, h)
                            
                            # 視線データをログに記録
                            self.log_gaze_data(gaze_end_x, gaze_end_y, 
                                             smoothed_eye_center[0], smoothed_eye_center[1])
                            
                            # 右下の小窓に視線位置とヒートマップを表示
                            self.draw_minimap(frame, smoothed_eye_center, smoothed_gaze)
                        
                        # ゲームモードでの視線チェック（矢印の動線全体を使用）
                        if self.game_mode and smoothed_gaze and smoothed_eye_center:
                            # 視線の矢印の先端位置を計算（統一スケール）
                            gaze_scale = 180  # 数字認識に最適化された統一スケール
                            gaze_end_x = int(smoothed_eye_center[0] + smoothed_gaze[0] * gaze_scale)
                            gaze_end_y = int(smoothed_eye_center[1] + smoothed_gaze[1] * gaze_scale)
                            
                            # 画面境界内にクランプ
                            gaze_end_x = max(50, min(w - 50, gaze_end_x))
                            gaze_end_y = max(50, min(h - 50, gaze_end_y))
                            
                            self.check_gaze_on_targets(smoothed_eye_center, gaze_end_x, gaze_end_y)
                    
                    status = "視線追跡中"
                else:
                    status = "顔の詳細を検出中..."
                    
                cv2.putText(frame, status, (10, h - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, 'No Person Detected', (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # フレームを表示
            window_title = 'Eye Tracker - Game Mode' if self.game_mode else 'Eye Tracker with Gaze Lines'
            cv2.imshow(window_title, frame)
            
            # ウィンドウが閉じられたかチェック
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                print("ウィンドウが閉じられました。アプリケーションを終了します...")
                if len(self.gaze_log) > 0:
                    print("視線ログを保存中...")
                    self.save_gaze_log()
                break
            
            # キーボード操作（クロスプラットフォーム対応）
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # 'q', 'Q', ESCキーで終了
                print("アプリケーションを終了します...")
                # 終了時にログを自動保存
                if len(self.gaze_log) > 0:
                    print("視線ログを保存中...")
                    self.save_gaze_log()
                break
            elif key == ord('g'):
                self.toggle_game_mode()
            elif key == ord('h'):
                self.toggle_heatmap_display()
            elif key == ord('c'):
                self.clear_heatmap()
            elif key == ord('s'):
                self.save_gaze_log()
            
            # ゲームモードの情報表示
            if self.game_mode:
                completed_targets = sum(1 for target in self.targets if target["color"] == (0, 255, 0))
                if self.next_target_number <= 2:
                    cv2.putText(frame, f'Next: {self.next_target_number} | Progress: {completed_targets}/2 Cleared (1.0s to clear)', 
                               (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, f'GAME COMPLETE! All {completed_targets}/2 Cleared!', 
                               (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, 'Press G to exit game mode | Follow numbers 1->2 in order!', 
                           (10, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                
                # ゲームモードでの座標情報を表示
                if smoothed_gaze and smoothed_eye_center:
                    # 視線の終点座標を計算（統一スケール）
                    gaze_scale = 180
                    gaze_end_x = int(smoothed_eye_center[0] + smoothed_gaze[0] * gaze_scale)
                    gaze_end_y = int(smoothed_eye_center[1] + smoothed_gaze[1] * gaze_scale)
                    
                    # 画面境界内にクランプ
                    gaze_end_x = max(50, min(w - 50, gaze_end_x))
                    gaze_end_y = max(50, min(h - 50, gaze_end_y))
                    
                    # 左下基準(0,0)座標に変換
                    eye_center_x_bottom = smoothed_eye_center[0]
                    eye_center_y_bottom = h - smoothed_eye_center[1]  # Y軸を反転
                    
                    gaze_end_x_bottom = gaze_end_x
                    gaze_end_y_bottom = h - gaze_end_y  # Y軸を反転
                    
                    # 座標情報を左上角に表示
                    coord_bg_color = (40, 40, 40)  # 暗いグレー背景
                    coord_text_color = (0, 255, 255)  # シアン色テキスト
                    
                    # 背景ボックスを描画
                    cv2.rectangle(frame, (10, 10), (380, 100), coord_bg_color, -1)
                    cv2.rectangle(frame, (10, 10), (380, 100), (255, 255, 255), 2)
                    
                    # 座標情報のヘッダー
                    cv2.putText(frame, '=== GAME MODE COORDINATES (Bottom-Left Origin) ===', 
                               (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                    
                    # 目の中心座標（左下基準）
                    cv2.putText(frame, f'Eye Center (Bottom-Left): ({eye_center_x_bottom}, {eye_center_y_bottom})', 
                               (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, coord_text_color, 1)
                    
                    # 視線方向ベクトル（計算データ）
                    cv2.putText(frame, f'Gaze Vector (Calculated): ({smoothed_gaze[0]:.1f}, {-smoothed_gaze[1]:.1f})', 
                               (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, coord_text_color, 1)
                    
                    # 視線終点座標（左下基準）
                    cv2.putText(frame, f'Gaze Target (Bottom-Left): ({gaze_end_x_bottom}, {gaze_end_y_bottom})', 
                               (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, coord_text_color, 1)
                    
                    # スケール情報（数字認識最適化版）
                    cv2.putText(frame, f'Scale Factor: {gaze_scale} (Optimized for Numbers) | Screen Size: {w}x{h}', 
                               (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            else:
                cv2.putText(frame, 'Press G for game mode | H: Heatmap | C: Clear | S: Save Log', 
                           (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            # ヒートマップとログ情報を左上に表示
            info_y = 100
            cv2.putText(frame, f'Heatmap: {"ON" if self.show_heatmap else "OFF"} | Log entries: {len(self.gaze_log)}', 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # ヒートマップのホットスポット数を表示
            if self.show_heatmap:
                hotspots = np.sum(self.heatmap_data > 0.1)
                cv2.putText(frame, f'Active hotspots: {hotspots}', 
                           (10, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
        
        # リソースの解放
        print("リソースを解放中...")
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("アプリケーションが正常に終了しました。")

if __name__ == "__main__":
    tracker = SimpleEyeTracker()
    tracker.run()