import json
import math
import time
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from fpdf import FPDF
import os
import io
from PIL import Image, ImageDraw, ImageFont

# ================= 視線推定ユーティリティ =================

class GazeEstimator:
    """
    MediaPipeのFace Meshを使用して視線を推定するクラス。
    キャリブレーション機能も含む。
    """

    # MediaPipeのFace Meshランドマークのインデックス
    LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    LEFT_IRIS = [474, 475, 476, 477, 468, 469, 470, 471, 472]
    RIGHT_IRIS = [469, 470, 471, 472, 473, 474, 475, 476, 477]

    def __init__(self, det_conf=0.5, trk_conf=0.5):
        """MediaPipe FaceMeshモデルを初期化"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # 虹彩（IRIS）ランドマークを使用するためにTrue
            min_detection_confidence=det_conf,
            min_tracking_confidence=trk_conf,
        )

        # --- キャリブレーション用データ ---
        # 視線ベクトル(raw_vec)の水平・垂直方向の最小・最大値
        self.rx_left: Optional[float] = None
        self.rx_right: Optional[float] = None
        self.ry_top: Optional[float] = None
        self.ry_bottom: Optional[float] = None
        # キャリブレーションによって計算されたオフセット（ピクセル単位）
        self.offx_px: float = 0.0
        self.offy_px: float = 0.0
        # キャリブレーション済みかどうかのフラグ
        self.calibrated: bool = False
        # キャリブレーション時の基準となった画面サイズ
        self.base_w: Optional[int] = None
        self.base_h: Optional[int] = None

    @staticmethod
    def _center(landmarks, idxs):
        """指定されたインデックスのランドマーク群の中心座標を計算する"""
        pts = np.array([[landmarks[i].x, landmarks[i].y] for i in idxs if i < len(landmarks)])
        if len(pts) == 0:
            return None
        return pts.mean(axis=0)

    def _iris_center_circle_px(self, landmarks, indices, w, h):
        """
        虹彩のランドマーク群に最小外接円をあてはめ、その中心座標（ピクセル単位）を返す
        """
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices if i < len(landmarks)]
        if len(pts) < 3:
            return None
        pts_np = np.array(pts, dtype=np.float32)
        (cx, cy), _ = cv2.minEnclosingCircle(pts_np)
        return (cx, cy)

    def set_calibration_values(
        self,
        rx_left: float, rx_right: float,
        ry_top: float, ry_bottom: float,
        offx_px: float, offy_px: float,
        base_w: int, base_h: int
    ):
        """キャリブレーションで計算された値をクラス変数に保存する"""
        self.rx_left = rx_left
        self.rx_right = rx_right
        self.ry_top = ry_top
        self.ry_bottom = ry_bottom
        self.offx_px = offx_px
        self.offy_px = offy_px
        self.base_w = base_w
        self.base_h = base_h
        self.calibrated = True

    def estimate(self, frame_bgr: np.ndarray):
        """
        フレーム画像から顔を検出し、視線（目の中心、視線の先）を推定する

        戻り値:
        - ok (bool): 検出成功フラグ
        - eye_px (Tuple[int, int]): 目の中心座標 (ピクセル)
        - end_px (Tuple[int, int]): 視線の先の座標 (ピクセル)
        - raw_vec (Tuple[float, float]): 生の視線ベクトル (正規化座標系)
        """
        h, w, _ = frame_bgr.shape
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            return False, None, None, None

        lm = res.multi_face_landmarks[0].landmark

        # 1. 目の中心座標 (左右の平均)
        lc = self._center(lm, self.LEFT_EYE)
        rc = self._center(lm, self.RIGHT_EYE)
        if lc is None or rc is None:
            return False, None, None, None
        eye_center = (lc + rc) / 2.0
        eye_px = (int(eye_center[0] * w), int(eye_center[1] * h))

        # 2. 虹彩の中心座標 (左右の平均)
        li_px = self._iris_center_circle_px(lm, self.LEFT_IRIS, w, h)
        ri_px = self._iris_center_circle_px(lm, self.RIGHT_IRIS, w, h)
        if li_px is None or ri_px is None:
            return False, None, None, None
        iris_px = ((li_px[0] + ri_px[0]) * 0.5, (li_px[1] + ri_px[1]) * 0.5)

        # 3. 「目の中心」から「虹彩の中心」へのベクトル (raw_vec) を計算
        iris_center = np.array([iris_px[0] / w, iris_px[1] / h]) # 虹彩中心を正規化
        eye_center_n = np.array([eye_center[0], eye_center[1]]) # 目の中心（正規化）
        raw_vec = iris_center - eye_center_n # (rx, ry)

        # 4. キャリブレーション済みかどうかで処理を分岐
        if self.calibrated and self.rx_left is not None and self.rx_right is not None \
           and self.ry_top is not None and self.ry_bottom is not None:
            # --- キャリブレーション適用 ---
            # 登録されたraw_vecの範囲と現在の画面サイズからスケールを計算
            dx = (self.rx_right - self.rx_left)
            dy = (self.ry_bottom - self.ry_top)

            if abs(dx) < 1e-9 or abs(dy) < 1e-9:
                # ゼロ除算防止
                scale_x = (w - 10) / 0.01
                scale_y = (h - 10) / 0.01
            else:
                scale_x = (w - 10) / dx
                scale_y = (h - 10) / dy

            # スケールとオフセットを適用
            gx = raw_vec[0] * scale_x + self.offx_px
            gy = raw_vec[1] * scale_y + self.offy_px
        else:
            # --- 未キャリブの簡易動作 ---
            # 暫定的なスケールを適用 (オフセットなし)
            scale_x = (w - 10) / 0.01
            scale_y = (h - 10) / 0.001
            gx = raw_vec[0] * scale_x
            gy = raw_vec[1] * scale_y

        # 5. 最終的な視線座標 (end_px) を計算
        end_px = (
            int(np.clip(eye_px[0] + gx, 10, w - 10)), # 画面外にはみ出ないようにclip
            int(np.clip(eye_px[1] + gy, 10, h - 10)),
        )

        return True, eye_px, end_px, (float(raw_vec[0]), float(raw_vec[1]))


# ================== Region 集計 ==================
class RegionTimer:
    """
    視線が各領域（Region）に滞在した合計時間を計測するクラス
    """
    def __init__(self):
        """4領域分の合計時間 (stats) を0で初期化"""
        self.num_regions = 4
        # self.stats[1] が Region I, [2] が Region II ...
        self.stats = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        self.last_ts = None # 最後にupdateが呼ばれた時刻

    def region_of(self, x: int, y: int, w: int, h: int) -> Optional[int]:
        """
        指定された座標(x, y)が、定義済みの4領域のどれに属するかを返す
        （要件資料のスクリーンショットに基づき、画面サイズ(w, h)に対する割合で定義）
        """
        # Region I: エラー文字 (上部中央)
        if (0.20 * w < x < 0.70 * w) and (0.05 * h < y < 0.15 * h):
            return 1
        # Region IV: 過去NG例 (右上)
        if (x > 0.73 * w) and (0.05 * h < y < 0.60 * h):
            return 4
        # Region III: 3Dデータ (左下)
        if (x < 0.22 * w) and (y > 0.60 * h):
            return 3
        # Region II: 基盤ウィンドウ (中央)
        if (0.22 * w < x < 0.73 * w) and (0.15 * h < y < 0.90 * h):
            return 2

        # どの領域でもない
        return None

    def update(self, gaze_px: Tuple[int, int], w: int, h: int):
        """
        視線座標(gaze_px)を受け取り、該当する領域の合計時間を更新する
        """
        now = time.time()
        if self.last_ts is None:
            self.last_ts = now; return

        # 1フレーム前の呼び出しからの経過時間 (delta time)
        dt = now - self.last_ts
        self.last_ts = now

        # 視線がどの領域にあるか判定
        r = self.region_of(gaze_px[0], gaze_px[1], w, h)

        # 該当する領域の合計時間(stats)にdtを加算
        if r is not None and r in self.stats:
            self.stats[r] += dt


def y_intercept(rx_left, rx_right, w):
    """
    キャリブレーション用の線形変換の係数（スケールa, オフセットb）を計算するヘルパー
    (x_screen = a * x_raw + b)
    """
    if abs(rx_right - rx_left) < 1e-9: return 0, 0
    a = w / (rx_right - rx_left)
    b = -(w/2) * (rx_right + rx_left) / (rx_right - rx_left)
    return a, b

# ================== ヒートマップ生成クラス ==================
class HeatmapGenerator:
    """視線ヒートマップを生成・管理するクラス"""
    def __init__(self, width: int, height: int):
        """ヒートマップ用のNumpy配列を初期化"""
        self.width = width; self.height = height
        self.heatmap_data = np.zeros((height, width), dtype=np.float64)
        self.decay_rate = 0.98 # 減衰率
        self.intensity = 5.0 # 視線が当たった際の強度
        self.radius = int(max(width, height) * 0.15) # 視線スポットの半径

    def update(self, gaze_point: Tuple[int, int]):
        """ヒートマップに視線位置(gaze_point)を追加（ガウス分布）"""
        x, y = gaze_point
        if x is None or y is None: return
        x_grid, y_grid = np.ogrid[:self.height, :self.width]
        dist_sq = (x_grid - y)**2 + (y_grid - x)**2
        gauss = self.intensity * np.exp(-dist_sq / (2 * (self.radius / 2)**2))
        self.heatmap_data += gauss

    def decay(self):
        """ヒートマップ全体を時間経過で減衰させる"""
        self.heatmap_data *= self.decay_rate
        self.heatmap_data[self.heatmap_data < 0.01] = 0

    def generate_minimap_image(self, current_gaze_point: Optional[Tuple[int, int]]) -> np.ndarray:
        """現在のヒートマップと視線位置を描画したミニマップ画像(ndarray)を生成する"""
        minimap_bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.rectangle(minimap_bg, (0, 0), (self.width, self.height), (30, 30, 30), -1)

        # ヒートマップを色付け(COLORMAP_JET)して描画
        if np.max(self.heatmap_data) > 0:
            norm_heatmap = self.heatmap_data / np.max(self.heatmap_data)
            heatmap_8u = (norm_heatmap * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_8u, cv2.COLORMAP_JET)
            minimap_bg = cv2.addWeighted(minimap_bg, 0.5, heatmap_color, 0.5, 0)

        # 現在の視線位置を赤い点で描画
        if current_gaze_point:
            cv2.circle(minimap_bg, current_gaze_point, 5, (0, 0, 255), -1)
            cv2.circle(minimap_bg, current_gaze_point, 7, (255, 255, 255), 1)

        cv2.rectangle(minimap_bg, (0, 0), (self.width - 1, self.height - 1), (255, 255, 255), 1)
        return minimap_bg

# ================ Video Processor =================
class VideoProcessor:
    """
    streamlit-webrtc の VideoProcessorFactory クラス。
    Webカメラの各フレームを処理する本体。
    """
    def __init__(self):
        self.est = GazeEstimator(0.5, 0.5) # 視線推定器
        self.running = False # 分析中フラグ (UIの「分析開始」ボタンでTrue)

        # --- キャリブレーション状態 ---
        self.targets = [("左上", 0.10, 0.10), ("右上", 0.90, 0.10), ("左下", 0.10, 0.90), ("右下", 0.90, 0.90)]
        self.calibrating = False # キャリブ中フラグ
        self.target_idx = 0 # 現在のターゲット (0:左上, 1:右上, ...)
        self.capture_request = False # サンプル取得中フラグ
        self.capture_buffer: List[Tuple[float, float, int, int]] = [] # サンプル一時保存用
        self.samples: Dict[str, Tuple[float, float, float, float]] = {} # ターゲットごとの平均サンプル
        self.last_w: Optional[int] = None; self.last_h: Optional[int] = None # 最後のフレームサイズ
        self.calib_ready = False # 4点のサンプルが揃ったフラグ

        # --- FPS計測 ---
        self.fps_start_time = 0; self.fps_frame_count = 0; self.fps = 0.0

        # --- ヒートマップ ---
        self.minimap_width = 220; self.minimap_height = 140
        self.heatmap_generator = HeatmapGenerator(self.minimap_width, self.minimap_height)
        self.show_heatmap = True # UIのチェックボックスで切り替え

        # --- 領域計測 ---
        self.num_regions = 4
        self.timer = RegionTimer() # 領域タイマー

    def start_calibration(self):
        """キャリブレーションを開始（UIボタンから呼ばれる）"""
        self.calibrating = True; self.calib_ready = False; self.target_idx = 0
        self.capture_request = False; self.capture_buffer.clear(); self.samples.clear()

    def request_sample(self):
        """現在のターゲットのサンプル取得を要求（UIボタンから呼ばれる）"""
        if self.calibrating and self.target_idx < len(self.targets):
            self.capture_request = True; self.capture_buffer.clear()

    def skip_target(self):
        """（デバッグ用）現在のターゲットをスキップ"""
        if self.calibrating and self.target_idx < len(self.targets):
            self.target_idx += 1; self.capture_request = False; self.capture_buffer.clear()
            if self.target_idx >= len(self.targets): self.calib_ready = True

    def reset_calibration(self):
        """キャリブレーションをリセット（UIボタンから呼ばれる）"""
        self.start_calibration()

    def _compute_and_apply_calibration(self) -> Tuple[bool, str]:
        """
        収集した4点のサンプル(self.samples)から、視線のスケールとオフセットを計算し、
        GazeEstimatorに適用する。
        """
        need = {"左上", "右上", "左下", "右下"}
        if not need.issubset(self.samples.keys()): return False, "四隅のサンプルが揃っていません。"
        if self.last_w is None or self.last_h is None: return False, "カメラから映像が取得できていません。"

        # 各点の平均値を取得
        rx_lu, ry_lu, ex_lu, ey_lu = self.samples["左上"]; rx_ru, ry_ru, ex_ru, ey_ru = self.samples["右上"]
        rx_ld, ry_ld, ex_ld, ey_ld = self.samples["左下"]; rx_rd, ry_rd, ex_rd, ey_rd = self.samples["右下"]

        # raw_vecのX方向（左右）の範囲を計算
        rx_left = (rx_lu + rx_ld) / 2.0; rx_right = (rx_ru + rx_rd) / 2.0
        # raw_vecのY方向（上下）の範囲を計算
        ry_top = (ry_lu + ry_ru) / 2.0; ry_bottom = (ry_ld + ry_rd) / 2.0

        w, h = self.last_w, self.last_h
        dx = rx_right - rx_left; dy = ry_bottom - ry_top

        if abs(dx) < 1e-9 or abs(dy) < 1e-9: return False, "差分が小さすぎます。サンプルを取り直してください。"

        # スケールとオフセットを計算
        scale_x, offx = y_intercept(rx_left, rx_right, w)
        scale_y, offy = y_intercept(ry_top, ry_bottom, h) # ※y_interceptはY軸（下向き正）でも使える

        # GazeEstimatorに値を設定
        self.est.set_calibration_values(
            rx_left=rx_left, rx_right=rx_right, ry_top=ry_top, ry_bottom=ry_bottom,
            offx_px=float(offx), offy_px=float(offy), base_w=w, base_h=h)

        return True, (f"適用 OK: dx={dx:.6g}, dy={dy:.6g}, offx={offx:.1f}px, offy={offy:.1f}px, "
                      f"scale_x={scale_x:.1f}, scale_y={scale_y:.1f} @ {w}x{h}, samples={rx_left, rx_right, ry_top, ry_bottom}")

    def _draw_guides(self, frame):
        """フレームに4領域のガイド（緑色の四角）を描画する"""
        h, w, _ = frame.shape; color = (0, 255, 0); thickness = 2
        # Region I
        r1_x1, r1_y1 = int(0.20 * w), int(0.05 * h); r1_x2, r1_y2 = int(0.70 * w), int(0.15 * h)
        cv2.rectangle(frame, (r1_x1, r1_y1), (r1_x2, r1_y2), color, thickness)
        cv2.putText(frame, "I", (r1_x1 + 5, r1_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        # Region II
        r2_x1, r2_y1 = int(0.22 * w), int(0.15 * h); r2_x2, r2_y2 = int(0.73 * w), int(0.90 * h)
        cv2.rectangle(frame, (r2_x1, r2_y1), (r2_x2, r2_y2), color, thickness)
        cv2.putText(frame, "II", (r2_x1 + 5, r2_y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        # Region III
        r3_x1, r3_y1 = int(0), int(0.60 * h); r3_x2, r3_y2 = int(0.22 * w), int(h)
        cv2.rectangle(frame, (r3_x1, r3_y1), (r3_x2, r3_y2), color, thickness)
        cv2.putText(frame, "III", (r3_x1 + 5, r3_y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        # Region IV
        r4_x1, r4_y1 = int(0.73 * w), int(0.05 * h); r4_x2, r4_y2 = int(w), int(0.60 * h)
        cv2.rectangle(frame, (r4_x1, r4_y1), (r4_x2, r4_y2), color, thickness)
        cv2.putText(frame, "IV", (r4_x1 + 5, r4_y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    def _draw_calib_target(self, frame):
        """フレームにキャリブレーション用のターゲット（緑色の円）を描画する"""
        if not self.calibrating: return; h, w, _ = frame.shape
        if self.calib_ready or self.target_idx >= len(self.targets): return
        name, nx, ny = self.targets[self.target_idx]; px, py = int(nx * w), int(ny * h)
        cv2.circle(frame, (px, py), 16, (0, 255, 0), 3); cv2.circle(frame, (px, py), 6, (0, 255, 255), -1)

    def draw_overlay(self, frame: np.ndarray, eye_px, end_px):
        """フレームに視線（目の中心と視線の先）を描画する"""
        if eye_px and end_px:
            cv2.circle(frame, eye_px, 6, (255, 255, 255), -1) # 目の中心(白)
            cv2.circle(frame, end_px, 6, (0, 255, 255), -1) # 視線の先(黄)
            # 軌跡を描画
            for i in range(8):
                t = i / (8 - 1); x = int(eye_px[0] + t * (end_px[0] - eye_px[0])); y = int(eye_px[1] + t * (end_px[1] - eye_px[1]))
                cv2.circle(frame, (x, y), max(2, 6 - i), (0, 200, 255), -1)
        return frame

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        WebRTCから新しいフレームが来るたびに呼び出されるメインの処理関数
        """
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # 左右反転
        h, w, _ = img.shape
        self.last_w, self.last_h = img.shape[1], img.shape[0]

        # 1. 視線推定
        ok, eye_px, end_px, raw_vec = self.est.estimate(img)

        # 2. FPS計測
        if self.fps_start_time == 0: self.fps_start_time = time.time()
        self.fps_frame_count += 1; elapsed_time = time.time() - self.fps_start_time
        if elapsed_time >= 1.0:
            self.fps = self.fps_frame_count / elapsed_time; self.fps_frame_count = 0; self.fps_start_time = time.time()

        # 3. キャリブレーションのサンプル収集中
        if self.calibrating and self.capture_request and ok and raw_vec:
            self.capture_buffer.append((raw_vec[0], raw_vec[1], eye_px[0], eye_px[1]))
            # 15フレーム分たまったら平均をとって保存
            if len(self.capture_buffer) >= 15:
                avg = np.mean(self.capture_buffer, axis=0); name, _, _ = self.targets[self.target_idx]
                self.samples[name] = tuple(float(v) for v in avg)
                self.capture_request = False; self.capture_buffer.clear(); self.target_idx += 1
                if self.target_idx >= len(self.targets): self.calib_ready = True # 4点完了

        # 4. 分析中 (running=True) のみタイマーとヒートマップを更新
        current_minimap_gaze = None
        if self.running and ok and end_px:
            # 4-1. 領域タイマーを更新
            self.timer.update(end_px, w, h)

            # 4-2. ヒートマップを更新
            if self.show_heatmap:
                # 画面座標(end_px)をミニマップ座標に変換
                minimap_x = int((end_px[0] / w) * self.minimap_width); minimap_y = int((end_px[1] / h) * self.minimap_height)
                current_minimap_gaze = (minimap_x, minimap_y)
                self.heatmap_generator.update(current_minimap_gaze)

        # 5. ヒートマップ（ミニマップ）の描画
        if self.show_heatmap:
            self.heatmap_generator.decay() # 減衰
            minimap_image = self.heatmap_generator.generate_minimap_image(current_minimap_gaze)
            # 画面右下に合成
            margin = 10; map_h, map_w, _ = minimap_image.shape; y_offset = h - map_h - margin; x_offset = w - map_w - margin
            img[y_offset:y_offset + map_h, x_offset:x_offset + map_w] = minimap_image

        # 6. オーバーレイ（視線、ガイド、ターゲット）を描画
        img = self.draw_overlay(img, eye_px if ok else None, end_px if ok else None)
        self._draw_guides(img)
        self._draw_calib_target(img) # キャリブ中のみ描画される

        # 7. FPSを描画
        fps_text = f"FPS: {self.fps:.1f}"; cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 8. 処理済みフレームを返す
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def apply_calibration(self) -> Tuple[bool, str]:
        """UIから呼ばれ、キャリブレーション計算・適用を実行する"""
        return self._compute_and_apply_calibration()

def get_region_labels(num_regions: int) -> Dict[int, str]:
    """領域IDに対応するラベル(I, II, III, IV)を返す"""
    if num_regions == 4:
        return {1: "I: エラー文字", 2: "II: 基盤ウィンドウ", 3: "III: 3Dデータウィンドウ", 4: "IV: 過去のNG例ウィンドウ"}
    if num_regions == 3: # 過去の履歴データ用
        return {1: "① 左", 2: "② 中央", 3: "③ 右"}
    return {i: f"エリア {i}" for i in range(1, num_regions + 1)}


# ================= 分析レポート =================
def generate_analysis_report(
    actual_stats: Dict[int, float],
    standard_durations: Dict[int, float],
    labels: Dict[int, str]
) -> Tuple[str, str]:
    """
    標準動作（合計時間）と分析動作（合計時間）を比較し、
    "標準動作：..." と "分析動作：..." の2つの文字列を生成する。
    """
    standard_parts = []
    analysis_parts = []

    # 基準の順番 (I, II, III, IV) で処理
    for region_id in sorted(standard_durations.keys()):
        label_name = labels.get(region_id, f"Region {region_id}").split(":")[0] # "I: エラー文字" -> "I"
        standard_sec = standard_durations.get(region_id, 0.0)
        actual_sec = actual_stats.get(region_id, 0.0)

        # 1. 標準動作文字列の作成 (例: "I(2秒)")
        standard_parts.append(f"{label_name}({standard_sec:.0f}秒)")

        # 2. 分析動作文字列の作成 (例: "I(2.0秒: ◯ ...)")
        tolerance = 1.0 # 許容誤差1秒

        if actual_sec == 0:
            symbol, comment = "×", "見ていない可能性が高いです"
        elif abs(actual_sec - standard_sec) <= tolerance:
            symbol, comment = "◯", "標準動作通りに見ています"
        elif actual_sec < standard_sec:
            symbol, comment = "△", "見ている時間が短いです"
        else: # actual_sec > standard_sec
            symbol, comment = "△", "見ている時間が長いです"

        analysis_parts.append(f"{label_name}({actual_sec:.1f}秒: {symbol} {comment})")

    # "→"で連結
    standard_str = "→".join(standard_parts)
    analysis_str = "→".join(analysis_parts)

    return standard_str, analysis_str

# ================= PDF =================
JAPANESE_FONT_PATH = "ipaexg00401/ipaexg.ttf" # 日本語フォントパス
STANDARD_DURATIONS = {1: 2.0, 2: 4.0, 3: 3.0, 4: 3.0} # 標準動作の秒数 (I:2s, II:4s, ...)

def create_history_pdf(history_data: list) -> io.BytesIO:
    """
    履歴データ(history_data)を受け取り、PDFファイル(BytesIO)を生成する
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('Japanese', '', JAPANESE_FONT_PATH, uni=True) # 日本語フォントを登録
    pdf.set_font('Japanese', '', 16)
    pdf.cell(0, 10, "アイトラッキング分析履歴", 0, 1, 'C')
    pdf.ln(5)

    # PDFの先頭に「基準（標準動作）」を記載
    standard_labels_for_pdf = get_region_labels(4)
    standard_str, _ = generate_analysis_report({}, STANDARD_DURATIONS, standard_labels_for_pdf)
    pdf.set_font('Japanese', '', 12)
    pdf.set_fill_color(245, 245, 245)
    pdf.cell(35, 8, "基準 (標準動作):", border=0, fill=False)
    pdf.multi_cell(0, 8, standard_str, border=1, fill=True)
    pdf.ln(5)

    # 履歴を1件ずつ処理 (reversedで新しい順に)
    for item in reversed(history_data):
        stats = item.get("stats", {})
        num_regions = len(stats)
        if num_regions == 0: continue

        labels = get_region_labels(num_regions) # 履歴のリージョン数に合わせたラベルを取得

        pdf.set_font('Japanese', '', 12)
        pdf.cell(0, 10, f"記録日時: {item['time']}", 0, 1)
        pdf.set_font('Japanese', '', 10)
        pdf.set_fill_color(240, 240, 240)

        # --- 履歴テーブルを描画 ---
        cell_width = 60; cell_height = 8
        pdf.cell(cell_width, cell_height, "範囲", border=1, fill=True, align='C')

        is_4_region_history = (num_regions == 4)
        if is_4_region_history:
            pdf.set_font('Japanese', '', 9) # 3列にするためフォントを少し小さく
            pdf.cell(cell_width, cell_height, "標準動作 (秒)", border=1, fill=True, align='C')
            pdf.cell(cell_width, cell_height, "合計秒", border=1, fill=True, align='C')
            pdf.set_font('Japanese', '', 10)
        else:
             pdf.cell(cell_width, cell_height, "合計秒", border=1, fill=True, align='C')
        pdf.ln()

        for i in sorted(stats.keys()):
            pdf.cell(cell_width, cell_height, labels.get(i, f"Region {i}"), border=1)
            if is_4_region_history:
                standard_sec = STANDARD_DURATIONS.get(i, 0.0)
                pdf.cell(cell_width, cell_height, f"{standard_sec:.1f}", border=1, align='C')
            pdf.cell(cell_width, cell_height, f"{stats.get(i, 0.0):.1f}", border=1, align='C')
            pdf.ln()
        # --- 履歴テーブルここまで ---

        # 4領域の履歴の場合、分析レポート（◯△×）も記載
        if num_regions == 4:
            _, analysis_str = generate_analysis_report(stats, STANDARD_DURATIONS, labels)
            pdf.set_font('Japanese', '', 10)
            pdf.cell(30, 8, "分析動作:", border=0)
            pdf.set_fill_color(250, 250, 250)
            pdf.multi_cell(0, 8, analysis_str, border=1, fill=True)

        pdf.ln(10)

    # PDFをBytesIOオブジェクトとして返す
    return io.BytesIO(pdf.output())


# ================= Streamlit UI ==================
st.set_page_config(page_title="基板実装状態の確認作業 アイトラッキングシステム", layout="wide")
st.title("基板実装状態の確認作業 アイトラッキングシステム")

# --- セッションステート初期化 ---
# 履歴データ
if "history" not in st.session_state: st.session_state.history = []
# 分析中フラグ
if "running" not in st.session_state: st.session_state.running = False
# 領域数は4で固定
st.session_state.num_regions = 4
# 現在の分析結果（合計時間）
if "current_stats" not in st.session_state or len(st.session_state.current_stats) != 4:
    st.session_state.current_stats = {i: 0.0 for i in range(1, 4 + 1)}

# 領域ラベルを取得
region_labels = get_region_labels(4)

# --- 画面レイアウト (2カラム) ---
cols = st.columns([1, 1])

# --- カラム1: カメラと操作ボタン ---
with cols[0]:
    st.subheader("カメラ映像")
    show_heatmap_toggle = st.checkbox("ヒートマップを表示", value=True)

    # WebRTCストリーマーの設定
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    ctx = webrtc_streamer(
        key="eyetracking",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor, # 上で定義したVideoProcessorクラスを使用
        async_processing=True,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
    )

    # VideoProcessorにUIの状態を渡す
    if ctx.state.playing and ctx.video_processor:
        ctx.video_processor.show_heatmap = show_heatmap_toggle
        ctx.video_processor.num_regions = 4 # 常に4領域

    # --- 操作ボタン ---
    c1, c2, c3 = st.columns(3)
    with c1:
        # 「分析開始」ボタン
        if st.button("分析開始", use_container_width=True):
            if ctx.state.playing:
                st.session_state.running = True
                if ctx.video_processor:
                    ctx.video_processor.num_regions = 4
                    ctx.video_processor.running = True
                    ctx.video_processor.timer = RegionTimer() # タイマーリセット
                    st.session_state.current_stats = ctx.video_processor.timer.stats.copy()
            else: st.warning("カメラを起動して下さい")
    with c2:
        # 「分析終了」ボタン
        if st.button("分析終了", use_container_width=True):
            if ctx.state.playing and st.session_state.running:
                st.session_state.running = False
                if ctx.video_processor:
                    ctx.video_processor.running = False
                    stats = ctx.video_processor.timer.stats
                    st.session_state.current_stats = stats
                    # 履歴に現在の分析結果を追加
                    st.session_state.history.append({
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "stats": stats.copy(),
                    })
            else: st.warning("分析が開始されていません")
    with c3:
        # 「キャリブ開始」ボタン
        if st.button("キャリブ開始", use_container_width=True):
            if ctx.state.playing:
                ctx.video_processor.start_calibration()
                st.success("キャリブを開始。各ターゲットを見て『サンプル取得』を押してください。")
            else: st.warning("カメラを起動して下さい")

    # --- キャリブレーション操作UI (キャリブ中のみ表示) ---
    if ctx.state.playing and ctx.video_processor and ctx.video_processor.calibrating:
        d1, d2, d3 = st.columns(3)
        with d1:
            # 「サンプル取得」ボタン
            if st.button("サンプル取得（15f）", use_container_width=True):
                if not ctx or not ctx.state.playing: st.warning("カメラを起動して下さい")
                elif not ctx.video_processor or not ctx.video_processor.calibrating: st.warning("まず『キャリブ開始』を押してください")
                else:
                    ctx.video_processor.request_sample(); st.info("取得中…（15フレーム）")
        with d2:
            # 「リセット」ボタン
            if st.button("リセット"): ctx.video_processor.reset_calibration()
        with d3:
            # 「適用」ボタン
            if st.button("適用"):
                ok, msg = ctx.video_processor.apply_calibration()
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

        # 取得状況の表示
        order = [t[0] for t in ctx.video_processor.targets]
        marks = ["✅" if name in ctx.video_processor.samples else "⬜" for name in order]
        st.caption("取得状況: " + " | ".join(f"{m} {n}" for m, n in zip(marks, order)))
        if ctx.video_processor.calib_ready: st.success("四隅のサンプルが揃いました。『適用』を押してください。")
        elif ctx.video_processor.target_idx < len(order): st.caption(f"次のターゲット: {order[ctx.video_processor.target_idx]}")

    # 分析中、UIの表示をリアルタイム更新
    if ctx.state.playing and ctx.video_processor and st.session_state.running:
        st.session_state.current_stats = ctx.video_processor.timer.stats

# --- カラム2: 分析結果 ---
with cols[1]:
    st.subheader("今回の分析結果 (合計時間)")
    cur = st.session_state.current_stats

    # --- 分析結果テーブル (3列) ---
    table_data = {
        "範囲": [],
        "標準動作 (秒)": [],
        "合計秒": []
    }
    for region_id in sorted(STANDARD_DURATIONS.keys()):
        label = region_labels.get(region_id, f"エリア {region_id}")
        standard_sec = STANDARD_DURATIONS.get(region_id, 0.0)
        actual_sec = cur.get(region_id, 0.0)

        table_data["範囲"].append(label)
        table_data["標準動作 (秒)"].append(f"{standard_sec:.1f}")
        table_data["合計秒"].append(f"{actual_sec:.1f}")

    st.table(table_data)
    # --- テーブルここまで ---


    # --- 動作比較レポート (テキスト) ---
    st.markdown("---")
    st.subheader("動作比較")
    standard_str, analysis_str = generate_analysis_report(cur, STANDARD_DURATIONS, region_labels)
    st.text("標準動作：")
    st.info(standard_str)
    st.text("分析動作：")
    st.success(analysis_str)
    # --- レポートここまで ---

    st.markdown("---")

    # --- JSONダウンロードボタン ---
    json_str = json.dumps({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_stats": {
            region_labels.get(i, f"Region {i}"): {"duration_sec": f"{cur.get(i, 0.0):.1f}"}
            for i in range(1, 4 + 1)
        },
        "analysis_report": {
            "standard": standard_str,
            "actual": analysis_str
        }
    }, ensure_ascii=False, indent=2)
    st.download_button(
        label="結果をJSONでダウンロード", data=json_str,
        file_name=f"gaze_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )

# --- 履歴セクション (カラムの外) ---
st.markdown("---")
st.subheader("履歴")

if not st.session_state.history:
    st.info("まだ履歴はありません。『分析開始』→『分析終了』で追加されます。")
else:
    # --- PDFダウンロードボタン ---
    if not os.path.exists(JAPANESE_FONT_PATH):
        st.error(f"日本語フォントファイルが見つかりません: {JAPANESE_FONT_PATH}\n"
                 "PDFをダウンロードするには、スクリプトと同じフォルダにフォントファイルを配置してください。")
    else:
        pdf_data = create_history_pdf(st.session_state.history) # PDF生成
        st.download_button(
            label="📈 全ての履歴をPDFでダウンロード",
            data=pdf_data,
            file_name=f"gaze_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )

    # --- 履歴を一件ずつ表示 ---
    for item in reversed(st.session_state.history):
        with st.expander(f"{item['time']} の結果"):
            stats = item.get("stats", {})
            num_regions_in_history = len(stats)
            if num_regions_in_history == 0: continue

            history_labels = get_region_labels(num_regions_in_history)

            # --- 履歴テーブル ---
            history_table_data = {
                "範囲": [],
                "標準動作 (秒)": [],
                "合計秒": []
            }
            is_4_region_history = (num_regions_in_history == 4)

            for i in sorted(stats.keys()):
                label = history_labels.get(i, f"エリア {i}")
                actual_sec = stats.get(i, 0.0)

                history_table_data["範囲"].append(label)
                history_table_data["合計秒"].append(f"{actual_sec:.1f}")

                if is_4_region_history:
                    standard_sec = STANDARD_DURATIONS.get(i, 0.0)
                    history_table_data["標準動作 (秒)"].append(f"{standard_sec:.1f}")
                else:
                    pass

            if not is_4_region_history:
                del history_table_data["標準動作 (秒)"]

            st.table(history_table_data)
            # --- 履歴テーブルここまで ---

            # --- 履歴レポート (テキスト) ---
            if num_regions_in_history == 4:
                _, analysis_str = generate_analysis_report(stats, STANDARD_DURATIONS, history_labels)
                st.text("分析動作：")
                st.info(analysis_str)
            # --- 履歴レポートここまで ---