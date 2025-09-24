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
    LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    LEFT_IRIS = [474, 475, 476, 477, 468, 469, 470, 471, 472]
    RIGHT_IRIS = [469, 470, 471, 472, 473, 474, 475, 476, 477]

    def __init__(self, det_conf=0.5, trk_conf=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=det_conf,
            min_tracking_confidence=trk_conf,
        )
        # ---- キャリブ保存（raw_vecの基準値 & バイアス計算用の平均eye座標） ----
        self.rx_left: Optional[float] = None
        self.rx_right: Optional[float] = None
        self.ry_top: Optional[float] = None
        self.ry_bottom: Optional[float] = None
        # バイアス（ピクセル）: gx = raw_x * scale_x + offx, gy = raw_y * scale_y + offy
        self.offx_px: float = 0.0
        self.offy_px: float = 0.0
        # キャリブ済みフラグ
        self.calibrated: bool = False
        # 参考: キャリブ時の画面サイズ（オフセット再現用）
        self.base_w: Optional[int] = None
        self.base_h: Optional[int] = None

    @staticmethod
    def _center(landmarks, idxs):
        pts = np.array([[landmarks[i].x, landmarks[i].y] for i in idxs if i < len(landmarks)])
        if len(pts) == 0:
            return None
        return pts.mean(axis=0)

    def _iris_center_circle_px(self, landmarks, indices, w, h):
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices if i < len(landmarks)]
        if len(pts) < 3:
            return None
        pts_np = np.array(pts, dtype=np.float32)
        (cx, cy), _ = cv2.minEnclosingCircle(pts_np)
        return (cx, cy)  # pixels

    def set_calibration_values(
        self,
        rx_left: float, rx_right: float,
        ry_top: float, ry_bottom: float,
        offx_px: float, offy_px: float,
        base_w: int, base_h: int
    ):
        """左右・上下のraw_vec基準と、ピクセルオフセット、基準解像度を保存"""
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
        戻り: ok, eye_px, end_px, raw_vec_meas
        - raw_vec_meas = (rx, ry) = iris_center - eye_center [0..1正規化座標系]
        - キャリブ済みなら:
            scale_x = (w - 10) / (rx_right - rx_left)
            scale_y = (h - 10) / (ry_bottom - ry_top)
            gx = rx * scale_x + offx_px
            gy = ry * scale_y + offy_px
          未キャリブ時は簡易既定（スケールのみ）で動作
        """
        h, w, _ = frame_bgr.shape
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return False, None, None, None
        lm = res.multi_face_landmarks[0].landmark

        # 目中心
        lc = self._center(lm, self.LEFT_EYE)
        rc = self._center(lm, self.RIGHT_EYE)
        if lc is None or rc is None:
            return False, None, None, None
        eye_center = (lc + rc) / 2.0
        eye_px = (int(eye_center[0] * w), int(eye_center[1] * h))

        # 虹彩中心（円あてはめ）
        li_px = self._iris_center_circle_px(lm, self.LEFT_IRIS, w, h)
        ri_px = self._iris_center_circle_px(lm, self.RIGHT_IRIS, w, h)
        if li_px is None or ri_px is None:
            return False, None, None, None
        iris_px = ((li_px[0] + ri_px[0]) * 0.5, (li_px[1] + ri_px[1]) * 0.5)

        # 生raw_vec
        iris_center = np.array([iris_px[0] / w, iris_px[1] / h])
        eye_center_n = np.array([eye_center[0], eye_center[1]])
        # 「目の中心」から「虹彩（ひとみ）の中心」に向かうベクトル
        raw_vec = iris_center - eye_center_n  # (rx, ry)
        # スケール＆オフセット
        if self.calibrated and self.rx_left is not None and self.rx_right is not None \
           and self.ry_top is not None and self.ry_bottom is not None:
            # 毎フレームの解像度から動的に算出
            dx = (self.rx_right - self.rx_left)
            dy = (self.ry_bottom - self.ry_top)
            # 安全
            if abs(dx) < 1e-9 or abs(dy) < 1e-9:
                scale_x = (w - 10) / 0.01  # フォールバック
                scale_y = (h - 10) / 0.01
            else:
                scale_x = (w - 10) / dx
                scale_y = (h - 10) / dy
            gx = raw_vec[0] * scale_x + self.offx_px
            gy = raw_vec[1] * scale_y + self.offy_px
        else:
            # 未キャリブの簡易既定（任意の暫定値）
            scale_x = (w - 10) / 0.01
            scale_y = (h - 10) / 0.001
            gx = raw_vec[0] * scale_x
            gy = raw_vec[1] * scale_y

        end_px = (
            int(np.clip(eye_px[0] + gx, 10, w - 10)),
            int(np.clip(eye_px[1] + gy, 10, h - 10)),
        )
        return True, eye_px, end_px, (float(raw_vec[0]), float(raw_vec[1]))


# ================== Region 集計 ==================
class RegionTimer:
    def __init__(self, num_regions: int):
        self.num_regions = num_regions
        # 指定されたリージョン数でstats辞書を初期化
        self.stats = {i: 0.0 for i in range(1, self.num_regions + 1)}
        self.last_ts = None

    def region_of(self, x: int, y: int, w: int, h: int) -> int:
        if self.num_regions == 4:
            cx, cy = w // 2, h // 2
            if x < cx and y < cy: return 1
            if x >= cx and y < cy: return 2
            if x < cx and y >= cy: return 3
            return 4
        elif self.num_regions == 3:
            # 縦に3分割
            if x < w / 3: return 1
            if x < 2 * w / 3: return 2
            return 3
        return 1

    def update(self, gaze_px: Tuple[int, int], w: int, h: int):
        now = time.time()
        if self.last_ts is None:
            self.last_ts = now
            return
        dt = now - self.last_ts
        r = self.region_of(gaze_px[0], gaze_px[1], w, h)
        self.stats[r] += dt
        self.last_ts = now

def y_intercept(rx_left, rx_right, w):
    if abs(rx_right - rx_left) < 1e-9: return 0, 0
    a = w / (rx_right - rx_left)
    b = -(w/2) * (rx_right + rx_left) / (rx_right - rx_left)
    return a, b

# ================== ★★★ ヒートマップ生成クラス (新規追加) ★★★ ==================
class HeatmapGenerator:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.heatmap_data = np.zeros((height, width), dtype=np.float64)
        self.decay_rate = 0.98
        self.intensity = 5.0
        self.radius = int(max(width, height) * 0.15) # ミニマップ内での半径を少し大きく

    def update(self, gaze_point: Tuple[int, int]):
        x, y = gaze_point
        if x is None or y is None: return

        # NumPyのスライシングとmeshgridを使って高速にガウス分布を適用
        x_grid, y_grid = np.ogrid[:self.height, :self.width]
        dist_sq = (x_grid - y)**2 + (y_grid - x)**2

        gauss = self.intensity * np.exp(-dist_sq / (2 * (self.radius / 2)**2))
        self.heatmap_data += gauss

    def decay(self):
        self.heatmap_data *= self.decay_rate
        self.heatmap_data[self.heatmap_data < 0.01] = 0

    def generate_minimap_image(self, current_gaze_point: Optional[Tuple[int, int]]) -> np.ndarray:
        # 1. ミニマップの背景を作成
        minimap_bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.rectangle(minimap_bg, (0, 0), (self.width, self.height), (30, 30, 30), -1)

        # 2. ヒートマップを描画
        if np.max(self.heatmap_data) > 0:
            norm_heatmap = self.heatmap_data / np.max(self.heatmap_data)
            heatmap_8u = (norm_heatmap * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_8u, cv2.COLORMAP_JET)
            # ヒートマップを背景に重ね合わせ
            minimap_bg = cv2.addWeighted(minimap_bg, 0.5, heatmap_color, 0.5, 0)

        # 3. グリッド線を描画
        for i in range(1, 4):
            x = (i * self.width) // 4
            cv2.line(minimap_bg, (x, 0), (x, self.height), (80, 80, 80), 1)
        for i in range(1, 3):
            y = (i * self.height) // 3
            cv2.line(minimap_bg, (0, y), (self.width, y), (80, 80, 80), 1)

        # 4. 現在の視線位置を描画 (赤い点)
        if current_gaze_point:
            cv2.circle(minimap_bg, current_gaze_point, 5, (0, 0, 255), -1)
            cv2.circle(minimap_bg, current_gaze_point, 7, (255, 255, 255), 1)

        # 5. 外枠を描画
        cv2.rectangle(minimap_bg, (0, 0), (self.width - 1, self.height - 1), (255, 255, 255), 1)

        return minimap_bg

# ================ Video Processor（キャリブ・ヒートマップ対応） =================
class VideoProcessor:
    def __init__(self):
        self.est = GazeEstimator(0.5, 0.5)
        self.running = False

        # ---- キャリブレーション状態 ----
        # 四隅: 左上→右上→左下→右下
        self.targets = [
            ("左上", 0.10, 0.10),
            ("右上", 0.90, 0.10),
            ("左下", 0.10, 0.90),
            ("右下", 0.90, 0.90),
        ]
        self.calibrating = False
        self.target_idx = 0
        self.capture_request = False
        self.capture_buffer: List[Tuple[float, float, int, int]] = []  # (rx, ry, eye_x, eye_y)
        # サンプル保存: name -> 平均 (rx, ry, eye_x, eye_y)
        self.samples: Dict[str, Tuple[float, float, float, float]] = {}
        # 直近のフレームサイズ
        self.last_w: Optional[int] = None
        self.last_h: Optional[int] = None
        self.calib_ready = False  # ★ 全点取得済みかどうか
        # FPS計測用の変数
        self.fps_start_time = 0
        self.fps_frame_count = 0
        self.fps = 0.0

        # ★★★ ヒートマップ関連の変数を追加 ★★★
        self.minimap_width = 220
        self.minimap_height = 140
        self.heatmap_generator = HeatmapGenerator(self.minimap_width, self.minimap_height)
        self.show_heatmap = True # UIから変更される

        # UIから渡されるのを待つため、デフォルト値で初期化
        self.num_regions = st.session_state.get("num_regions", 4)
        self.timer = RegionTimer(self.num_regions)


    def start_calibration(self):
        self.calibrating = True
        self.calib_ready = False
        self.target_idx = 0
        self.capture_request = False
        self.capture_buffer.clear()
        self.samples.clear()

    def request_sample(self):
        if self.calibrating and self.target_idx < len(self.targets):
            self.capture_request = True
            self.capture_buffer.clear()

    def skip_target(self):
        if self.calibrating and self.target_idx < len(self.targets):
            self.target_idx += 1
            self.capture_request = False
            self.capture_buffer.clear()
            if self.target_idx >= len(self.targets):
                self.calib_ready = True  # 全点終わり

    def reset_calibration(self):
        self.start_calibration()

    def _compute_and_apply_calibration(self) -> Tuple[bool, str]:
        """
        四隅のサンプルから:
          水平レンジ: rx_left (=左上/左下の平均) と rx_right (=右上/右下の平均)
          垂直レンジ: ry_top  (=左上/右上の平均) と ry_bottom(=左下/右下の平均)
        バイアス:
          offx = 平均( target_x_left - eye_x_left - rx_left * scale_x )
          offy = 平均( target_y_top  - eye_y_top  - ry_top  * scale_y )
        """
        need = {"左上", "右上", "左下", "右下"}
        if not need.issubset(self.samples.keys()):
            return False, "四隅のサンプルが揃っていません。"
        if self.last_w is None or self.last_h is None:
            return False, "カメラから映像が取得できていません。少し待ってから再度試してください。"

        # 平均を取り出し
        rx_lu, ry_lu, ex_lu, ey_lu = self.samples["左上"]
        rx_ru, ry_ru, ex_ru, ey_ru = self.samples["右上"]
        rx_ld, ry_ld, ex_ld, ey_ld = self.samples["左下"]
        rx_rd, ry_rd, ex_rd, ey_rd = self.samples["右下"]

        # 横レンジ（左群/右群）
        rx_left = (rx_lu + rx_ld) / 2.0
        rx_right = (rx_ru + rx_rd) / 2.0
        # 縦レンジ（上群/下群）
        ry_top = (ry_lu + ry_ru) / 2.0
        ry_bottom = (ry_ld + ry_rd) / 2.0

        # スケール計算に使う現在のフレームサイズ
        w, h = self.last_w, self.last_h
        dx = rx_right - rx_left
        dy = ry_bottom - ry_top
        if abs(dx) < 1e-9 or abs(dy) < 1e-9:
            return False, "差分が小さすぎます。サンプルを取り直してください。"
        scale_x = y_intercept(rx_left, rx_right, w)[0]
        scale_y = y_intercept(ry_top, ry_bottom, h)[0]
        offx = y_intercept(rx_left, rx_right, w)[1]
        offy = y_intercept(ry_top, ry_bottom, h)[1]
        # scale_x, offx = y_intercept(rx_left, rx_right, w)
        # scale_y, offy = y_intercept(ry_top, ry_bottom, h)
        # if scale_x is None: return False, "y_intercept calculation failed for x"
        # if scale_y is None:
        #     # Fallback or specific logic for ry_top/ry_bottom if needed
        #     ry_dx = ry_bottom - ry_top
        #     if abs(ry_dx) < 1e-9: return False, "Vertical difference too small"
        #     scale_y = h / ry_dx
        #     offy = - (h/2) * (ry_bottom + ry_top) / ry_dx

        # 推定値を保存（スケールは estimate() 内で毎フレーム w,h から再計算）
        self.est.set_calibration_values(
            rx_left=rx_left, rx_right=rx_right,
            ry_top=ry_top, ry_bottom=ry_bottom,
            offx_px=float(offx), offy_px=float(offy),
            base_w=w, base_h=h
        )
        return True, (f"適用 OK: dx={dx:.6g}, dy={dy:.6g}, "
                      f"offx={offx:.1f}px, offy={offy:.1f}px, "
                      f"scale_x={scale_x:.1f}, scale_y={scale_y:.1f} @ {w}x{h},"
                      f" samples={rx_left, rx_right, ry_top, ry_bottom}")


    def _draw_guides(self, frame):
        h, w, _ = frame.shape
        if self.num_regions == 4:
            cv2.line(frame, (w//2, 0), (w//2, h), (0, 0, 255), 2)
            cv2.line(frame, (0, h//2), (w, h//2), (0, 0, 255), 2)
        elif self.num_regions == 3:
            cv2.line(frame, (w//3, 0), (w//3, h), (0, 0, 255), 2)
            cv2.line(frame, (2*w//3, 0), (2*w//3, h), (0, 0, 255), 2)

    def _draw_calib_target(self, frame):
        if not self.calibrating:
            return
        h, w, _ = frame.shape

        if self.calib_ready or self.target_idx >= len(self.targets):
            return
        name, nx, ny = self.targets[self.target_idx]
        px, py = int(nx * w), int(ny * h)
        cv2.circle(frame, (px, py), 16, (0, 255, 0), 3)
        cv2.circle(frame, (px, py), 6, (0, 255, 255), -1)

    def draw_overlay(self, frame: np.ndarray, eye_px, end_px):
        if eye_px and end_px:
            cv2.circle(frame, eye_px, 6, (255, 255, 255), -1)
            cv2.circle(frame, end_px, 6, (0, 255, 255), -1)
            steps = 8
            for i in range(steps):
                t = i / (steps - 1)
                x = int(eye_px[0] + t * (end_px[0] - eye_px[0]))
                y = int(eye_px[1] + t * (end_px[1] - eye_px[1]))
                cv2.circle(frame, (x, y), max(2, 6 - i), (0, 200, 255), -1)
        return frame

    # ---- WebRTC フレーム処理 ----
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        self.last_w, self.last_h = img.shape[1], img.shape[0]
        # print("b",img.shape[1], img.shape[0])

        ok, eye_px, end_px, raw_vec = self.est.estimate(img)
        # FPSを計算
        if self.fps_start_time == 0:
            self.fps_start_time = time.time()
        self.fps_frame_count += 1
        elapsed_time = time.time() - self.fps_start_time
        if elapsed_time >= 1.0: # 1秒ごとに更新
            self.fps = self.fps_frame_count / elapsed_time
            self.fps_frame_count = 0
            self.fps_start_time = time.time()

        # キャリブ収集
        if self.calibrating and self.capture_request and ok and raw_vec:
            self.capture_buffer.append((raw_vec[0], raw_vec[1], eye_px[0], eye_px[1]))
            if len(self.capture_buffer) >= 15:
                avg = np.mean(self.capture_buffer, axis=0)
                name, _, _ = self.targets[self.target_idx]
                self.samples[name] = tuple(float(v) for v in avg)
                self.capture_request = False
                self.capture_buffer.clear()
                self.target_idx += 1
                if self.target_idx >= len(self.targets): self.calib_ready = True

        # ★★★ ミニマップ用に視線座標をスケーリングして更新 ★★★
        current_minimap_gaze = None
        if self.running and ok and end_px:
            self.timer.update(end_px, w, h)
            if self.show_heatmap:
                # 画面全体の座標をミニマップ座標に変換
                minimap_x = int((end_px[0] / w) * self.minimap_width)
                minimap_y = int((end_px[1] / h) * self.minimap_height)
                current_minimap_gaze = (minimap_x, minimap_y)
                self.heatmap_generator.update(current_minimap_gaze)

        # ★★★ ヒートマップを減衰させ、ミニマップを生成して貼り付け ★★★
        if self.show_heatmap:
            self.heatmap_generator.decay()
            minimap_image = self.heatmap_generator.generate_minimap_image(current_minimap_gaze)

            # 画面右下にミニマップを配置
            margin = 10
            map_h, map_w, _ = minimap_image.shape
            y_offset = h - map_h - margin
            x_offset = w - map_w - margin

            # NumPyスライシングで画像を合成
            img[y_offset:y_offset + map_h, x_offset:x_offset + map_w] = minimap_image

        # 描画処理
        img = self.draw_overlay(img, eye_px if ok else None, end_px if ok else None)
        self._draw_guides(img)

        if self.calibrating and not self.calib_ready and self.target_idx < len(self.targets):
            name, nx, ny = self.targets[self.target_idx]
            px, py = int(nx * w), int(ny * h)
            cv2.circle(img, (px, py), 16, (0, 255, 0), 3)
            cv2.circle(img, (px, py), 6, (0, 255, 255), -1)

        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

        # self._draw_guides(img)
        # self._draw_calib_target(img)

        # fps_text = f"FPS: {self.fps:.1f}"
        # cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # return av.VideoFrame.from_ndarray(img, format="bgr24")


    # UI側から呼ぶ
    def apply_calibration(self) -> Tuple[bool, str]:
        return self._compute_and_apply_calibration()

def get_region_labels(num_regions: int) -> Dict[int, str]:
    """リージョン数に応じたラベルを返す"""
    if num_regions == 4:
        return {1: "① 左上", 2: "② 右上", 3: "③ 左下", 4: "④ 右下"}
    if num_regions == 3:
        return {1: "① 左", 2: "② 中央", 3: "③ 右"}
    # 他のリージョン数が必要な場合はここに追加
    return {i: f"エリア {i}" for i in range(1, num_regions + 1)}


# ================= PDF=================
JAPANESE_FONT_PATH = "ipaexg00401/ipaexg.ttf"

def create_history_pdf(history_data: list) -> io.BytesIO:
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('Japanese', '', JAPANESE_FONT_PATH, uni=True)
    pdf.set_font('Japanese', '', 16)
    pdf.cell(0, 10, "アイトラッキング分析履歴", 0, 1, 'C')
    pdf.ln(10)
    for item in reversed(history_data):
        stats = item.get("stats", {})
        num_regions = len(stats)
        if num_regions == 0: continue
        labels = get_region_labels(num_regions)

        pdf.set_font('Japanese', '', 12)
        pdf.cell(0, 10, f"記録日時: {item['time']}", 0, 1)
        pdf.set_font('Japanese', '', 10)
        pdf.set_fill_color(240, 240, 240)
        cell_width = 60; cell_height = 8
        pdf.cell(cell_width, cell_height, "範囲", border=1, fill=True, align='C')
        pdf.cell(cell_width, cell_height, "合計秒", border=1, fill=True, align='C')
        pdf.ln()
        for i in range(1, num_regions + 1):
            pdf.cell(cell_width, cell_height, labels.get(i, f"Region {i}"), border=1)
            pdf.cell(cell_width, cell_height, f"{stats.get(i, 0.0):.1f}", border=1)
            pdf.ln()
        pdf.ln(10)
    return io.BytesIO(pdf.output())


# ================= Streamlit UI ==================
st.set_page_config(page_title="簡易アイトラッキング（ヒートマップ版）", layout="wide")
st.title("簡易アイトラッキング")

# セッションステート初期化
if "history" not in st.session_state:
    st.session_state.history = []
if "num_regions" not in st.session_state:
    st.session_state.num_regions = 4 # デフォルト値
if "running" not in st.session_state:
    st.session_state.running = False

def on_region_change():
    st.session_state.num_regions = st.session_state.selectbox_regions
    st.session_state.current_stats = {i: 0.0 for i in range(1, st.session_state.num_regions + 1)}


num_regions = st.sidebar.selectbox(
    "分析エリアの分割数を選択", [3, 4],
    index=[3, 4].index(st.session_state.num_regions),
    on_change=on_region_change,
    key='selectbox_regions'
)

if "current_stats" not in st.session_state or len(st.session_state.current_stats) != st.session_state.num_regions:
    st.session_state.current_stats = {i: 0.0 for i in range(1, st.session_state.num_regions + 1)}

region_labels = get_region_labels(st.session_state.num_regions)



cols = st.columns([1, 1])

with cols[0]:
    st.subheader("カメラ映像")
    # ★★★ UIにヒートマップのチェックボックスを追加 ★★★
    show_heatmap_toggle = st.checkbox("ヒートマップを表示", value=True)

    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    ctx = webrtc_streamer(
        key="eyetracking",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        async_processing=True,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
    )

    # ★★★ チェックボックスの状態をVideoProcessorに渡す ★★★
    if ctx.state.playing and ctx.video_processor:
        ctx.video_processor.show_heatmap = show_heatmap_toggle
        # VideoProcessorのリージョン数をUIの選択に同期
        if ctx.video_processor.num_regions != st.session_state.num_regions:
            ctx.video_processor.num_regions = st.session_state.num_regions
            ctx.video_processor.timer = RegionTimer(st.session_state.num_regions)


    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("分析開始", use_container_width=True):
            if ctx.state.playing:
                st.session_state.running = True
                if ctx.video_processor:
                    # タイマーを現在のリージョン数でリセット
                    ctx.video_processor.num_regions = st.session_state.num_regions
                    ctx.video_processor.running = True
                    ctx.video_processor.timer = RegionTimer(st.session_state.num_regions)
                    st.session_state.current_stats = ctx.video_processor.timer.stats.copy()
            else: st.warning("カメラを起動して下さい")
    with c2:
        if st.button("分析終了", use_container_width=True):
            if ctx.state.playing and st.session_state.running:
                st.session_state.running = False
                if ctx.video_processor:
                    ctx.video_processor.running = False
                    stats = ctx.video_processor.timer.stats
                    st.session_state.current_stats = stats
                    st.session_state.history.append({
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "stats": stats.copy(),
                    })
            else: st.warning("分析が開始されていません")
    with c3:
        if st.button("キャリブ開始", use_container_width=True):
            if ctx.state.playing:
                ctx.video_processor.start_calibration()
                st.success("キャリブを開始。各ターゲットを見て『サンプル取得』を押してください。")
            else: st.warning("カメラを起動して下さい")

    # 進捗の行を順序通りに
    if ctx.state.playing and ctx.video_processor and ctx.video_processor.calibrating:
        d1, d2, d3 = st.columns(3)
        with d1:
            if st.button("サンプル取得（15f）", use_container_width=True):
                if not ctx or not ctx.state.playing:
                    st.warning("カメラを起動して下さい")
                elif not ctx.video_processor or not ctx.video_processor.calibrating:
                    st.warning("まず『キャリブ開始』を押してください")
                else:
                    ctx.video_processor.request_sample()
                    st.info("取得中…（15フレーム）")
        with d2:
            if st.button("リセット"):
                ctx.video_processor.reset_calibration()
        with d3:
            if st.button("適用"):
                ok, msg = ctx.video_processor.apply_calibration()
                if ok: st.success(msg)
                else: st.error(msg)

        order = [t[0] for t in ctx.video_processor.targets]
        marks = ["✅" if name in ctx.video_processor.samples else "⬜" for name in order]
        st.caption("取得状況: " + " | ".join(f"{m} {n}" for m, n in zip(marks, order)))
        if ctx.video_processor.calib_ready:
            st.success("四隅のサンプルが揃いました。『適用』を押してください。")
        elif ctx.video_processor.target_idx < len(order):
            st.caption(f"次のターゲット: {order[ctx.video_processor.target_idx]}")

    if ctx.state.playing and ctx.video_processor and st.session_state.running:
        st.session_state.current_stats = ctx.video_processor.timer.stats
with cols[1]:
    st.subheader("今回の分析結果")
    cur = st.session_state.current_stats
    # ★ テーブル表示を動的に生成
    table_rows = [(region_labels.get(i, f"エリア {i}"), cur.get(i, 0.0)) for i in range(1, num_regions + 1)]
    st.table({"範囲": [r[0] for r in table_rows], "合計秒": [f"{r[1]:.1f}" for r in table_rows]})

    # ★ JSON出力を動的に生成
    json_str = json.dumps({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results": {
            region_labels.get(i, f"Region {i}"): {"duration_sec": f"{cur.get(i, 0.0):.1f}"}
            for i in range(1, num_regions + 1)
        }
    }, ensure_ascii=False, indent=2)
    st.download_button(
        label="結果をJSONでダウンロード", data=json_str,
        file_name=f"gaze_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )

st.markdown("---")
st.subheader("履歴")

if not st.session_state.history:
    st.info("まだ履歴はありません。『分析開始』→『分析終了』で追加されます。")
else:
    if not os.path.exists(JAPANESE_FONT_PATH):
        st.error(f"日本語フォントファイルが見つかりません: {JAPANESE_FONT_PATH}\n"
                 "PDFをダウンロードするには、スクリプトと同じフォルダにフォントファイルを配置してください。")
    else:
        pdf_data = create_history_pdf(st.session_state.history)
        st.download_button(
            label="📈 全ての履歴をPDFでダウンロード",
            data=pdf_data,
            file_name=f"gaze_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )

    for item in reversed(st.session_state.history):
        with st.expander(f"{item['time']} の結果"):
            stats = item.get("stats", {})
            num_regions_in_history = len(stats)
            if num_regions_in_history == 0: continue
            history_labels = get_region_labels(num_regions_in_history)
            rows = [(history_labels.get(i, f"エリア {i}"), stats.get(i, 0.0)) for i in history_labels]
            st.table({"範囲": [r[0] for r in rows], "合計秒": [f"{r[1]:.1f}" for r in rows]})
