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
    def __init__(self):
        self.stats = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        self.last_ts = None

    @staticmethod
    def region_of(x: int, y: int, w: int, h: int) -> int:
        cx, cy = w // 2, h // 2
        if x < cx and y < cy:
            return 1
        if x >= cx and y < cy:
            return 2
        if x < cx and y >= cy:
            return 3
        return 4

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
    a = w / (rx_right - rx_left)  # 傾き
    b = -(w/2) * (rx_right + rx_left) / (rx_right - rx_left)  # y切片
    return a, b

# ================ Video Processor（キャリブ対応） =================
class VideoProcessor:
    def __init__(self):
        self.est = GazeEstimator(0.5, 0.5)
        self.timer = RegionTimer()
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

    # ---- 描画 ----
    def _draw_guides(self, frame):
        h, w, _ = frame.shape
        cv2.line(frame, (w//2, 0), (w//2, h), (0, 0, 255), 2)
        cv2.line(frame, (0, h//2), (w, h//2), (0, 0, 255), 2)

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

        ok, eye_px, end_px, raw_vec = self.est.estimate(img)


        # キャリブ収集
        if self.calibrating and self.capture_request and ok and raw_vec is not None:
            rx, ry = raw_vec
            self.capture_buffer.append((rx, ry, eye_px[0], eye_px[1]))
            if len(self.capture_buffer) >= 15:
                avg = np.mean(self.capture_buffer, axis=0)  # (rx, ry, ex, ey)
                name, _, _ = self.targets[self.target_idx]
                # ★上書きOK（取り直し対応）
                self.samples[name] = (float(avg[0]), float(avg[1]), float(avg[2]), float(avg[3]))
                # ★次のターゲットへ
                self.capture_request = False
                self.capture_buffer.clear()
                self.target_idx += 1
                if self.target_idx >= len(self.targets):
                    self.calib_ready = True

        if self.running and ok and end_px is not None:
            self.timer.update(end_px, img.shape[1], img.shape[0])

        img = self.draw_overlay(img, eye_px if ok else None, end_px if ok else None)
        self._draw_guides(img)
        self._draw_calib_target(img)

        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


    # UI側から呼ぶ
    def apply_calibration(self) -> Tuple[bool, str]:
        return self._compute_and_apply_calibration()


# ================= PDF（そのまま） =================
JAPANESE_FONT_PATH = "ipaexg00401/ipaexg.ttf"

def create_history_pdf(history_data: list) -> io.BytesIO:
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('Japanese', '', JAPANESE_FONT_PATH, uni=True)
    pdf.set_font('Japanese', '', 16)
    pdf.cell(0, 10, "アイトラッキング分析履歴", 0, 1, 'C')
    pdf.ln(10)
    for item in reversed(history_data):
        pdf.set_font('Japanese', '', 12)
        pdf.cell(0, 10, f"記録日時: {item['time']}", 0, 1)
        pdf.set_font('Japanese', '', 10)
        pdf.set_fill_color(240, 240, 240)
        cell_width = 60; cell_height = 8
        pdf.cell(cell_width, cell_height, "範囲", border=1, fill=True, align='C')
        pdf.cell(cell_width, cell_height, "合計秒", border=1, fill=True, align='C')
        pdf.ln()
        s = item["stats"]
        rows = [("① 左上", s[1]), ("② 右上", s[2]), ("③ 左下", s[3]), ("④ 右下", s[4])]
        for row in rows:
            pdf.cell(cell_width, cell_height, row[0], border=1)
            pdf.cell(cell_width, cell_height, f"{row[1]:.1f} 秒", border=1)
            pdf.ln()
        pdf.ln(10)
    return io.BytesIO(pdf.output())


# ================= Streamlit UI ==================
st.set_page_config(page_title="簡易アイトラッキング（キャリブ版）", layout="wide")
st.title("簡易アイトラッキング")

# セッションステート初期化
if "history" not in st.session_state:
    st.session_state.history = []
if "current_stats" not in st.session_state:
    st.session_state.current_stats = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
if "running" not in st.session_state:
    st.session_state.running = False

cols = st.columns([1, 1])

with cols[0]:
    st.subheader("カメラ映像")
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    ctx = webrtc_streamer(
        key="eyetracking",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        async_processing=True,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("分析開始", use_container_width=True):
            if not ctx or not ctx.state.playing:
                st.warning("カメラを起動して下さい")
            else:
                st.session_state.running = True
                if ctx.video_processor:
                    ctx.video_processor.running = True
                    ctx.video_processor.timer = RegionTimer()
    with c2:
        if st.button("分析終了", use_container_width=True):
            if not ctx or not ctx.state.playing:
                st.warning("カメラを起動して下さい")
            elif not st.session_state.running:
                st.warning("まだ分析は開始されていません")
            else:
                st.session_state.running = False
                if ctx.video_processor:
                    ctx.video_processor.running = False
                    stats = ctx.video_processor.timer.stats
                    st.session_state.current_stats = stats
                    st.session_state.history.append({
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "stats": stats.copy(),
                    })
    with c3:
        if st.button("キャリブ開始", use_container_width=True):
            if not ctx or not ctx.state.playing:
                st.warning("カメラを起動して下さい")
            else:
                ctx.video_processor.start_calibration()
                st.success("キャリブを開始。左上→右上→左下→右下で『サンプル取得』を押してください。")

    # キャリブ操作
    if ctx and ctx.video_processor and ctx.video_processor.calibrating:
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
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
            if ctx and ctx.video_processor and ctx.video_processor.calibrating:
                # 進捗の行を順序通りに
                order = [t[0] for t in ctx.video_processor.targets]
                marks = ["✅" if name in ctx.video_processor.samples else "⬜" for name in order]
                st.caption("取得状況: " + " | ".join([f"{m} {n}" for m, n in zip(marks, order)]))

                # 次のターゲット名
                if ctx.video_processor.calib_ready:
                    st.success("四隅のサンプルが揃いました。『適用』を押してください。")
                else:
                    next_name = order[ctx.video_processor.target_idx] if ctx.video_processor.target_idx < len(order) else "—"
                    st.caption(f"次のターゲット: {next_name}")


    # ラン中は最新値反映
    if ctx and ctx.video_processor and st.session_state.running:
        st.session_state.current_stats = ctx.video_processor.timer.stats

with cols[1]:
    st.subheader("今回の分析結果（どこを何秒見たか）")
    cur = st.session_state.current_stats
    table_rows = [("① 左上", cur[1]), ("② 右上", cur[2]), ("③ 左下", cur[3]), ("④ 右下", cur[4])]
    st.table({"範囲": [r[0] for r in table_rows], "合計秒": [f"{r[1]:.1f}" for r in table_rows]})

    json_obj = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "結果": {
            "対象範囲：①": {"合計秒数": f"{cur[1]:.1f}s"},
            "対象範囲：②": {"合計秒数": f"{cur[2]:.1f}s"},
            "対象範囲：③": {"合計秒数": f"{cur[3]:.1f}s"},
            "対象範囲：④": {"合計秒数": f"{cur[4]:.1f}s"},
        },
    }
    st.download_button(
        label="今回の結果をJSONでダウンロード",
        data=json.dumps(json_obj, ensure_ascii=False, indent=2),
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
            s = item["stats"]
            rows = [("① 左上", s[1]), ("② 右上", s[2]), ("③ 左下", s[3]), ("④ 右下", s[4])]
            st.table({"範囲": [r[0] for r in rows], "合計秒": [f"{r[1]:.1f}" for r in rows]})
