import json
import math
import time
from datetime import datetime
from typing import Dict, Tuple

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from fpdf import FPDF
import os
import io
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

    @staticmethod
    def _center(landmarks, idxs):
        pts = np.array([[landmarks[i].x, landmarks[i].y] for i in idxs if i < len(landmarks)])
        if len(pts) == 0:
            return None
        return pts.mean(axis=0)

    def estimate(self, frame_bgr: np.ndarray):
        """戻り値: ok, eye_px, gaze_end_px
        ok=False の場合は検出失敗
        """
        h, w, _ = frame_bgr.shape
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return False, None, None
        lm = res.multi_face_landmarks[0].landmark
        lc = self._center(lm, self.LEFT_EYE)
        rc = self._center(lm, self.RIGHT_EYE)
        li = self._center(lm, self.LEFT_IRIS)
        ri = self._center(lm, self.RIGHT_IRIS)
        if lc is None or rc is None or li is None or ri is None:
            return False, None, None

        eye_center = (lc + rc) / 2.0
        iris_center = (li + ri) / 2.0
        raw_vec = iris_center - eye_center
        # スケーリングと緩やかな非線形補正
        gx = raw_vec[0] * 1200.0
        gy = raw_vec[1] * 2000.0
        gx = math.copysign(abs(gx) ** 0.9, gx)
        gy = math.copysign(abs(gy) ** 0.9, gy)
        eye_px = (int(eye_center[0] * w), int(eye_center[1] * h))
        scale = 180
        end_px = (
            int(np.clip(eye_px[0] + gx * scale, 10, w - 10)),
            int(np.clip(eye_px[1] + gy * scale, 10, h - 10)),
        )
        return True, eye_px, end_px

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


# ================ Video Processor =================
class VideoProcessor:
    def __init__(self):
        self.est = GazeEstimator(0.5, 0.5)
        self.timer = RegionTimer()
        self.running = False

    def draw_overlay(self, frame: np.ndarray, eye_px, end_px):
        h, w, _ = frame.shape
        # 4分割のガイド
        cv2.line(frame, (w//2, 0), (w//2, h), (0, 0, 255), 2)
        cv2.line(frame, (0, h//2), (w, h//2), (0, 0, 255), 2)
        if eye_px and end_px:
            cv2.circle(frame, eye_px, 6, (255, 255, 255), -1)
            cv2.circle(frame, end_px, 6, (0, 255, 255), -1)
            # 簡易ヒートライン
            steps = 8
            for i in range(steps):
                t = i / (steps - 1)
                x = int(eye_px[0] + t * (end_px[0] - eye_px[0]))
                y = int(eye_px[1] + t * (end_px[1] - eye_px[1]))
                cv2.circle(frame, (x, y), max(2, 6 - i), (0, 200, 255), -1)
        return frame

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        ok, eye_px, end_px = self.est.estimate(img)
        if self.running and ok and end_px is not None:
            self.timer.update(end_px, img.shape[1], img.shape[0])
        img = self.draw_overlay(img, eye_px if ok else None, end_px if ok else None)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 日本語フォントファイルのパス (例: 'ipaexg.ttf')
# スクリプトと同じディレクトリにフォントファイルを配置してください。
JAPANESE_FONT_PATH = "ipaexg00401/ipaexg.ttf"

def create_history_pdf(history_data: list) -> io.BytesIO:
    """履歴データからPDFを生成してメモリ上のファイルオブジェクトとして返す"""
    pdf = FPDF()
    pdf.add_page()

    # 日本語フォントを追加
    pdf.add_font('Japanese', '', JAPANESE_FONT_PATH, uni=True)
    pdf.set_font('Japanese', '', 16)

    # タイトル
    pdf.cell(0, 10, "アイトラッキング分析履歴", 0, 1, 'C')
    pdf.ln(10)

    # 各履歴項目をテーブルとして描画
    for item in reversed(history_data):
        pdf.set_font('Japanese', '', 12)
        pdf.cell(0, 10, f"記録日時: {item['time']}", 0, 1)

        # テーブルヘッダー
        pdf.set_font('Japanese', '', 10)
        pdf.set_fill_color(240, 240, 240)
        cell_width = 60
        cell_height = 8
        pdf.cell(cell_width, cell_height, "範囲", border=1, fill=True, align='C')
        pdf.cell(cell_width, cell_height, "合計秒", border=1, fill=True, align='C')
        pdf.ln()

        # テーブルデータ
        s = item["stats"]
        rows = [("① 左上", s[1]), ("② 右上", s[2]), ("③ 左下", s[3]), ("④ 右下", s[4])]
        for row in rows:
            pdf.cell(cell_width, cell_height, row[0], border=1)
            pdf.cell(cell_width, cell_height, f"{row[1]:.1f} 秒", border=1)
            pdf.ln()

        pdf.ln(10) # 各履歴の間にスペースを空ける

    # データをメモリ上のファイルとして返すように変更
    return io.BytesIO(pdf.output())# ================= Streamlit UI ==================
st.set_page_config(page_title="簡易アイトラッキング", layout="wide")
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

    start, stop = st.columns(2)
    with start:
        if st.button("分析開始", use_container_width=True):
            if not ctx or not ctx.state.playing:
                st.warning("カメラを起動して下さい")
            else:
                st.session_state.running = True
                if ctx.video_processor:
                    ctx.video_processor.running = True
                    ctx.video_processor.timer = RegionTimer()
    with stop:
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

    if ctx.video_processor and st.session_state.running:
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

# (ここから変更・追加)
if not st.session_state.history:
    st.info("まだ履歴はありません。『分析開始』→『分析終了』で追加されます。")
else:
    # PDFダウンロードボタンの追加
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

    # 履歴リストの表示
    for item in reversed(st.session_state.history):
        with st.expander(f"{item['time']} の結果"):
            s = item["stats"]
            rows = [("① 左上", s[1]), ("② 右上", s[2]), ("③ 左下", s[3]), ("④ 右下", s[4])]
            st.table({"範囲": [r[0] for r in rows], "合計秒": [f"{r[1]:.1f}" for r in rows]})