import streamlit as st
import asyncio
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
import av
from imutils import face_utils
import dlib
import cv2
import numpy as np

st.title("Streamlit WebRTC using DLIB")
st.write("This is a sample to integrate DLIB :D ")

p = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


class VideoProcessor(VideoProcessorBase):
    @staticmethod
    def draw_points(image, face_landmarks, start_point, end_point, is_closed=False):
        points = []
        for i in range(start_point, end_point + 1):
            point = [face_landmarks.part(i).x, face_landmarks.part(i).y]
            points.append(point)

        points = np.array(points, dtype=np.int32)
        cv2.polylines(
            image, [points], is_closed, (255, 200, 0), thickness=2, lineType=cv2.LINE_8
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for (i, rect) in enumerate(rects):
