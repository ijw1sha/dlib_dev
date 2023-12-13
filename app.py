import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer
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
            s = predictor(gray, rect)
            s = face_utils.shape_to_np(s)

            if i == 0:
                print("Total number of face landmarks detected ", len(s))

                self.draw_points(img, s, 0, 16)  # Jaw line
                self.draw_points(img, s, 17, 21)  # Left eyebrow
                self.draw_points(img, s, 22, 26)  # Right eyebrow
                self.draw_points(img, s, 27, 30)  # Nose bridge
                self.draw_points(img, s, 30, 35, True)  # Lower nose
                self.draw_points(img, s, 36, 41, True)  # Left eye
                self.draw_points(img, s, 42, 47, True)  # Right Eye
                self.draw_points(img, s, 48, 59, True)  # Outer lip
                self.draw_points(img, s, 60, 67, True)  # Inner lip
                for (i, y) in s:
                    cv2.circle(img, (i, y), 2, (0, 255, 0), -1)

        st.image(img, channels="BGR", use_column_width=True)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="dlib",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
)
