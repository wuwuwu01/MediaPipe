import cv2
import mediapipe as mp
import time
import tkinter as tk
from PIL import Image, ImageTk
import threading
import random
from tkinter import font as tkFont

mp_drawing = mp.solutions.drawing_utils  #绘图模块
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands  #手部关键点检测
mp.solutions.face_mesh  # 人脸网状检测
mp.solutions.face_detection  # 人脸识别
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                 max_num_faces=2,
                                 refine_landmarks=True,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

from tkinter import messagebox
import numpy as np
import math
from deepface import DeepFace


def emotion(frame):
    # 在视频帧上分析与显示情绪，但还未显示出来
    for result in DeepFace.analyze(frame, detector_backend="mtcnn", actions=['emotion'], enforce_detection=False):
        emotion = result["dominant_emotion"]
        region = result["region"]

        # 绘制人脸矩形
        cv2.rectangle(
            frame, (region["x"], region["y"]),
            (region["x"] + region["w"], region["y"] + region["h"]),
            (128, 0, 128), 1
        )

        # 填充情绪文字区域颜色
        cv2.rectangle(
            frame, (region["x"] - 135, region["y"]-10),
            (region["x"] - 135 + len(f"Emotion: {emotion}") * 10, region["y"] + 10),
            (211, 211, 211), -1
        )

        # 绘制情绪文字
        cv2.putText(frame,
                    f"Emotion: {emotion}",
                    (region["x"] - 135, region["y"]+5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA)

def face(image):
    cv2.destroyAllWindows()
    cap = cv2.VideoCapture(0)  # 打开默认摄像头
    frame_count = 0  # 帧计数器
    processing_interval = 5  # 设置处理的帧间隔
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    while True:# cap.isOpened():
        success, frame = cap.read()  # 读取视频帧
        if not success:
            break
        #frame_count += 1
        #if frame_count % processing_interval == 0:
        # 在这里添加代码进行面部识别和情绪判断
        # 将图像从 BGR 格式转换为 RGB 格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 进行面部识别
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 在这里处理面部特征点，并进行情绪判断
                # 在图像上绘制面部特征点
                #print(len(face_landmarks.landmark))   #有478个关键点
                mp_drawing.draw_landmarks(image=frame, landmark_list=face_landmarks,
                                        # 选取关键点
                                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                                        # 绘制关键点，若为None，表示不绘制关键点，也可以指定点的颜色、粗细、半径
                                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0),thickness=1,circle_radius=1),
                                        # 绘制连接线的格式
                                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
          
        emotion(frame)  
                
        cv2.imshow("Face Emotion Detection/ Pressing the 'Tab' key to quit./", frame)  # 显示视频帧

        if cv2.waitKey(100) & 0xFF == 9:   # 按下 'q' 键退出循环
            # 关闭窗口
            cv2.destroyAllWindows()
            
            # 释放摄像头资源
            cap.release()
            break


class GestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils
    def is_fist(self, hand_landmarks):
        # 判断拳头的逻辑
        # 手指指尖靠近MCP关节
        close_fingers = True
        for finger_tip, finger_mcp in zip(
            [self.mp_hands.HandLandmark.THUMB_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
             self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.RING_FINGER_TIP,
             self.mp_hands.HandLandmark.PINKY_TIP],
            [self.mp_hands.HandLandmark.THUMB_MCP, self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
             self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP, self.mp_hands.HandLandmark.RING_FINGER_MCP,
             self.mp_hands.HandLandmark.PINKY_MCP]):
            tip = hand_landmarks.landmark[finger_tip]
            mcp = hand_landmarks.landmark[finger_mcp]
            if tip.y > mcp.y:
                close_fingers = False
                break
        return close_fingers
    
    def determine_winner(self, player, computer):
        # 这个方法判断游戏结果
        if player == computer:
            return "平局"
        elif (player == "石头" and computer == "剪刀") or \
             (player == "剪刀" and computer == "布") or \
             (player == "布" and computer == "石头"):
            return "你赢了"
        else:
            return "你输了"
    def is_ok_gesture(self, hand_landmarks):
        # 获取大拇指和食指的指尖坐标
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # 计算大拇指和食指指尖之间的距离
        distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

        # 判断大拇指和食指是否靠近
        if distance < 0.05:  # 阈值可能需要根据实际情况调整
            # 检查其他手指是否伸直
            middle_straight = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < \
                              hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
            ring_straight = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y < \
                            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP].y
            pinky_straight = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y < \
                             hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP].y

            return middle_straight and ring_straight and pinky_straight

        return False
    def is_scissors(self, hand_landmarks):
        # 判断剪刀手的逻辑
        # 食指和中指伸直，其他手指弯曲
        index_straight = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < \
                         hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y
        middle_straight = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < \
                          hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
        ring_bent = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y > \
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP].y
        pinky_bent = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y > \
                     hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP].y

        return index_straight and middle_straight and ring_bent and pinky_bent
    def is_palm_open(self, hand_landmarks):
            # 获取手指指尖和MCP关节的坐标
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

            thumb_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
            index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
            middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            ring_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
            pinky_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]

            # 检查所有手指的指尖是否都高于对应的MCP关节
            is_open = all([
                thumb_tip.y < thumb_mcp.y,
                index_tip.y < index_mcp.y,
                middle_tip.y < middle_mcp.y,
                ring_tip.y < ring_mcp.y,
                pinky_tip.y < pinky_mcp.y
            ])
            return is_open

    def is_thumb_up(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        thumb_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        # 检测拇指指尖是否高于其它两个关键点。
        return thumb_tip.y < thumb_ip.y < thumb_mcp.y < index_mcp.y
    def count_extended_fingers(self, hand_landmarks):
        extended_fingers = 0
        finger_tips = [self.mp_hands.HandLandmark.THUMB_TIP,
                        self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                        self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        self.mp_hands.HandLandmark.RING_FINGER_TIP,
                        self.mp_hands.HandLandmark.PINKY_TIP]
        finger_mcp = [self.mp_hands.HandLandmark.THUMB_MCP,
                        self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
                        self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                        self.mp_hands.HandLandmark.RING_FINGER_MCP,
                        self.mp_hands.HandLandmark.PINKY_MCP]

        for tip, mcp in zip(finger_tips, finger_mcp):
            tip_position = hand_landmarks.landmark[tip]
            mcp_position = hand_landmarks.landmark[mcp]
            if tip_position.y < mcp_position.y:
                extended_fingers += 1

        return extended_fingers
    def run(self,window):
        cap = cv2.VideoCapture(0)
        time_palm_open = None
        time_thumb_up = None
        time_fist = None
        time_scissors = None
        end = False
        game=False
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            results = self.hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    if game:
                        player_gesture =None
                        if self.is_scissors(hand_landmarks):#v
                            player_gesture="剪刀"
                        elif self.is_palm_open(hand_landmarks):
                            player_gesture="布"
                        else:
                            player_gesture="石头"
                        computer_gesture = random.choice(["石头", "剪刀", "布"])
                        window.update(lambda: window.display_text(f"电脑出了：{computer_gesture}"))
                        time.sleep(2)

                        result = self.determine_winner(player_gesture, computer_gesture)
                        window.update(lambda: window.display_text(result))
                        time.sleep(3)

                        game=False
                    if self.is_palm_open(hand_landmarks):                            
                        if time_palm_open is None:
                            time_palm_open = time.time()
                        elif time.time() - time_palm_open > 4:
                            window.update(lambda: window.display_text("程序即将结束"))
                            window.root.after(1500, window.root.destroy)
                            end=True
                        elif time.time() - time_palm_open > 2:
                            window.update(lambda: window.display_text("手掌张开超过2秒,再持续2秒结束程序"))
                    else:
                        time_palm_open = None
                    
                    # 检测拇指向上
                    if self.is_thumb_up(hand_landmarks):
                        if time_thumb_up is None:
                            time_thumb_up = time.time()
                        elif time.time() - time_thumb_up > 0.5:
                            window.update(lambda: window.display_text("手势为点赞，游戏开始"))
                            time.sleep(2)
                            time_thumb_up = None
                            for count in ["3", "2", "1"]:
                                window.update(lambda: window.display_text(count))
                                time.sleep(1)
                            game=True
                            break
                    else:
                        time_thumb_up = None
                    """ extended_fingers = self.count_extended_fingers(hand_landmarks)
                    if extended_fingers == 1:
                        window.update(lambda: window.display_text("1"))
                    elif extended_fingers == 2:
                        window.update(lambda: window.display_text("2"))
                    elif extended_fingers == 3:
                        window.update(lambda: window.display_text("3"))
                    elif extended_fingers == 4:
                        window.update(lambda: window.display_text("4")) """

                    # 检测ok
                    if self.is_ok_gesture(hand_landmarks):
                        if time_fist is None:
                            time_fist = time.time()
                        elif time.time() - time_fist > 0.5:
                            window.update(lambda: window.display_text("手势为ok"))
                            
                            face(image)
                            cv2.imshow('MediaPipe Hands', image)
                            self.run(window)
                            time_fist = None
                    else:
                        time_fist = None

                    # 检测剪刀手
                    if self.is_scissors(hand_landmarks):
                        if time_scissors is None:
                            time_scissors = time.time()
                        elif time.time() - time_scissors > 0.5:
                            window.update(lambda: window.display_text("手势为v"))
                            time_scissors = None
                    else:
                        time_scissors = None
                

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27 or end:
                break


        cap.release()
        cv2.destroyAllWindows()

class DisplayWindow:
    def __init__(self, title="Display Window", width=300, height=250):
        self.root = tk.Tk()
        self.root.title(title)
        self.width = width
        self.height = height
        self.root.geometry(f"{width}x{height}")

         # 更新上方的固定标签
        gesture_instructions = (
            "比 OK 实现面部识别\n"
            "点赞来局猜拳 3\n"
            "张开手掌四秒结束程序"
        )
        self.label_top = tk.Label(self.root, text=gesture_instructions, justify=tk.LEFT)
        self.label_top.pack()

        # 创建下方的更新标签
        self.label_bottom = tk.Label(self.root, text="")
        self.label_bottom.pack(expand=True)

        self.root.attributes('-topmost', True)

    def display_text(self, text):
        customFont = tkFont.Font(family="Helvetica", size=12)
        self.label_bottom.config(text=text, font=customFont)

    def display_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize((100, 100), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        self.label_bottom.config(image=photo)
        self.label_bottom.image = photo

    def update(self, func, *args):
        self.root.after(0, func, *args)

    def run(self):
        self.display_text("开始检测手势")
        self.root.mainloop()


window = DisplayWindow()
# 在后台线程中运行手势识别
recognizer = GestureRecognizer()
thread = threading.Thread(target=recognizer.run,args=(window,))
thread.start()
# 在主线程中运行Tkinter的mainloop
window.run()