import streamlit as st
import cv2
import os
import pandas as pd
from deepface import DeepFace
from datetime import datetime
from fer import FER
import matplotlib.pyplot as plt
import cv2

st.set_page_config(page_title="AI Attendance System", layout="wide") 

@st.cache_resource
def load_models():
    from fer import FER
    return FER(mtcnn=True)

emotion_model = load_models()


emotion_detector = FER(mtcnn=True) 

# ================= INIT =================
LOG_PATH = "logs/attendance.csv"
if not os.path.exists("logs"):
    os.makedirs("logs")
if not os.path.exists(LOG_PATH):
    pd.DataFrame(columns=["Name", "Time", "Emotion"]).to_csv(LOG_PATH, index=False)

# ============ FUNCTION: FACE VERIFY ============
def verify_face(image_path, known_faces_folder="dataset"):
    try:
        result = DeepFace.find(
            img_path=image_path,
            db_path=known_faces_folder,
            enforce_detection=False,
            detector_backend="opencv"
        )
        if len(result) > 0 and not result[0].empty:
            identity_path = result[0].iloc[0]["identity"]
            name = identity_path.split(os.sep)[-2]
            return name
    except Exception as e:
        st.warning(f"Face recognition error: {str(e)}")
    return "Unknown"

# ========== FUNCTION: EMOTION DETECTION ==========
def detect_emotion(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "Neutral"
        emotion, score = emotion_model.top_emotion(img)
        if emotion:
            st.write(f"🎯 Emotion: {emotion} ({score*100:.2f}%)")
            return emotion
        return "Neutral"
    except Exception as e:
        st.warning(f"Emotion detection error: {str(e)}")
        return "Neutral"

# ========= FUNCTION: ATTENDANCE LOG ==========
def log_attendance(name, emotion):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    df = pd.read_csv(LOG_PATH)

    if not ((df['Name'] == name) & (df['Time'].str.contains(now.strftime("%Y-%m-%d")))).any():
        new_row = pd.DataFrame([[name, timestamp, emotion]], columns=["Name", "Time", "Emotion"])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(LOG_PATH, index=False)

# ================= STREAMLIT UI =================


st.title("📸 Face Recognition + Emotion-Based Attendance System")
st.sidebar.title("📂 Menu")
st.sidebar.image("https://i.imgur.com/6M513NZ.png", width=100)  # Optional logo
st.sidebar.markdown("### 🤖 AI Attendance System")
st.sidebar.markdown("Built by **Barath**  \nAI/ML Engineer in the making ")
st.sidebar.markdown("---")


mode = st.sidebar.radio("Choose Mode", ["🎥 Webcam", "📤 Upload Image", "➕ Register New Face", "📋 View Attendance Log"])


# ============ MODE 1: Webcam ================
if mode == "🎥 Webcam":
    st.subheader("Live Webcam Recognition")
    if st.button("Start Camera"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to access webcam")
                break

            frame = cv2.flip(frame, 1)
            cv2.imwrite("temp.jpg", frame)

            name = verify_face("temp.jpg")
            emotion = detect_emotion("temp.jpg")
            log_attendance(name, emotion)

            cv2.putText(frame, f"{name} - {emotion}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            stframe.image(frame, channels="BGR")

        cap.release()
        cv2.destroyAllWindows()

# ============ MODE 2: Upload Image ============
elif mode == "📤 Upload Image":
    st.subheader("Upload an Image")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.read())

        st.image("uploaded_image.jpg", caption="Uploaded Image", use_column_width=True)
        name = verify_face("uploaded_image.jpg")
        emotion = detect_emotion("uploaded_image.jpg")
        log_attendance(name, emotion)
        st.success(f"✅ Detected: {name} | Emotion: {emotion}")

# ============ MODE 3: Register New Face ============
elif mode == "➕ Register New Face":
    st.subheader("Register a New Face")

    new_name = st.text_input("Enter name to register")
    register = st.button("Start Camera to Register")

    if register and new_name.strip() != "":
        folder_path = f"dataset/{new_name}"
        os.makedirs(folder_path, exist_ok=True)

        cap = cv2.VideoCapture(0)
        st.info("Press 'S' to Save Image | 'Q' to Quit")

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Camera Error")
                break

            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "Press 'S' to Save | 'Q' to Quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.imshow("Register Face", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                img_path = os.path.join(folder_path, f"{count}.jpg")
                cv2.imwrite(img_path, frame)
                st.success(f"✅ Saved: {img_path}")
                count += 1

            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    elif register:
        st.warning("Please enter a name.")

# ============ MODE 4: View Log ============
elif mode == "📋 View Attendance Log":
    st.subheader("Attendance Log")
    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        st.dataframe(df)
    else:
        st.info("No attendance logged yet.")
