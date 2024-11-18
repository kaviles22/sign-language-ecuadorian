import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

# Cargar el modelo entrenado y el diccionario de etiquetas
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 
    17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

# Inicializar Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Variables para el seguimiento del tiempo y la predicción
current_letter = None
start_time = None

# Inicializar FastAPI
app = FastAPI()

def process_frame():
    """Procesar los fotogramas y dibujar predicciones."""
    global current_letter, start_time

    cap = cv2.VideoCapture(0)
    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # imagen en la que dibujar
                    hand_landmarks,  # salida del modelo
                    mp_hands.HAND_CONNECTIONS,  # conexiones de la mano
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            if predicted_character == current_letter:
                # Si la predicción es la misma, verificar el tiempo
                if time.time() - start_time >= 3:
                    # Dibujar la letra en la parte inferior
                    cv2.putText(frame, predicted_character, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4,
                                cv2.LINE_AA)
            else:
                # Si la predicción cambia, actualizar la letra y reiniciar el temporizador
                current_letter = predicted_character
                start_time = time.time()

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.get("/video-feed")
def video_feed():
    """Transmitir la alimentación de video con predicciones."""
    return StreamingResponse(process_frame(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
def home():
    """Punto de inicio."""
    return {"message": "¡Bienvenido a la API de reconocimiento de gestos de mano!"}
