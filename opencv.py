import cv2
import numpy as np
import tensorflow as tf
import time
# Prbamos la camara
cap = cv2.VideoCapture(0)

# Cargamos el modelo
model = tf.keras.models.load_model('./models/unet_personal.h5')

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        start_frame = time.time()
        # We have to resize the frame to the same size as the model (384x384)
        frame = cv2.resize(frame, (384, 384), interpolation=cv2.INTER_CUBIC)
        frame = np.expand_dims(frame, axis=0)  # Add a batch dimension
        prediction = model.predict(frame, verbose=0)
        prediction_t = (prediction > 0.5).astype(np.uint8)

        # Reduce the dimension of prediction from (1, 384, 384, 1) to (384, 384)
        prediction_t = np.squeeze(prediction_t, axis=0)
        prediction_t = np.squeeze(prediction_t, axis=-1)

        # Convert from grayscale to RGB (just for plotting)
        prediction_t_color = cv2.cvtColor(prediction_t * 255, cv2.COLOR_GRAY2RGB)


        # Overlay the mask on top of the frame
        alpha = 0.5  # You can adjust this value for transparency
        overlay = cv2.addWeighted(frame[0], alpha, prediction_t_color, 1 - alpha, 0)
    
        # Resize the overlayed image to fit the screen 1280x720
        overlay = cv2.resize(overlay, (1280, 720), interpolation=cv2.INTER_CUBIC)
        
        end_frame = time.time()

        # Calculate the fps
        fps = 1 / (end_frame - start_frame)

        # Display fps on frame
        cv2.putText(overlay, f'FPS: {fps:.2f}', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', overlay)        


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
