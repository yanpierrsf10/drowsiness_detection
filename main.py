#################################################################################
# ACME&CIA
# Title : Drownsiness detection.
# Description : Ths method is based on nodding (head position) and eye blinking.
#################################################################################

#Import necessary libraries
import cv2
import mediapipe as mp
import numpy as np
import time

#FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5,max_num_faces = 1)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

#Frame Properties
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)

#Index of landmarks
index_left_eye = [33, 160, 158, 133, 153, 144]
index_right_eye = [362, 385, 387, 263, 373, 380]
index_face = [33, 263, 1, 61, 291, 199, 8]

#EAR function
def eye_aspect_ratio(coordinates):
    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))
    d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))
    d_C = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))
    return (d_A + d_B) / (2 * d_C)

#Get landmark image coordinates
def get_pixels(landmark):
    lx = int(face_landmarks.landmark[landmark].x * img_w)
    ly = int(face_landmarks.landmark[landmark].y * img_h)
    lz = face_landmarks.landmark[landmark].z
    return lx, ly, lz

while cap.isOpened():

    success, image = cap.read()
    start = time.time()
    
    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    
    # To improve performance
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #Initialize values
    img_h, img_w, img_c = image.shape
    nose_2d = []
    nose_3d = []
    face_3d = []
    face_2d = []
    lear = []
    rear = []
    coordinates_left_eye = []
    coordinates_right_eye = []

    #Extracting landmarks
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for index in index_left_eye:
                x, y, z = get_pixels(index)
                coordinates_left_eye.append([x, y])
            for index in index_right_eye:
                x, y, z = get_pixels(index)
                coordinates_right_eye.append([x, y])
            for index in index_face:
                x, y, z = get_pixels(index)
                face_2d.append([x, y])
                face_3d.append([x, y, z])
                if index == 1 :
                    nose_2d = (x,y)
                    nose_3d = (x,y,z*3000)

            #Eye Aspect Ratio (EAR) of each eye
            left_ear = eye_aspect_ratio(coordinates_left_eye)         
            right_ear = eye_aspect_ratio(coordinates_right_eye)
            
            #Get the average EAR
            avg_ear = (left_ear+right_ear)/2
    
            # Convert face landmarks coordinates to NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert face landmarks coordinates to NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])
            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            a_x = angles[0] * 360
            a_y = angles[1] * 360
            a_z = angles[2] * 360

            # See where the user's head tilting
            if a_x < -7.5 or avg_ear < 0.25:
                text = "Take a break"
            else:
                text = "Keep going"

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + a_y * 10) , int(nose_2d[1] - a_x * 10))
            
            cv2.line(image, p1, p2, (255, 0, 0), 3)

            # Add the text on the image
            cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(image, "x: " + str(np.round(a_x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (222, 222, 22), 2)
            cv2.putText(image, "y: " + str(np.round(a_y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (222, 222, 22), 2)
            cv2.putText(image, "z: " + str(np.round(a_z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (222, 222, 22), 2)
            cv2.putText(image, str(np.round(left_ear,2)), (75, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (147, 0, 255), 2)
            cv2.putText(image, str(np.round(right_ear,2)), (500, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (147, 0, 255), 2)
            cv2.putText(image, str(np.round(avg_ear,2)), (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (147, 0, 255), 2)
            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime

        cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)


    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()