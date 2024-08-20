import cv2
import numpy as np
import onnxruntime as ort
import os
import pandas

COLLECT_DATASET = False
INFERENCE = True
IMAGE_PATH = "imgs"
DATA_FILE = "pose_dataset.csv"

keypoints_map = {
    "nose": 0,
    "leftEye": 1,
    "rightEye": 2,
    "leftEar": 3,
    "rightEar": 4,
    "leftShoulder": 5,
    "rightShoulder": 6,
    "leftElbow": 7,
    "rightElbow": 8,
    "leftWrist": 9,
    "rightWrist": 10,
    "leftHip": 11,
    "rightHip": 12,
    "leftKnee": 13,
    "rightKnee": 14,
    "leftAnkle": 15,
    "rightAnkle": 16
}

onnx_model_path = 'movenet_singlepose_lightning_4.onnx'
ort_session = ort.InferenceSession(onnx_model_path)

cls_model_path = 'angle features/classifier.onnx'
ort_cls_session = ort.InferenceSession(cls_model_path)

id_ctr = 0
collected_data = []
collected_data_angles = []

def calculate_angle(A, B, C):
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)

    BA = A - B
    BC = C - B

    angle_rad = np.arctan2(BC[1], BC[0]) - np.arctan2(BA[1], BA[0])
    angle_rad = angle_rad % (2 * np.pi)
    
    if angle_rad < 0:
        angle_rad += 2 * np.pi

    return angle_rad

def preprocess(image):
    h, w = image.shape[:2]
    target_size = 192
    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_w = target_size
        new_h = int(target_size / aspect_ratio)
    else:
        new_h = target_size
        new_w = int(target_size * aspect_ratio)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top_padding = (target_size - new_h) // 2
    bottom_padding = target_size - new_h - top_padding
    left_padding = (target_size - new_w) // 2
    right_padding = target_size - new_w - left_padding

    resized_image = cv2.copyMakeBorder(
        resized_image, 
        top_padding, bottom_padding, left_padding, right_padding, 
        cv2.BORDER_CONSTANT, 
        value=[0, 0, 0]
    )
    
    input_tensor = resized_image.astype(np.int32)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor

def draw_keypoints_and_angle(image, keypoints):
    needed_kp = ["rightHip", "leftHip", "rightShoulder", "leftShoulder", "rightElbow", "leftElbow", "rightWrist", "leftWrist"]
    angles_kp = [["rightHip", "rightShoulder", "rightElbow"], ["leftHip", "leftShoulder", "leftElbow"], ["rightShoulder", "rightElbow","rightWrist"], ["leftShoulder", "leftElbow","leftWrist"]]
    threshold = 0.4
    angles = list()

    for i in range(0, len(keypoints), 3):
        x, y, confidence = keypoints[i], keypoints[i + 1], keypoints[i + 2]
        if confidence > threshold:
            cv2.circle(image, (int(y * image.shape[1]), int(x * image.shape[0])), 5, (0, 255, 0), -1)
    
    for triple in angles_kp:
        triple_kp = []
        for joint in triple:
            if keypoints[keypoints_map[joint]*3+2] > threshold:
                triple_kp.append(keypoints[keypoints_map[joint]*3:keypoints_map[joint]*3+2])

        if len(triple_kp) == 3:
            angle = calculate_angle(*triple_kp)
            angles.append(angle)

            angle_deg = np.degrees(angle)
            
            triple_kp_pixels = [(int(kp[1] * image.shape[1]), int(kp[0] * image.shape[0])) for kp in triple_kp]

            cv2.line(image, triple_kp_pixels[0], triple_kp_pixels[1], (255, 0, 0), 2)
            cv2.line(image, triple_kp_pixels[1], triple_kp_pixels[2], (255, 0, 0), 2)

            text_pos = triple_kp_pixels[1]
            text_pos = (text_pos[0], text_pos[1]-20)

            cv2.putText(image, f'{angle_deg:.2f}', text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        else:
            angles.append(-1.0) # Missing angle 

    missing_angles = angles.count(-1.0)

    if missing_angles == 0 and COLLECT_DATASET:
        global id_ctr
        img_path = os.path.join(IMAGE_PATH, f"img_{id_ctr}.jpg")
        kps = [keypoints[keypoints_map[kp]*3:keypoints_map[kp]*3+2] for kp in needed_kp]
        kps = [coord for kp in kps for coord in kp]
        collected_data.append([f"img_{id_ctr}.jpg", *kps, np.NaN])
        collected_data_angles.append([f"img_{id_ctr}.jpg", *angles, np.NaN])

        id_ctr+=1

        cv2.imwrite(img_path, image)
    
    if missing_angles < 2 and INFERENCE:
        # CLS inference
        input_tensor = np.array(angles, dtype=np.float32)
        input_tensor = np.expand_dims(input_tensor, 0)
        
        ort_inputs = {ort_cls_session.get_inputs()[0].name: input_tensor}
        ort_outs = ort_cls_session.run(None, ort_inputs)

        prediction = ort_outs[0][0][0]

        text_pos = (30,30)

        cv2.putText(image, f'Ready Pose = {prediction > 0}', text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    return image


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess(frame)

    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    ort_outs = ort_session.run(None, ort_inputs)

    keypoints = ort_outs[0].flatten()

    output_frame = draw_keypoints_and_angle(frame, keypoints)

    cv2.imshow('Pose Estimation', output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        if COLLECT_DATASET:
            df = pandas.DataFrame(collected_data, columns=["image_name", "rightHip_x", "rightHip_y", "leftHip_x", "leftHip_y", "rightShoulder_x", "rightShoulder_y", "leftShoulder_x", "leftShoulder_y", "rightElbow_x", "rightElbow_y", "leftElbow_x", "leftElbow_y", "rightWrist_x", "rightWrist_y", "leftWrist_x", "leftWrist_y", "label"])
            df.to_csv("kp features/kp_"+DATA_FILE)
            df = pandas.DataFrame(collected_data_angles, columns=["image_name", "angle_rightHip_rightShoulder_rightElbow", "angle_leftHip_leftShoulder_leftElbow", "angle_rightShoulder_rightElbow_rightWrist","angle_leftShoulder_leftElbow_leftWrist", "label"])
            df.to_csv("angles features/angles_"+DATA_FILE)
        break

cap.release()
cv2.destroyAllWindows()