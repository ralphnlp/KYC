from crypt import methods
import time
import os
import cv2
import torch
from flask import request, Flask, jsonify
from models import Card_Detector, Face_Card, MTCNNet, FaceNet

app = Flask(__name__)

card_detector = Card_Detector("./weights/card_weight.onnx")
face_card_detector = Face_Card("./weights/face_card_weight.onnx")
mtcnn  = MTCNNet()
facenet = FaceNet()


@app.route('/detect_face_card', methods=['POST'])
def detect_face_card():

    input_img_path = request.form.get('input_img_path')
    face_path= './process_bin/anchor_face.jpg'

    input_img = cv2.imread(input_img_path)
    card = card_detector.predict(input_img)
    if card is not None:
        w = 640
        h = int(w*card.shape[0] / card.shape[1])
        card = cv2.resize(card, (w, h))
        _, boxes = face_card_detector.predict(card.copy())

        if len(boxes) == 1:
            _, confidences, [top_left, bottom_right] = boxes[0]
            if confidences >= 0.8:
                face = card[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]].copy()
                cv2.imwrite(face_path, face)
                return jsonify({'face_path': face_path})
    return jsonify({'face_path': None})


@app.route('/matching_face', methods=['POST'])
def matching_face():

    anchor_face_path='./process_bin/anchor_face.jpg'
    flag = request.form.get('flag')

    with torch.no_grad():
        if flag == True:
            pass
        else:            
            anchor_face = cv2.imread(anchor_face_path)
            anchor_feature = facenet.extract(anchor_face)
            anchor_feature = torch.as_tensor(anchor_feature)

        frame_path = request.form.get('frame_path')
        frame = cv2.imread(frame_path)
        boxes = mtcnn.detect_face(frame)
        if boxes is not None:
            for box in boxes:
                xmin, ymin, xmax, ymax = [int(i) for i in box]
                frame_face = frame[ymin:ymax, xmin:xmax].copy()
                frame_feature = facenet.extract(frame_face)
                cosine = torch.cosine_similarity(anchor_feature, torch.as_tensor(frame_feature), dim=0).numpy()
                return jsonify({'cosine_sim': float(cosine), 'bbox': list([[xmin, ymin], [xmax, ymax]])})
        return jsonify({'cosine_sim': None})


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.2', port=8080)