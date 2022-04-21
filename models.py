import numpy as np
import cv2
import base64
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms

def affine(img, coord):
    
    top_left, bottom_left, bottom_right = [np.asarray(element) for element in coord]
    new_w = int(np.linalg.norm(bottom_left-bottom_right, 2))
    new_h = int(np.linalg.norm(top_left-bottom_left, 2))

    pts1 = np.float32([top_left, bottom_left, bottom_right])
    pts2 = np.float32([[0, 0], [0, new_h], [new_w, new_h]])

    M = cv2.getAffineTransform(pts1,pts2)
    new_img = cv2.warpAffine(img, M, (new_w, new_h))
    return new_img


class MTCNNet:
    mtcnn = MTCNN()

    def __init__(self) -> None:
        self.threshold = 0.9
        
    def detect_face(self, img):
        boxes, probs = self.mtcnn.detect(img)
        if boxes is not None:
            boxes_ = []
            for i, prob in enumerate(probs):
                if prob  >= self.threshold:
                    boxes_.append(boxes[i])
            return boxes_
        return None


class FaceNet:
    
    face_net = InceptionResnetV1(pretrained='vggface2').eval()
    def __init__(self) -> None:
        pass

    def extract(self, face):
        face = cv2.resize(face, (160, 160))
        face = transforms.ToTensor()(face)
        feature = self.face_net(face.unsqueeze(0))
        feature = feature.detach().numpy()[0]
        return feature


class Yolo5:
    
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 640
    SCORE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.45
    CONFIDENCE_THRESHOLD = 0.45

    def __init__(self, weight_path, classes = None):
        self.weight_path = weight_path
        self.classes = classes
        self.net = cv2.dnn.readNet(weight_path)
        return self.net

    def pre_process(self, input_image):
      blob = cv2.dnn.blobFromImage(input_image, 1/255,  (self.INPUT_WIDTH, self.INPUT_HEIGHT), [0,0,0], 1, crop=False)
      self.net.setInput(blob)
      outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
      return outputs


    def post_process(self, input_image, outputs):

        class_ids = []
        confidences = []
        boxes = []
        rows = outputs[0].shape[1]
        image_height, image_width = input_image.shape[:2]
        x_factor = image_width / self.INPUT_WIDTH
        y_factor =  image_height / self.INPUT_HEIGHT

        for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]
            if confidence >= self.CONFIDENCE_THRESHOLD:
                classes_scores = row[5:]
                class_id = np.argmax(classes_scores)
                if (classes_scores[class_id] > self.SCORE_THRESHOLD):
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    cx, cy, w, h = row[0], row[1], row[2], row[3]
                    left = int((cx - w/2) * x_factor)
                    top = int((cy - h/2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
        outputs = []

        for i in indices:
            box = np.asarray(boxes[i], dtype=np.int16)
            top_left = [box[0], box[1]]
            bottom_right = [box[0]+box[2], box[1]+box[3]]
            outputs.append([class_ids[i], confidences[i], [top_left, bottom_right]])
            cv2.rectangle(input_image, top_left, bottom_right, color=(255, 255, 0), thickness=1)

        return input_image, outputs


    def predict(self, input_image):
        outputs = self.pre_process(input_image)
        outputs = self.post_process(input_image, outputs)
        return outputs


class Card_Detector(Yolo5):

    def __init__(self, weight_path):
        self.classes = ['bottom_left', 'bottom_right', 'top_right', 'top_left']
        super().__init__(weight_path, self.classes)
        

    def post_process(self, input_image, outputs):

        class_ids = []
        confidences = []
        boxes = []
        rows = outputs[0].shape[1]
        image_height, image_width = input_image.shape[:2]
        x_factor = image_width / self.INPUT_WIDTH
        y_factor =  image_height / self.INPUT_HEIGHT

        for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]
            if confidence >= self.CONFIDENCE_THRESHOLD:
                classes_scores = row[5:]
                class_id = np.argmax(classes_scores)
                if (classes_scores[class_id] > self.SCORE_THRESHOLD):
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    cx, cy, w, h = row[0], row[1], row[2], row[3]
                    left = int((cx - w/2) * x_factor)
                    top = int((cy - h/2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        copy_img = input_image.copy()
        bag = {}
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)

        if len(indices) >= 3:

            for i in indices:
                box = boxes[i]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]   
                
                cv2.rectangle(copy_img, (left, top), (left+width, top+height), (0, 0, 255), thickness=1)
                cls, conf = self.classes[class_ids[i]], confidences[i]
                if cls not in bag:
                    bag[cls] = [left, top, width, height, np.linalg.norm(np.asarray([left, top]), 2), conf]
                else:
                    if conf > bag[cls][-1]:
                        bag[cls] = [left, top, width, height, np.linalg.norm(np.asarray([left, top]), 2), conf]
            
            flag = True
            for cls in [self.classes[0], self.classes[1], self.classes[3]]:
                if cls not in bag.keys():
                    flag = False
                    break

            if flag == True:
                
                keys = np.asarray(list(bag.keys()))
                values = np.asarray(list(bag.values()))

                sorted_index = np.argsort(values[:, 4], axis=0)
                keys, values = keys[sorted_index], values[sorted_index]

                bag[keys[0]] = [values[0][0], values[0, 1]]
                bag[keys[-1]] = [values[-1][0]+values[-1][2], values[-1][1]+values[-1][3]]

                if values[1][0] < values[-1][0] and values[1][1] > values[0][1]:
                    bag[keys[1]] = [values[1][0], values[1][1]+values[1][3]]
                    bag[keys[2]] = [values[2][0]+values[2][1], values[2][1]]

                elif values[1][0] > values[0][0] and values[1][1] < values[-1][1]:
                    bag[keys[1]] = [values[1][0]+values[1][2], values[1][1]]
                    bag[keys[2]] = [values[2][0], values[2][1]+values[2][3]]

                new_img = affine(input_image, (bag[self.classes[3]][:2], bag[self.classes[0]][:2], bag[self.classes[1]][:2]))
                return new_img
                
        return None


class Face_Card(Yolo5):
    def __init__(self, weight_path, classes=None):
        super().__init__(weight_path, classes)
        self.classes = ['face']