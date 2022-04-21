import cv2
import streamlit as st
from PIL import Image
import requests
import os


input_img_path = './process_bin/input_img_path.jpg'
frame_path = './process_bin/frame.jpg'
face_path = './process_bin/anchor_face.jpg'
url = 'http://127.0.0.2:8080/'
n = 0

if __name__ == '__main__':

    st.title("KYC")
    file = st.file_uploader(label='')
    if file != None:

        input_img = Image.open(file)
        input_img.save(input_img_path)
        reponse = requests.post(url+'detect_face_card', data={'input_img_path': input_img_path}).json()

        if reponse['face_path'] != None:
            
            flag = False
            run = st.checkbox('Run')
            FRAME_WINDOW = st.image([])
            camera = cv2.VideoCapture(0)

            while run:
                _, frame = camera.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(frame_path, frame)
                reponse = requests.post(url+'matching_face', data={'frame_path': frame_path, 'flag': flag}).json()
                flag = True
                cosine_sim = reponse['cosine_sim']

                if cosine_sim != None:
                    bbox = reponse['bbox']
                    cv2.rectangle(frame, (bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1]), (0,255,0))
                    if cosine_sim >= 0.75:
                        n = n + 1

                    if n >= 30:
                        cv2.putText(frame, 'true', (20, 40), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=2, color=(0, 0, 255), thickness=2)
                    else:
                        cv2.putText(frame, str(round(cosine_sim, 2)), (20, 40), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=2, color=(0, 0, 255), thickness=2)
                FRAME_WINDOW.image(frame)
        else:
            st.write('None anchor face')
    
    os.system('rm ./process_bin/*')