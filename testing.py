import math
import time
import sys
import torch
from torchvision import transforms
sys.path.append("..")
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint
from keras.models import load_model
import numpy as np
import cv2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
model = weigths['model']
_ = model.float().eval()
if torch.cuda.is_available():
    model.half().to(device)
#firemodel
fire_model=load_model('fire_detection.h5')
#accidentmodel
crash_model=load_model('crash_detection.h5')

#image pre-processing
def image_preprocessing(frame):
    resized = cv2.resize(frame, (224, 224))
    normalized = np.array(resized) / 255.0
    preprocessed = np.expand_dims(normalized, axis=0)
    return preprocessed

#accident detection method
def accidentdetection(frame,accident_count):
    img = image_preprocessing(frame)
    pred = fire_model.predict(img)[0][0]
    if pred > 0.5:
        crash_class_label = "Accident"
    else:
        crash_class_label = "No Accident"
    if crash_class_label == "No Accident":
        accident_count = 0
    else:
        accident_count += 1
    if accident_count > 50:
        return [1, accident_count, crash_class_label]
    else:
        return [0, accident_count, crash_class_label]

#fire detection method
def firedetection(frame,fire_count):
    img = image_preprocessing(frame)
    pred = fire_model.predict(img)[0][0]
    if pred>0.5:
        fire_class_label = "Fire"
    else:
        fire_class_label="No_Fire"
    if fire_class_label=="No_Fire":
        fire_count=0
    else:
        fire_count+=1
    if fire_count>50:
        return[1,fire_count,fire_class_label]
    else:
        return [0,fire_count,fire_class_label]

#fall detection method
def fallDetection(frame, fallCount,prev):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = letterbox(image, 250, stride=64, auto=True)[0]
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        # get predictions
        with torch.no_grad():
            output, _ = model(image)
        # Apply non max suppression
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'],
                                         kpt_label=True)
        output = output_to_keypoint(output)
        im0 = image[0].permute(1, 2, 0) * 255
        im0 = im0.cpu().numpy().astype(np.uint8)

        # reshape image format to (BGR)
        im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
        startTime = time.time();
        for idx in range(output.shape[0]):
            left_shoulder_y = output[idx][23]
            left_shoulder_x = output[idx][22]
            right_shoulder_y = output[idx][26]

            left_body_y = output[idx][41]
            left_body_x = output[idx][40]

            len_factor = math.sqrt(((left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2))

            left_foot_y = output[idx][53]

            if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (
                    len_factor / 2) and left_shoulder_y > left_body_y - (len_factor / 2):
                fallCount += 1
                print("Person Fell Down")
                if fallCount > 3:
                    return [1,prev, fallCount, 1]
                return [1,prev,fallCount, 0]
        return [0,prev, fallCount, 0]

#sending mail
def send_email(subject, message, from_email, to_email, password,image):
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))
    fireimg=MIMEImage(image,name='fireimage.jpg')
    msg.attach(fireimg)
    try:
        #connecting to server
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print("Error sending email: ", e)
    finally:
        server.quit()

#mail details
fire_subject = "Fire detected!"
accident_subject="Accident detected !"
fall_subject = "Fall detected!"
fire_message = "Alert! A fire has been detected."
accident_message="Alert ! Acccident has been occured"
fall_message = "Alert! A Fall has been detected."
from_email = "nani47727@gmail.com"
to_email = "saikumarmemories@gmail.com"
password = "ehbpyivgrjqbeumd"

#main driver code
detection_option =int(input("Enter the option (1.fire , 2. accident , 3. fall) : "))

#firedetection
if detection_option==1:
    cap = cv2.VideoCapture("Accident2.mp4")
    fire_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #image converting to jpg
        success, img_encoded = cv2.imencode('.jpg', frame)
        mail_img = img_encoded.tobytes()
        fire_result = firedetection(frame, fire_count)
        fire_count = fire_result[1]
        if fire_result[0]:
            fire_count = 0
            #sending mail
            send_email(fire_subject, fire_message, from_email, to_email, password, mail_img)
            print("fire_detected")
        cv2.putText(frame,
                    "Class : {0}".format(fire_result[2]),
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        img = cv2.resize(frame, (800, 600))
        #display frames
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

#accident detection
elif detection_option==2:
    #video capture
    cap = cv2.VideoCapture("crash.mp4")
    accident_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #image converting to jpg
        success, img_encoded = cv2.imencode('.jpg', frame)
        mail_img = img_encoded.tobytes()
        accident_result = accidentdetection(frame, accident_count)
        accident_count = accident_result[1]
        if accident_result[0]:
            accident_count = 0
            #sending mail
            send_email(accident_subject, accident_message, from_email, to_email, password, mail_img)
            print("Accident detected")
        cv2.putText(frame,
                    "Class : {0}".format(accident_result[2]),
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        img = cv2.resize(frame, (800, 600))
        #display frames
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

#fall detection
else :
    cap = cv2.VideoCapture("fall2.mp4")
    fps = 6
    prev = 0
    fall_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        te = time.time() - prev
        #image converting to jpg
        success, img_encoded = cv2.imencode('.jpg', frame)
        mail_img = img_encoded.tobytes()
        if te > 1.0 / fps:
            prev = time.time()
            data = fallDetection(frame, fall_count, prev)
            fall_count=data[2]
            if (data[3]):
                fall_count = 0
                print("fall detected")
                #sending mail
                send_email(fall_subject, fall_message, from_email, to_email, password, mail_img)
        cv2.putText(frame,
                    "Class : {0}".format("fall" if data[0] else "no fall"),
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        img = cv2.resize(frame, (800, 600))
        #display frames
        cv2.imshow('frame', img)
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if current_frame == total_frames:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()