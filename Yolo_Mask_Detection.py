import cv2
import numpy as np


def set_network():
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    return layerOutputs

#Read Network
net = cv2.dnn.readNet('resources/yolov3_training_3000.weights', 'resources/yolov3_testing.cfg')

#Check The Classes
classes = []
with open("resources/classes.txt", "r") as f:
    classes = f.read().splitlines()

#Capture the Video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 50)
fps = int(cap.get(5))
print("fps:", fps)

#Set Rondomly Color
colors = np.random.uniform(0, 255, size=(100, 3))

#Start the Video
while cap.isOpened():
    #Read the Frames
    ret, frame = cap.read()
    



    #Resize The Shape 
    frame = cv2.resize(frame,(800,600))
    #Check The Frame Shape
    height, width, channal = frame.shape
    
    if ret == True:
        
        layerOutputs = set_network()

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:

            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.2:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        if len(indexes)>0:
            
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i],2))
                color = colors[i]
                cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(frame, label + " " + confidence, (x, y+20), font, 1, (255,255,255), 2)

        cv2.imshow('Face Mask Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
             break
    else:
        break

#end the video 
cap.release()
cv2.destroyAllWindows() 
