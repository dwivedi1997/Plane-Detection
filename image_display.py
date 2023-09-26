import os
import cv2
import matplotlib.pyplot as plt

class image_visualization():

    def __init__(self, base_dir,image_name,bboxes,img_width,img_height,labels):

        self.base_dir=base_dir
        self.image_name=image_name
        self.bboxes=bboxes
        self.img_width=img_width
        self.img_height=img_height
        self.labels=labels

    def visualize(self):

        self.image=cv2.imread(os.path.join(self.base_dir,self.image_name))
        self.image=cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB)
        i=0

        for box in self.bboxes:
            cid, x_center, y_center, box_width, box_height = box
            x_center   *= self.img_width
            y_center   *= self.img_height
            box_width  *= self.img_width
            box_height *= self.img_height

            xmin = int(x_center-(box_width/2))
            xmax = int(xmin + box_width)
            ymin = int(y_center-(box_height/2))
            ymax = int(ymin + box_height)
            cv2.rectangle(self.image, (xmin,ymin), (xmax,ymax), (0, 255,0), thickness = 2)

            ((label_width, label_height), _) = cv2.getTextSize(
                str(self.labels[i]),
                fontFace = cv2.FONT_HERSHEY_PLAIN,
                fontScale = 1.75,
                thickness = 2
            )

            cv2.rectangle(
                self.image,
                (int(xmin),int(ymin)),
                (
                    int(xmin + label_width + label_width *0.05),
                    int(ymin + label_height + label_height * 0.25)
                ),
                color = (0, 255, 0),
                thickness = cv2.FILLED
            )

            cv2.putText(
                self.image,
                str(self.labels[i]),
                org = (int(xmin), int(ymin + label_height + label_height * 0.25)),  #bottom left point of the text
                fontFace = cv2.FONT_HERSHEY_PLAIN,
                fontScale = 1.75,
                color = (255, 255, 255),
                thickness = 2
            )

        i = i + 1
        plt.figure(figsize=(10,10))
        plt.imshow(self.image)
        plt.axis('off')