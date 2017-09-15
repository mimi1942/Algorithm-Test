#Environment
OS : Ubuntu 16.04 LTS
Python : 2.7.3
Opencv : 3.3.0
+ OpenCV Contirb 
+ PIL library
(If you want to install it, Plz write down below code in your terminal
 "sudo pip install pillow")
*Need Laptop with Webcam 



#RUN instructions
1. make a face dataset
$ python dataset_generator.py

and enter your id(1 or 2)


2. train trainner
$ python trainner.py


3. run your face_recognizer
$ python main.py

if you want to change the Text(name) assign by id, 
you have to modify main.py



#Reference site:
http://thecodacus.com/opencv-python-face-detection/#.WWx4tCeRXCI

