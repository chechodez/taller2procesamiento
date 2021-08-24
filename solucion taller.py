from main import TF
import cv2
import sys
import os
import numpy as np
path =r"C:\Users\chech\OneDrive\Escritorio"#Definición de donde se encuentra la imagen de la huella
image_name = "01_1.tif"#nombre del archivo
path_file = os.path.join(path, image_name)
image = cv2.imread(path_file)#lectura de imagen
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)# se pasa la imagen a grises
clase=TF(image_gray)#se inicializa la clase y se corre el código del constructor
clase.set_theta(0,5)#se definde theta y delta theta
a=clase.filtering()#proceso de filtrado ángulo 0° delta de 5°
clase.set_theta(45,5)#se definde theta y delta theta
b=clase.filtering()##proceso de filtrado ángulo 45° delta de 5°
clase.set_theta(90,5)#se definde theta y delta theta
c=clase.filtering()##proceso de filtrado ángulo 90° delta de 5°
clase.set_theta(135,5)#se definde theta y delta theta
d=clase.filtering()##proceso de filtrado ángulo 135° delta de 5°
cv2.imshow("Promedio de los 4 filtros", (a+b+c+d)/(4))
cv2.waitKey(0)