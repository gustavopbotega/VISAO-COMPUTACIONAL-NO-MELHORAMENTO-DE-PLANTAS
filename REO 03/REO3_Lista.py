import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

os.chdir('C:\\Users\\Gustavo\\Pictures')


'''
a) Aplique o filtro de média com cinco diferentes tamanhos de kernel e
compare os resultados com a imagem original;
'''

img_bgr = cv2.imread('17.jpg')
img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)

blur_5 = cv2.blur(img_rgb,(5,5))
blur_20 = cv2.blur(img_rgb,(20,20))
blur_35 = cv2.blur(img_rgb,(35,35))
blur_51 = cv2.blur(img_rgb,(51,51))
blur_75 = cv2.blur(img_rgb,(75,75))

imagens = [img_rgb,blur_5, blur_20, blur_35, blur_51, blur_75]
titulos = ['Original','5x5','20x20', '35x35', '51x51', '75x75']

for i in range(6):
    plt.subplot(3,2,i+1);plt.imshow(imagens[i])
    plt.xticks([]);plt.yticks([])
    plt.title(titulos[i])
plt.show()

'''
b) Aplique diferentes tipos de filtros com pelo menos dois tamanhos de kernel 
e compare os resultados entre si e com a imagem original
'''
median_20 = cv2.medianBlur(img_rgb,21,)
median_51 = cv2.medianBlur(img_rgb,51)
gaussian_20 = cv2.GaussianBlur(img_rgb,(21,21),0)
gaussian_51 = cv2.GaussianBlur(img_rgb,(51,51),0)

imagens = [img_rgb,median_20, median_51,gaussian_20, gaussian_51]
titulos = ['Original','Median 21x21','Median 51x51','Gaussian 21x21', 'Gaussian 51x51']

for i in range(5):
    plt.subplot(3,2,i+1);plt.imshow(imagens[i])
    plt.xticks([]);plt.yticks([])
    plt.title(titulos[i])
plt.show()

'''
c) Realize a segmentação da imagem utilizando o processo de limiarização. 
Utilizando o reconhecimento de contornos, identifique e salve os objetos de
interesse. Além disso, acesse as bibliotecas Opencv e Scikit-Image, verifique 
as variáveis que podem ser mensuradas e extraia as informações pertinentes
(crie e salve uma tabela com estes dados). Apresente todas as imagens 
obtidas ao longo deste processo.
'''
import pandas as pd
import my_functions as mf

hsv = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2HSV)
s = hsv[:,:,1]
s = cv2.medianBlur(s,35)

l,thresh = cv2.threshold(s,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
folhas_s_fundo = cv2.bitwise_and(img_rgb,img_rgb,mask=thresh)
plt.imshow(thresh, cmap='gray')
####OBTENDO CADA FOLHA EM SEPARADO###
mask = np.zeros(img_rgb.shape,dtype = np.uint8)
cnts = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

####Obtendo as lesoes
folhas_hsv = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2YCR_CB)
h,se,v = cv2.split(folhas_hsv)
__,thrsh1 = cv2.threshold(se,135,255,cv2.THRESH_BINARY)

plt.imshow(thrsh1, cmap='gray')

dimen = []
for (i,c) in enumerate(cnts):
    (x,y,w,h) = cv2.boundingRect(c)
    obj = folhas_s_fundo[y:y+h,x:x+w]
    obj_bgr = cv2.cvtColor(obj,cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'folha{i+1}.png',obj_bgr)
    area = cv2.contourArea(c)
    area_count = cv2.countNonZero(obj[:,:,1])
    razao = (h/w).__round__(2)
    ####Lesoes
    les = thrsh1[y:y+h,x:x+w]
    cv2.imwrite(f'Lesao_folha{i + 1}.png',les)
    area_lesao = cv2.countNonZero(les)
    contorno = cv2.findContours(les,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contorno = contorno[0] if len(contorno) == 2 else contorno[1]
    contorno = len(contorno)
    razao_lesao = ((area_lesao/area_count)*100).__round__(2)
    dimen += [[str(i + 1), str(h), str(w), str(area), str(razao),
               str(area_lesao),str(contorno),str(razao_lesao)]]
    cv2.namedWindow('Folha',cv2.WINDOW_NORMAL)
    cv2.imshow('Folha',les)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

dados_folhas = pd.DataFrame(dimen)
dados_folhas = dados_folhas.rename(columns={0:'FOLHA', 1: 'ALTURA_FOLHA',2:'LARGURA_FOLHA',3:'AREA_FOLHA',4:'RAZAO_FOLHA',
                                            5:'AREA_LESÃO',6:'NUMERO DE PUSTULAS',7:'RAZAO DA LESÃO'})
dados_folhas.to_csv('medidas.csv',index=False)

'''
d) Utilizando máscaras, apresente o histograma somente dos objetos de interesse.
'''
red = cv2.calcHist([img_rgb],[0],thresh,[256],[0,255])
green = cv2.calcHist([img_rgb],[1],thresh,[256],[0,255])
blue = cv2.calcHist([img_rgb],[2],thresh,[256],[0,255])

plt.subplot(3,1,1);plt.plot(red,color ='r')
plt.xticks([])
plt.title('Histograma em vermelho')

plt.subplot(3,1,2);plt.plot(green,color ='g')
plt.xticks([])
plt.title('Histograma em verde')

plt.subplot(3,1,3);plt.plot(blue,color ='b')
plt.title('Histograma em azul')

plt.show()

'''
e) Realize a segmentação da imagem utilizando a técnica de k-means. 
Apresente as imagens obtidas neste processo.
'''
img_rgb = cv2.medianBlur(img_rgb,35)

pixels = img_rgb.reshape((-1,3))
valores=np.float32(pixels)

criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,0.2)
k = 2

dist,labels,(centers) = cv2.kmeans(valores,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
val,contagens = np.unique(labels,return_counts=True)
centers = np.uint8(centers)
matriz_segmentada = centers[labels]
matriz_segmentada = matriz_segmentada.reshape(img_rgb.shape)

plt.imshow(matriz_segmentada)

'''
f) Realize a segmentação da imagem utilizando a técnica de watershed.
Apresente as imagens obtidas neste processo
'''
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import imutils

img_rgb = cv2.medianBlur(img_rgb,35)
gray = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)
thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
plt.imshow(thresh)
plt.show()

shifted = cv2.pyrMeanShiftFiltering(img_rgb, 21, 51)
plt.imshow(shifted)
plt.show()

D = ndimage.distance_transform_edt(thresh)
plt.imshow(D)
plt.show()


localMax = peak_local_max(D, indices=False, min_distance=200,labels=thresh)
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
plt.imshow(labels)
plt.show()

# loop over the unique labels returned by the Watershed
# algorithm
for label in np.unique(labels):
# otherwise, allocate memory for the label region and draw
	# it on the mask
	mask = np.zeros(gray.shape, dtype="uint8")
	mask[labels == label] = 255
	# detect contours in the mask and grab the largest one
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)
	# draw a circle enclosing the object
	((x, y), r) = cv2.minEnclosingCircle(c)
    img = img_rgb.copy()
    cv2.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 2)
	cv2.putText(img, "{}".format(label), (int(x) - 10, int(y)),
	cv2.FONT_HERSHEY_DUPLEX, 5, (0, 0, 255), 2)
# show the output image
cv2.namedWindow("Output",cv2.WINDOW_NORMAL)
cv2.imshow("Output", img)
cv2.waitKey(0)