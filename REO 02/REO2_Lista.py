import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import my_functions as mf


##Carregar imagem
os.chdir('D:\\My Drive\\Arquivos-notebook\\Python-projects\\Visao_computacional\\REO 02')
img_bgr = cv2.imread('17.jpg')
img_gray = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB);r,g,b = cv2.split(img_rgb)
img_hsv = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV);h,s,v = cv2.split(img_hsv)
img_yCrCb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2YCrCb);Y,Cr,Cb = cv2.split(img_yCrCb)
img_Lab = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2Lab);L,a,bi = cv2.split(img_Lab)
img_branco = np.zeros((img_rgb.shape[0],img_rgb.shape[1],img_rgb.shape[2]),dtype='uint8');img_branco[:,:,:] = 255

#a) Apresente a imagem e as informações de número de linhas e colunas;
# número de canais e número total de pixels
linha,coluna,canais = img_bgr.shape

plt.figure('Informações da imagem')
plt.imshow(img_rgb)
plt.xticks([]);plt.yticks([])
plt.title('Imagem em RGB')
plt.xlabel('Numero de Linhas: {}\nNumero de Colunas: {}\nCanais: {}'.format(linha,coluna,canais),fontweight='bold')
plt.show()
print('Temos na imagem {} linhas {} colunas e {} canais.'.format(linha,coluna,canais))

#b) Faça um recorte da imagem para obter somente a área de interesse. Utilize esta imagem para
# a solução das próximas alternativas
img_rec = img_rgb[:2400,2600:3900]

plt.imshow(img_rec)
plt.xticks([]);plt.yticks([])
plt.title('Imagem Recortada')
plt.show()

#c) Converta a imagem colorida para uma de escala de cinza (intensidade) e a apresente utilizando
# os mapas de cores “Escala de Cinza” e “JET”;
img_rec_gray = cv2.cvtColor(img_rec,cv2.COLOR_RGB2GRAY)

plt.subplot(1,2,1);plt.imshow(img_rec_gray, cmap='gray')
plt.xticks([]);plt.yticks([])
plt.title('Imagem em escala de cinza gray')

plt.subplot(1,2,2);plt.imshow(img_rec_gray, cmap='jet')
plt.xticks([]);plt.yticks([])
plt.title('Imagem em escala de cinza jet')
plt.show()

#d) Apresente a imagem em escala de cinza e o seu respectivo histograma; Relacione o histograma
# e a imagem.
hist_gray = cv2.calcHist([img_rec_gray],[0],None,[256],[0,255])

plt.subplot(2,1,2)
plt.plot(hist_gray)
plt.title('Histograma escala de cinza')

plt.subplot(2,1,1)
plt.imshow(img_rec_gray,cmap='jet')
plt.title('Imagem em escala de cinza')
plt.show()

#e) Utilizando a imagem em escala de cinza (intensidade) realize a segmentação da imagem de
# modo a remover o fundo da imagem utilizando um limiar manual e o limiar obtido pela técnica
# de Otsu. Nesta questão apresente o histograma com marcação dos limiares utilizados, a
# imagem limiarizada (binarizada) e a imagem colorida final obtida da segmentação. Explique
# os resultados.
ret,trh_otsu = cv2.threshold(img_rec_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
ret1,trh_manual = cv2.threshold(img_rec_gray,150,255,cv2.THRESH_BINARY_INV)

img_s_manual = cv2.bitwise_and(img_rec,img_rec,mask=trh_manual)
img_s_otsu = cv2.bitwise_and(img_rec,img_rec,mask=trh_otsu)

plt.subplot(3,2,1);plt.imshow(img_rec)
plt.xticks([]);plt.yticks([])
plt.title('Imagem RGB')

plt.subplot(3,2,2),plt.imshow(img_rec_gray)
plt.xticks([]);plt.yticks([])
plt.title('Imagem em escala de cinza')

plt.subplot(3,2,3),plt.imshow(img_s_manual)
plt.xticks([]);plt.yticks([])
plt.title('Imagem Manual')

plt.subplot(3,2,4),plt.imshow(img_s_otsu)
plt.xticks([]);plt.yticks([])
plt.title('Imagem OTSU')

plt.subplot(3,2,5),plt.plot(hist_gray)
plt.axvline(x=ret, color = "g")
plt.xticks([])
plt.title('Histograma Manual')

plt.subplot(3,2,6),plt.plot(hist_gray)
plt.axvline(x=ret1, color = "b")
plt.xticks([])
plt.title('Histograma OTSU')

plt.show()
#f) Apresente uma figura contento a imagem selecionada nos sistemas RGB, Lab, HSV e YCrCb.

imagens = [img_rgb,img_hsv,img_Lab,img_yCrCb]
titulos = ['Imagem em RGB',"Imagem em HSV",'Imagem em Lab','Imagem em YCrCb']

for i in range(4):
    plt.subplot(2,2,i+1);plt.imshow(imagens[i])
    plt.xticks([]);plt.yticks([])
    plt.title(titulos[i])
plt.show()

#g) Apresente uma figura para cada um dos sistemas de cores (RGB, HSV, Lab e YCrCb)
# contendo a imagem de cada um dos canais e seus respectivos histogramas.

####RGB
hist_r = cv2.calcHist([r],[0],None,[255],[0,255])
hist_b = cv2.calcHist([b],[0],None,[255],[0,255])
hist_g = cv2.calcHist([g],[0],None,[255],[0,255])

plt.subplot(3,3,2);plt.imshow(img_rgb)
plt.xticks([]);plt.yticks([])
plt.title('Imagem RGB')

plt.subplot(3,3,4);plt.imshow(r)
plt.xticks([]);plt.yticks([])
plt.title('Red')

plt.subplot(3,3,5);plt.imshow(g)
plt.xticks([]);plt.yticks([])
plt.title('Green')

plt.subplot(3,3,6);plt.imshow(b)
plt.xticks([]);plt.yticks([])
plt.title('Blue')

plt.subplot(3,3,7);plt.plot(hist_r,color = 'r')
plt.yticks([])
plt.title('Histograma Red')

plt.subplot(3,3,8);plt.plot(hist_g,color = 'g')
plt.yticks([])
plt.title('Histograma Green')

plt.subplot(3,3,9);plt.plot(hist_b,color = 'b')
plt.yticks([])
plt.title('Histograma Blue')

plt.show()

####HSV
hist_h = cv2.calcHist([h],[0],None,[255],[0,255])
hist_s = cv2.calcHist([s],[0],None,[255],[0,255])
hist_v = cv2.calcHist([v],[0],None,[255],[0,255])

imgs = [img_branco,img_hsv,img_branco,h,s,v,hist_h,hist_s,hist_v]
titulos = ['',"Imagem HSV",'',"Hue","Saturation","Value","Histograma Hue",
           'Histograma Saturation','Histograma Value']


for i in range(9):
    plt.subplot(3,3,i+1)
    if titulos[i][0:5] == 'Histo':
        plt.plot(imgs[i])
        plt.yticks([])
        plt.title(titulos[i])
    else:
        plt.imshow(imgs[i])
        plt.xticks([]);plt.yticks([])
        plt.title(titulos[i])
plt.show()



####Lab
hist_l = cv2.calcHist([L],[0],None,[255],[0,255])
hist_a = cv2.calcHist([a],[0],None,[255],[0,255])
hist_bi = cv2.calcHist([bi],[0],None,[255],[0,255])

imgs = [img_branco,img_Lab,img_branco,L,a,bi,hist_l,hist_a,hist_bi]
titulos = ['',"Imagem Lab",'',"L","a","b","Histograma L",
           'Histograma a','Histograma b']


for i in range(9):
    plt.subplot(3,3,i+1)
    if titulos[i][0:5] == 'Histo':
        plt.plot(imgs[i])
        plt.yticks([])
        plt.title(titulos[i])
    else:
        plt.imshow(imgs[i])
        plt.xticks([]);plt.yticks([])
        plt.title(titulos[i])
plt.show()

#####YCrCb

hist_Y = cv2.calcHist([Y],[0],None,[255],[0,255])
hist_Cr = cv2.calcHist([Cr],[0],None,[255],[0,255])
hist_Cb = cv2.calcHist([Cb],[0],None,[255],[0,255])

imgs = [img_branco,img_yCrCb,img_branco,Y,Cr,Cb,hist_Y,hist_Cr,hist_Cb]
titulos = ['',"Imagem YCrCb",'',"Y","Cr","Cb","Histograma Y",
           'Histograma Cr','Histograma Cb']


for i in range(9):
    plt.subplot(3,3,i+1)
    if titulos[i][0:5] == 'Histo':
        plt.plot(imgs[i])
        plt.yticks([])
        plt.title(titulos[i])
    else:
        plt.imshow(imgs[i])
        plt.xticks([]);plt.yticks([])
        plt.title(titulos[i])
plt.show()

#h) Encontre o sistema de cor e o respectivo canal que propicie melhor segmentação da imagem
#de modo a remover o fundo da imagem utilizando limiar manual e limiar obtido pela técnica
#de Otsu. Nesta questão apresente o histograma com marcação dos limiares utilizados, a
#imagem limiarizada (binarizada) e a imagem colorida final obtida da segmentação. Explique
#os resultados e sua escolha pelo sistema de cor e canal utilizado na segmentação. Nesta
#questão apresente a imagem limiarizada (binarizada) e a imagem colorida final obtida da
#segmentação.

#Escolha do canal Saturation
img_rec_hsv = cv2.cvtColor(img_rec,cv2.COLOR_RGB2HSV)
r_h,r_s,r_v = cv2.split(img_rec_hsv)

rts,thres_s = cv2.threshold(r_s,50,255,cv2.THRESH_BINARY)
rtot,thres_otsu = cv2.threshold(r_s,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

s_bit_manual = cv2.bitwise_and(img_rec ,img_rec, mask=thres_s)
s_bit_otsu = cv2.bitwise_and(img_rec,img_rec,mask=thres_otsu)
plt.imshow(s_bit_otsu)
s_hist = cv2.calcHist([r_s],[0],None,[255],[0,255])

plt.subplot(3,2,1);plt.imshow(thres_s)
plt.xticks([]);plt.yticks([])
plt.title('Threshold manual')

plt.subplot(3,2,2);plt.imshow(thres_otsu)
plt.xticks([]);plt.yticks([])
plt.title('Threshold Otsu')

plt.subplot(3,2,3)
plt.plot(s_hist)
plt.axvline(x= rts)
plt.title('Histograma manual')

plt.subplot(3,2,4)
plt.plot(s_hist)
plt.axvline(x= rtot)
plt.title('Histograma otsu')

plt.subplot(3,2,5);plt.imshow(s_bit_manual)
plt.xticks([]);plt.yticks([])
plt.title('Imagem colorida manual')

plt.subplot(3,2,6);plt.imshow(s_bit_otsu)
plt.xticks([]);plt.yticks([])
plt.title('Imagem colorida Otsu')

plt.show()

#i) Obtenha o histograma de cada um dos canais da imagem em RGB, utilizando como mascara
# a imagem limiarizada (binarizada) da letra h.

mask_hist_r = cv2.calcHist([img_rec],[0],thres_otsu,[256],[0,255])
mask_hist_g = cv2.calcHist([img_rec],[1],thres_otsu,[256],[0,255])
mask_hist_b = cv2.calcHist([img_rec],[2],thres_otsu,[256],[0,255])

plt.subplot(3,1,1);plt.plot(mask_hist_r,color ='r')
plt.xticks([])
plt.title('Histograma em vermelho')

plt.subplot(3,1,2);plt.plot(mask_hist_g,color ='g')
plt.xticks([])
plt.title('Histograma em verde')

plt.subplot(3,1,3);plt.plot(mask_hist_b,color ='b')
plt.title('Histograma em azul')

plt.show()

#j) Realize operações aritméticas na imagem em RGB de modo a realçar os aspectos de seu
#interesse. Exemplo (2*R-0.5*G). Explique a sua escolha pelas operações aritméticas. Segue
#abaixo algumas sugestões.

#s_bit_otsu

arit = ((2*s_bit_otsu[:,:,0]) - (0.5 *s_bit_otsu[:,:,1]))
arit = arit.astype(np.uint8)
plt.imshow(arit,cmap='jet')
hist = cv2.calcHist([arit],[0],None,[256],[1,255])

lim,doença_folhas = cv2.threshold(arit,210,255,cv2.THRESH_BINARY)

plt.imshow(doença_folhas,cmap='jet')
plt.show()

plt.imshow(img_rec)

