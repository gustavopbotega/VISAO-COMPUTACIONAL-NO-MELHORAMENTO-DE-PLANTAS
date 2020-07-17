########################################################################################################################
# DATA: 02/07/2020
# DISCIPLINA: VISÃO COMPUTACIONAL NO MELHORAMENTO DE PLANTAS
# PROFESSOR: VINÍCIUS QUINTÃO CARNEIRO
# E-MAIL: vinicius.carneiro@ufla.br
# GITHUB: vqcarneiro
# ALUNO: Gustavo Pucci
########################################################################################################################
import numpy as np

# REO 01 - LISTA DE EXERCÍCIOS

# EXERCÍCIO 01:
# a) Declare os valores 43.5,150.30,17,28,35,79,20,99.07,15 como um array numpy.
val = np.array([43.5,150.30,17,28,35,79,20,99.07,15])

# b) Obtenha as informações de dimensão, média, máximo, mínimo e variância deste vetor.
dimen = len(val) # ou np.shape(val)[0]
media = np.mean(val).round(2)
maximo = np.max(val)
minimo = np.min(val)
variancia = np.var(val).round(2)

print(f"nesse conjunto de dados de dimensão {dimen}, temos uma media de {media},"
      f"\n variancia de {variancia}, seu maior numero é {maximo} e o menor {minimo}")

# c) Obtenha um novo vetor em que cada elemento é dado pelo quadrado da diferença entre cada elemento do vetor declarado
# na letra a e o valor da média deste.
quadrado = (val - media)**2
print(quadrado)

# d) Obtenha um novo vetor que contenha todos os valores superiores a 30.
sup30 = val[val>30]
print(sup30)

# e) Identifique quais as posições do vetor original possuem valores superiores a 30
pos = np.where(val>30)
print(sup30)

# f) Apresente um vetor que contenha os valores da primeira, quinta e última posição.
vet = val[[0,4,-1]]
print(vet)

# g) Crie uma estrutura de repetição usando o for para apresentar cada valor e a sua respectiva posição durante as iterações
p = 1
for i in val:
      print(i)
      print(p)
      p+=1

# h) Crie uma estrutura de repetição usando o for para fazer a soma dos quadrados de cada valor do vetor.
quad=0

for i in val:
      quad += i**2

print(quad)

# i) Crie uma estrutura de repetição usando o while para apresentar todos os valores do vetor
i=0
while i != len(val):
      print(val[i])
      i += 1

# j) Crie um sequência de valores com mesmo tamanho do vetor original e que inicie em 1 e o passo seja também 1.
val1 = np.array(range(1,10,1))
print(val1)

# h) Concatene o vetor da letra a com o vetor da letra j.
val3 = np.append(val,val1)
print(val3)

########################################################################################################################
########################################################################################################################
########################################################################################################################

# Exercício 02
#a) Declare a matriz abaixo com a biblioteca numpy.
# 1 3 22
# 2 8 18
# 3 4 22
# 4 1 23
# 5 2 52
# 6 2 18
# 7 2 25

matrix = np.array([ [1, 3, 22],
                  [2, 8, 18],
                  [3,4,22],
                  [4,1,23],
                  [5,2,52],
                  [6,2,18],
                  [7,2,25]])

# b) Obtenha o número de linhas e de colunas desta matriz
linha,coluna = np.shape(matrix)

# c) Obtenha as médias das colunas 2 e 3.
mediacol2 = np.mean(matrix[:,1])
mediacol3 = np.mean(matrix[:,2])
print("A media da coluna dois é {} e a media da coluna três é {}".format(mediacol2.round(2),mediacol3.round(2)))

# d) Obtenha as médias das linhas considerando somente as colunas 2 e 3
medialin2 = np.mean(matrix[1,:])
medialin3 = np.mean(matrix[2,:])
print("A media da linha dois é {} e a media da linha três é {}".format(medialin2.round(2),medialin3.round(2)))

# e) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma doença e
# e a terceira peso de 100 grãos. Obtenha os genótipos que possuem nota de severidade inferior a 5.
gendoe = matrix[matrix[:,1]<5]
print(gendoe)

# f) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma doença e
# e a terceira peso de 100 grãos. Obtenha os genótipos que possuem nota de peso de 100 grãos superior ou igual a 22.
genpro = matrix[matrix[:,2]>=22]
print(genpro)

# g) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma doença e
# e a terceira peso de 100 grãos. Obtenha os genótipos que possuem nota de severidade igual ou inferior a 3 e peso de 100
# grãos igual ou superior a 22.
mask = matrix[:,2]>=22
mask1 = matrix[:,1]<=3
genprodo = matrix[mask & mask1]
print(genprodo)


# h) Crie uma estrutura de repetição com uso do for (loop) para apresentar na tela cada uma das posições da matriz e o seu
#  respectivo valor. Utilize um iterador para mostrar ao usuário quantas vezes está sendo repetido.
#  Apresente a seguinte mensagem a cada iteração "Na linha X e na coluna Y ocorre o valor: Z".
#  Nesta estrutura crie uma lista que armazene os genótipos com peso de 100 grãos igual ou superior a 25
it  = 0
for i in np.arange(0,linha,1):
      for j in np.arange(0,coluna,1):
            it +=1
            print(f'interação {it}')
            print(f'Na linha {i+1} \n e coluna {j+1} \n Ocorre o valor {matrix[i,j]}')
            print('-'*20)
            matrix25 = matrix[matrix[:,2] >= 25].tolist()

print(f'Os genotipos {[item[0] for item in matrix25]} possuem peso de 100 grãos igual ou superior a 25')
########################################################################################################################
########################################################################################################################
########################################################################################################################

# EXERCÍCIO 03:
# a) Crie uma função em um arquivo externo (outro arquivo .py) para calcular a média e a variância amostral um vetor qualquer, baseada em um loop (for).

from function import med_var_amostral
vetor = np.random.normal(10,10,100)
mean,var = med_var_amostral(vetor)

# b) Simule três arrays com a biblioteca numpy de 10, 100, e 1000 valores e com distribuição normal com média 100 e variância 2500. Pesquise na documentação do numpy por funções de simulação.
size10 = np.random.normal(100,50,10)
size100 = np.random.normal(100,50,100)
size1000 = np.random.normal(100,50,1000)

# c) Utilize a função criada na letra a para obter as médias e variâncias dos vetores simulados na letra b.
med10,  var10   = med_var_amostral(size10)
med100, var100  = med_var_amostral(size100)
med1000,var1000 = med_var_amostral(size1000)


# d) Crie histogramas com a biblioteca matplotlib dos vetores simulados com valores de 10, 100, 1000 e 100000.
size10000 = np.random.normal(100,50,10000)
med10000,var10000 = med_var_amostral(size10000)

tamanhos = list((10,100,1000,10000))
medias = list((med10,med100,med1000,med10000))
variancias = list((var10,var100,var1000,var10000))
hist = np.array([tamanhos,medias,variancias])
hist = np.transpose(hist)

import matplotlib.pyplot as plt

#####MEDIAS####
x = np.arange(len(tamanhos))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, medias, width, label='Medias',align='center')
plt.ylim([90,110])
plt.plot([-0.5,3.5], [100,100],color = 'r')
ax.set_ylabel('Médias')
ax.set_title(' Vetores simulados com valores de \n 10, 100, 1000 e 100000 amostras')
ax.set_xticks(x)
ax.set_xticklabels(tamanhos)
ax.legend()
plt.show()

#######Variancias##########
fig1, ax1 = plt.subplots()
rects2 = ax1.bar(x - width/2, variancias, width, label='Variancias')
plt.ylim([(np.min(variancias) - 200 ),(np.max(variancias) + 200 )])
plt.plot([-0.5,3.5], [2500,2500],color = 'r')
ax1.set_ylabel('Variancias')
ax1.set_title(' Vetores simulados com valores de \n 10, 100, 1000 e 100000 amostras')
ax1.set_xticks(x)
ax1.set_xticklabels(tamanhos)
ax1.legend()
plt.show()
########################################################################################################################
########################################################################################################################
########################################################################################################################

# EXERCÍCIO 04:
# a) O arquivo dados.txt contem a avaliação de genótipos (primeira coluna) em repetições (segunda coluna) quanto a quatro
# variáveis (terceira coluna em diante). Portanto, carregue o arquivo dados.txt com a biblioteca numpy, apresente os dados e obtenha as informações
# de dimensão desta matriz.
import os
os.chdir('D:\\My Drive\\Arquivos-notebook\\Python-projects\\Visao_computacional\\lista')

dados = np.genfromtxt('dados.txt')
print(dados)
linha,coluna = np.shape(dados)
print(f'Os dados possuem {linha} linhas e {coluna} colunas ')

# b) Pesquise sobre as funções np.unique e np.where da biblioteca numpy
help(np.unique)
help(np.where)

# c) Obtenha de forma automática os genótipos e quantas repetições foram avaliadas
genotipos = np.unique(dados[:,0]).tolist()
repetição = np.unique(dados[:,1]).tolist()
print(f'Temos {len(genotipos)} genótipos avaliados em {len(repetição)} repetições')


# d) Apresente uma matriz contendo somente as colunas 1, 2 e 4
submatriz = dados[:,[0,1,3]]
print(submatriz)

# e) Obtenha uma matriz que contenha o máximo, o mínimo, a
# média e a variância de cada genótipo para a variavel da coluna 4.
# Salve esta matriz em bloco de notas.
np.set_printoptions(suppress=True) ###Não deixa numpy aprsentar em notação cientifica
resul = np.zeros((len(genotipos),5))
it =0
for i in range(0, len(genotipos)):
    it += 1
    resul[i,0] = int(it)
    vit = submatriz[submatriz[:, 0] == it]
    resul[i,1] = np.max(vit[:,2]).round(2)
    resul[i,2] = np.min(vit[:,2]).round(2)
    resul[i,3] = np.mean(vit[:,2]).round(2)
    resul[i,4] = np.var(vit[:,2]).round(2)
print(resul)

np.savetxt('Matrix_genotipos.txt',resul,fmt='%2.2f',delimiter ='\t')

# f) Obtenha os genótipos que possuem média (médias das repetições)
# igual ou superior a 500 da matriz gerada na letra anterior.
maior500 = resul[resul[:,3] > 500]

print(f'Os genótipos que possuem media superior a 500 são {maior500[:,0]}')

# g) Apresente os seguintes graficos:
#    - Médias dos genótipos para cada variável. Utilizar o comando plt.subplot para mostrar mais de um grafico por figura
#    - Disperão 2D da médias dos genótipos (Utilizar as três primeiras variáveis). No eixo X uma variável e no eixo Y outra.
import matplotlib.pyplot as plt

resultados = np.zeros((len(genotipos),6))
it = 0
for i in range(0, len(genotipos)):
    it += 1
    resultados[i,0] = it
    vit1 = dados[dados[:, 0] == it]
    resultados[i,1] = np.mean(vit1[:,2]).round(2)
    resultados[i,2] = np.mean(vit1[:,3]).round(2)
    resultados[i,3] = np.mean(vit1[:,4]).round(2)
    resultados[i,4] = np.mean(vit1[:,5]).round(2)
    resultados[i,5] = np.mean(vit1[:,6]).round(2)

######Plotagem
color = ['r','g','blue','yellow','black']
titles = ['Variavel 1', 'Variavel 2', 'Variavel 3', 'Variavel 4' , 'Variavel 5']
for i in range(5):
    plt.subplot(3, 2, i + 1), plt.bar(resultados[:,0],resultados[:,i],color =color [i])
    plt.title(titles[i])
    plt.ylim((np.min(resultados[:,i]) - np.std(resultados[:,i])**0.5,
              (np.max(resultados[:,i]) + np.std(resultados[:,i])**0.5)))
    plt.xticks(resultados[:,0]), plt.ylabel('Medias')
plt.show()

####Plotagem 2
plt.clf()
colors = ["red", "green", "blue","orange",'black'
          ,'purple','c','m','y','pink']

medias3 = resultados[:,:4]


plt.subplot(2, 2, 1),
for w in range(10):
    plt.scatter(medias3[w,1],medias3[w,2], c = colors[w],label=medias3[w,0],alpha =0.3)
    plt.annotate(str(w+1),[medias3[w,1],medias3[w,2]])
plt.xlabel(f'Variable 1')
plt.ylabel(f'Variable 2')

plt.subplot(2, 2, 2),
for w in range(10):
    plt.scatter(medias3[w,1],medias3[w,3], c = colors[w],label=medias3[w,0],alpha =0.3)
    plt.annotate(str(w+1),[medias3[w,1],medias3[w,3]])
plt.xlabel(f'Variable 1')
plt.ylabel(f'Variable 3')

plt.subplot(2, 2, 3),
for w in range(10):
    plt.scatter(medias3[w,2],medias3[w,3], c = colors[w],label=medias3[w,0],alpha =0.3)
    plt.annotate(str(w+1),[medias3[w,2],medias3[w,3]])
plt.xlabel(f'Variable 2')
plt.ylabel(f'Variable 3')
plt.show()

########################################################################################################################
########################################################################################################################
#######################################################################################################################
