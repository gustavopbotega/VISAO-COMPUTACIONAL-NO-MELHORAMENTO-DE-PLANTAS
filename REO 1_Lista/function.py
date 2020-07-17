import os
os.chdir('D:\\My Drive\\Arquivos-notebook\\Python-projects\\maturação')
def med_var_amostral(dado):
    soma=0
    v1=0
    somaquadrada = 0
    for i in dado:
        soma += i
    media = soma/len(dado)
    for j in dado:
        v1 = (j - media)**2
        somaquadrada += v1
    var = somaquadrada/(len(dado)-1)
    print(f'Média: {media.__round__(4)}')
    print(f'Variância amostral: {var.__round__(4)}')
    return  media,var