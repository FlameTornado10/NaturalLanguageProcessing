from cProfile import label
import os
import math
import matplotlib
import matplotlib.pyplot as plt
from numpy import average


def calculate_item(array_ref,array_res,n):
    ref = []
    res = []
    for index,_ in enumerate(array_ref):
        if index <= len(array_ref) - n:
            temp = []
            for i in range(n):
                temp.append(array_ref[index+i])
            ref.append(temp)

    for index,_ in enumerate(array_res):
        if index <= len(array_res) - n:
            temp = []
            for i in range(n):
                temp.append(array_res[index+i])
            res.append(temp)

    find = 0
    for item in res:
        if item in ref:
            find+=1

    result = find/len(res)
    if result == 0:
        return 1/len(res)
    else:
        return result

def calculate_bleu(reference,result,N):
    reference_array = reference.split()
    result_array = result.split()
    if len(result_array) > len(reference_array):
        bp = 1
    else:
        bp = math.exp(1-len(reference_array)/len(result_array))
    
    ssum = 0
    for i in range(N):
        ssum += math.log(calculate_item(reference_array,result_array,i+1))

    return bp * math.exp(1/N * ssum)


google = []
baidu = []
sougou = []

def main():
    with open('reference.txt','r',encoding='utf8') as f_refer:
        with open('google_rst.txt','r',encoding='utf8') as f_google:
            refer = f_refer.readline()
            google_rst = f_google.readline()
            print(refer)
            while refer:
                google.append(calculate_bleu(refer, google_rst, 4))
                refer = f_refer.readline()
                google_rst = f_google.readline()
        f_refer.close()
        f_google.close()
    with open('reference.txt', 'r', encoding='utf8') as f_refer:
        with open('baidu_rst.txt','r',encoding='utf8') as f_baidu:
            refer = f_refer.readline()
            baidu_rst = f_baidu.readline()
            print(refer)
            while refer:
                baidu.append(calculate_bleu(refer, baidu_rst, 4))
                refer = f_refer.readline()
                baidu_rst = f_baidu.readline()
        f_refer.close()
        f_baidu.close()
    with open('reference.txt', 'r', encoding='utf8') as f_refer:
        with open('sougou_rst.txt', 'r', encoding='utf8') as f_sougou:
            refer = f_refer.readline()
            sougou_rst = f_sougou.readline()
            print(refer)
            while refer:
                sougou.append(calculate_bleu(refer, sougou_rst, 4))
                refer = f_refer.readline()
                sougou_rst = f_sougou.readline()
        f_refer.close()
        f_sougou.close()

    ave_google = average(google)
    ave_baidu = average(baidu)
    ave_sougou = average(sougou)
    print("GOOGLE AVERAGE BLEU:{}".format(ave_google))
    print("BAIDU  AVERAGE BLEU:{}".format(ave_baidu))
    print("SOUGOU AVERAGE BLEU:{}".format(ave_sougou))

    x = []
    for i in range(1,31):
        x.append(i)
    plt.figure()
    plt.plot(google,'r',label='google')
    plt.plot(baidu,'g',label='baidu')
    plt.plot(sougou,'b',label='sougou')
    plt.xlabel("sentence")
    plt.ylabel("BLEU")
    plt.legend()
    plt.show()

    
    


if __name__ == '__main__':
    main()