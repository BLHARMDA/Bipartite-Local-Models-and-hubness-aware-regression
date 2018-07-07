# coding=utf-8
import numpy as np
import scipy as sp
import scipy.sparse
import openpyxl as xlwt
import xlrd
import random
import datetime
import copy
import numba
import sys,os
reload(sys) 
sys.setdefaultencoding('gbk') 

#################################################

# # # # # PATH # # # # #
full_path=os.path.realpath(__file__)
eop=full_path.rfind(__file__)
main_path=full_path[0:eop]
folder_path=full_path[0:eop]+u'DATA'

# # # # # GLOBALPARAMS # # # # #

# Constants
M_num=495                                           #miRNA num
D_num=383                                           #disease num
MD_num=5430                                         #associations num

# md
A=[['0'],['0']]                                     #dict for m-d
m_d=np.zeros(1)                                     #m-d associations mat
m_d_backup=np.zeros(1)                              #backup for m-d

# ss
ss=np.zeros(1)                                      #ss mat
ss_w=np.zeros(1)                                    #ss weighted mat

# fs
fs=np.zeros(1)                                      #fs mat
fs_w=np.zeros(1)                                    #fs weighted mat

# predict result
y_local=np.zeros(1)                                 #global loocv results
y_global=np.zeros(1)                                #local loocv results

######################################################################

# md
test_path_md=folder_path+u'/1.miRNA-disease associations/miRNA-disease_index.xlsx'
# ss
test_path_ss_1=folder_path+u'/2.disease semantic similarity 1/SS1.txt'
test_path_ss_2=folder_path+u'/3.disease semantic similarity 2/SS2.txt'
test_path_ss_w=folder_path+u'/2.disease semantic similarity 1/SSW1.txt'
# fs
test_path_fs=folder_path+u'/4.miRNA functional similarity/FS.txt'
test_path_fs_w=folder_path+u'/4.miRNA functional similarity/FSW.txt'

######################################################################

# # # # # ALGORITHM START # # # # #

def __start__(path=folder_path,nm=M_num,nd=D_num):  #path:root path of the code
    global folder_path,M_num,D_num
    folder_path=path
    M_num=nm
    D_num=nd
    load_md(test_path_md)                           #load all data
    load_ss(test_path_ss_1,test_path_ss_2,test_path_ss_w)
    load_fs(test_path_fs,test_path_fs_w)

# # # # # DATA LOADER # # # # #

def load_md(path_md):
    
    global M_num,D_num,MD_num
    global A,m_d,m_d_backup

    # read xlsx
    md_table=xlrd.open_workbook(path_md)
    md_sheet=md_table.sheet_by_index(0)
    MD_num=md_sheet.nrows
    
    # save m-d associations in numpy
    m_d=np.zeros([M_num,D_num])                     #init for m_d
    for i in range(MD_num):
        row_index=int(md_sheet.cell_value(rowx=i,colx=0))-1
        col_index=int(md_sheet.cell_value(rowx=i,colx=1))-1
        A[0].append(int(row_index))
        A[1].append(int(col_index))
        m_d[row_index,col_index]=1
    m_d_backup=copy.copy(m_d)
    del A[0][0],A[1][0]                             #remove init data for A

def load_ss(path_ss_1,path_ss_2,path_ss_w):

    global ss,ss_w
    
    # read ss into numpy array
    ss_1=np.loadtxt(path_ss_1)
    ss_2=np.loadtxt(path_ss_2)
    ss_w=np.loadtxt(path_ss_w)
    
    ss=(ss_1+ss_2)/2

def load_fs(path_fs,path_fs_w):
    
    global fs,fs_w
    
    # read fs into numpy array
    fs=np.loadtxt(path_fs)
    fs_w=np.loadtxt(path_fs_w)

######################################################################

# # # NUMBA ACCELERATE # # # 

@numba.jit
def jitsum(x):                                      #sum by row
    [m,n]=x.shape
    s=np.zeros(m)
    for i in range(int(m)):
        for j in range(int(n)):
            s[i]+=x[i,j]
    return s
@numba.jit
def jitsumt(x):                                     #sum by col
    [m,n]=x.shape
    s=np.zeros(n)
    for i in range(int(m)):
        for j in range(int(n)):
            s[j]+=x[i,j]
    return s
@numba.jit
def jitsumall(x):                                   #sum all entries
    [m,n]=x.shape
    s=0
    for i in range(int(m)):
        for j in range(int(n)):
            s+=x[i,j]
    return s

# # # # # PREPROCESS # # # # #

@numba.jit
def __init__():
    
    global M_num,D_num
    global ss,ss_w,fs,fs_w,m_d
    
    ''' Guassian Profile similarity into gs_m and gs_d '''
    #cal Gamma
    Gamma_d_s=1
    Gamma_m_s=1
    md_f=jitsumall(m_d*m_d)
    Gamma_d=Gamma_d_s/(md_f/D_num)
    Gamma_m=Gamma_m_s/(md_f/M_num)
    
    #cal KD
    IP_d=np.tile(m_d.T,[1,D_num])-np.resize(m_d.T,[1,M_num*D_num])  #DxMD
    IP_d=np.resize(IP_d*IP_d,[D_num*D_num,M_num])
    gs_d=np.exp(-Gamma_d*np.resize(jitsum(IP_d),[D_num,D_num]))

    #cal KM
    IP_m=np.tile(m_d,[1,M_num])-np.resize(m_d,[1,M_num*D_num])      #MxMD
    IP_m=np.resize(IP_m*IP_m,[M_num*M_num,D_num])
    gs_m=np.exp(-Gamma_m*np.resize(jitsum(IP_m),[M_num,M_num]))
    
    ''' Integration '''
    m_m=fs*fs_w+(1-fs_w)*gs_m                                       #SS for 1 GS for 0
    d_d=ss*ss_w+(1-ss_w)*gs_d                                       #FS for 1 GS for 0
    
    #cal Jaccard similarity
    m_j=np.ones([M_num,M_num])
    d_j=np.ones([D_num,D_num])
    #retrive i and j-th row in m_d
    Mi=np.dot(m_d,m_d.T)                                            #intersect
    Di=np.dot(m_d.T,m_d)
    Mu_t=np.tile(m_d,[1,M_num])+np.resize(m_d,[1,M_num*D_num])      #union
    Mu_t=np.resize(Mu_t,[M_num*M_num,D_num])
    Mu=np.resize(jitsum(Mu_t),[M_num,M_num])-Mi
    Du_t=np.tile(m_d.T,[1,D_num])+np.resize(m_d.T,[1,M_num*D_num])
    Du_t=np.resize(Du_t,[D_num*D_num,M_num])
    Du=np.resize(jitsum(Du_t),[D_num,D_num])-Di
    m_j=Mi/(Mu+(Mu==0))
    d_j=Di/(Du+(Du==0))
    
    #Enhansed repression
    M=np.column_stack((m_m,m_j))
    D=np.column_stack((d_d,d_j))

    return M,D
########################################################################

#rank by euclid distance, 1 for rank less than k
@numba.jit
def kNNmat(rankdata,k): #kNN
    num=int(rankdata.shape[0])
    rank=np.zeros([num,num])
    for i in range(num):
        rki=np.argsort(rankdata[i])
        ranki=np.zeros(num)
        for j in range(num):
            t=rki[j]
            ranki[t]=j
        for j in range(num):
            rank[i,j]=1 if ranki[j]<k else 0
    return rank

@numba.jit
def MatEu(mat): #euclid distance mat
    num=int(mat.shape[0])
    mt=np.tile(mat,[1,num])-np.resize(mat,[1,num*2*num]) #a-b
    mt=np.resize(mt*mt,[num*num,2*num])
    mt=np.resize(jitsum(mt),[num,num])
    return mt

@numba.jit
def BLHARMDA(k,M,D): #k:the k in kNN N:repeated times
    
    global M_num,D_num,m_d

    y=m_d
    yc_w=0
    yc_k=0
    IP=np.tile(jitsumt(m_d)==0,[M_num,1])+np.tile(jitsum(m_d)==0,[D_num,1]).T!=0
    Mx=np.tile(m_d,[2,1])   #2MxD
    Dx=np.tile(m_d,[1,2])   #Mx2D
    
    # cal eu distance
    M_eu=MatEu(M)
    D_eu=MatEu(D)
    # rank
    knn_m=kNNmat(M_eu,k)    #MxM
    knn_d=kNNmat(D_eu,k)    #DxD
    
    ''' ECkNN for known diseases and mirna '''
    yc_m=np.dot(knn_m.T,y)  #RkNN x m_d MxD
    yc_d=np.dot(y,knn_d)    #MxD
    yc_m_2=np.tile(jitsumt(knn_m),[D_num,1]).T
    yc_d_2=np.tile(jitsumt(knn_d),[M_num,1])
    
    #算yc_d和yc_m
    yc_d+=(yc_d_2==0)*y     #y for entries dont have Rknn
    yc_d_2+=(yc_d_2==0)
    yc_d/=yc_d_2            #to 1
    
    yc_m+=(yc_m_2==0)*y     #y for 0 entries
    yc_m_2+=(yc_m_2==0)
    yc_m/=yc_m_2
    
    #prediction result y_d and y_m
    x_m=np.resize(yc_m.T,[1,D_num*M_num])*np.tile(knn_m,[1,D_num])  #MxMD
    x_d=np.resize(yc_d,[1,M_num*D_num])*np.tile(knn_d,[1,M_num])    #DxDM
    sum_m=jitsum(np.resize(x_m,[M_num*D_num,M_num]))                #1xMD
    sum_d=jitsum(np.resize(x_d,[D_num*M_num,D_num]))
    y_m=np.resize(sum_m,[M_num,D_num])
    y_d=np.resize(sum_d,[D_num,M_num]).T
    
    #cal yc_k
    yc_k+=(y_m+y_d)/(2.0*k)
    
    ''' weighted for new diseases and mirna '''
    #cal yc_1,yc_2
    yc_1_b=jitsum(M)                                                #Mx1
    yc_2_b=jitsum(D)                                                #Dx1
    yc_1=np.dot(M,Mx).T/(yc_1_b+(yc_1_b==0))                        #DxM
    yc_2=np.dot(Dx,D.T)/(yc_2_b+(yc_2_b==0))                        #MxD
    
    #added to repeated cal result
    yc_w+=(yc_1.T+yc_2)/2.0

    ''' final output '''
    yc=IP*yc_w+(1-IP)*yc_k
    
    return yc

# # # # # # Case Study # # # # # #

# COMMON PROCESS #
dbDEMC=[]
miR2Disease=[]
task=[]
Mindex_dict={}      #dict for miR
Mindex_digit=[]     #array for miR
Dindex_digit=[]

# dbDEMC
path_dbDEMC=folder_path+u'/dbDEMC.xlsx'
# miR2Disease
path_miR2Disease=folder_path+u'/miR2Disease.xlsx'
# task
path_task=folder_path+u'/stat.xlsx'
# Mindex
path_Mindex=folder_path+u'/1.miRNA-disease associations/miRNA_index.xlsx'
# Dindex
path_Dindex=folder_path+u'/1.miRNA-disease associations/disease_index.xlsx'

def load4cs():

    global MD_num,M_num,D_num,dbDEMC_num,miR2Disease_num,task_num
    global dbDEMC,miR2Disease,Mindex_dict,Mindex_digit,Dindex_dict,Dindex_digit,task
    
    dbDEMC_table=xlrd.open_workbook(path_dbDEMC)
    miR2Disease_table=xlrd.open_workbook(path_miR2Disease)
    task_table=xlrd.open_workbook(path_task)
    Mindex_table=xlrd.open_workbook(path_Mindex)
    Dindex_table=xlrd.open_workbook(path_Dindex)

    dbDEMC_sheet=dbDEMC_table.sheet_by_index(0)
    miR2Disease_sheet=miR2Disease_table.sheet_by_index(0)
    task_sheet=task_table.sheet_by_index(0)
    Mindex_sheet=Mindex_table.sheet_by_index(0)
    Dindex_sheet=Dindex_table.sheet_by_index(0)

    dbDEMC_num=dbDEMC_sheet.nrows
    miR2Disease_num=miR2Disease_sheet.nrows
    task_num=task_sheet.nrows-1

    # read dbDEMC
    for i in range(dbDEMC_num):
        m_data=str(dbDEMC_sheet.cell_value(rowx=i,colx=0))
        d_data=str(dbDEMC_sheet.cell_value(rowx=i,colx=1))
        d_data=d_data.lower().replace(',','')
        dbDEMC.append((m_data,d_data))

    # read miR2Disease
    for i in range(miR2Disease_num):
        m_data=str(miR2Disease_sheet.cell_value(rowx=i,colx=0))
        d_data=str(miR2Disease_sheet.cell_value(rowx=i,colx=1))
        d_data=d_data.lower().replace(',','')
        miR2Disease.append((m_data,d_data))

    # read task
    for i in range(1,task_num+1):
        data=str(task_sheet.cell_value(rowx=i,colx=0))
        task.append(data)

    # read Mindex
    for i in range(M_num):
        data=str(Mindex_sheet.cell_value(rowx=i,colx=1))
        Mindex_digit.append(data)

    # read Dindex
    for i in range(D_num):
        data=str(Dindex_sheet.cell_value(rowx=i,colx=1))
        Dindex_digit.append(data)

# # # Case 1 # # #

def scoresNranks(k=100):

    global M_num,D_num,MD_num
    global m_d
    scores_unknown=[]

    m_d=copy.copy(m_d_backup)
    [M,D]=__init__()
    yc=BLHARMDA(k,M,D)
    
    count=0
    for j in range(M_num):
        for k in range(D_num):
            if m_d[j,k]==0:
                count+=1
                M_name=Mindex_digit[j]
                D_name=Dindex_digit[k]
                scores_unknown.append((D_name,M_name,yc[j,k]))
    scores_unknown=sorted(scores_unknown,key=lambda scores_unknown:scores_unknown[2],reverse=True)

    scores_table=xlwt.Workbook()
    scores_sheet=scores_table.active
    for i in range(count):
        scores_sheet.append(scores_unknown[i])
    save_path=main_path+u'/RESULT/Result.xlsx'    
    scores_table.save(save_path)

    for d in range(D_num):
        scores_i=yc[:,d]
        rank_i=np.argsort(-scores_i)
        output_i=[]
        for j in rank_i:
            output_i.append((Mindex_digit[j],scores_i[j]))
        i_table=xlwt.Workbook()
        i_sheet=i_table.active
        for i in range(M_num):
            i_sheet.append(output_i[i])
        i_path=main_path+u'/RESULT/DISEASES/'+Dindex_digit[d].replace(',','')+'.xlsx'
        i_table.save(i_path)

def case1_test():
    
    global M_num,D_num,MD_num,task_num
    global Mindex_digit,Dindex_digit,dbDEMC,miR2Disease,task
    outcome=[('','10','20','50')]

    #for all diseases
    for d in range(task_num):
        d_name=task[d]
        i_path=main_path+u'/RESULT/DISEASES/'+d_name+'.xlsx'
        i_table=xlrd.open_workbook(i_path)
        i_sheet=i_table.sheet_by_index(0)
        d_check=d_name.lower().replace(',','')
        
        count=0
        count10=0
        count20=0
        test_case=[]
        for i in range(50):
            if i==10:
                count10=count
            if i==20:
                count20=count
            m_name=str(i_sheet.cell_value(rowx=i,colx=0))
            score=str(i_sheet.cell_value(rowx=i,colx=1))
            pair=(m_name,d_check)
            db_str=''
            flag=0
            if pair in dbDEMC:
                db_str+='dbDEMC'
                flag=1
            if pair in miR2Disease:
                if flag==1:
                    db_str+=' and '
                db_str+='miR2Disease'
                flag=1
            count+=flag
            test_case.append((d_name,m_name,score,db_str))

        case_table=xlwt.Workbook()
        case_sheet=case_table.active
        for i in range(50):
            case_sheet.append(test_case[i])
        case_path=main_path+u'/RESULT/EVALUATE/'+d_name+'.xlsx'
        case_table.save(case_path)
        outcome.append((d_name,count10,count20,count))
        
    outcome_table=xlwt.Workbook()
    outcome_sheet=outcome_table.active
    for i in range(task_num):
        outcome_sheet.append(outcome[i])
    outcome_path=main_path+u'/RESULT/STAT.xlsx'
    outcome_table.save(outcome_path)

# # # # # TEST # # # # #

''' RUN '''
def __run__(k):
    __start__()
    load4cs()
    print 'data loaded\n'
    scoresNranks(k)
    case1_test()
    print 'done\n'

''' MAIN '''
#__run__(0)