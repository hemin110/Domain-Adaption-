# -*- coding: utf-8 -*-
"""
Created on Sat May 27 09:32:29 2016

@author: APAC
"""
import numpy as np
from scipy import sparse as sp


def DAELM(TrainingData_File , TrainingData_File_tardomain , TestingData_File , Elm_Type="CLASSIFIER" , NumberofHiddenNeurons=100 ,ActivationFunction="sig",NL ):
    
    '''
    TrainingData_File = [Ts,Xs], A_train
    TrainingData_File_tardomain=[Tt,Xt] , B_train
    TestingData_File=[TestT,X_te] B_test
    Elm_Type = CLASSIFIER
    NumberofHiddenNeurons = 100 default
    ActivationFunction = sig
    NL = model selection
    you can fineturn the model by modify Cs and Ct
    '''
    Cs = 0.01
    Ct = 0.01
    # select type
    REGRESSION=0
    CLASSIFIER=1
    #load train data
    train_data = TrainingData_File
    T = train_data[:,0].T
    P = train_data[:,1:train_data.shape[1]].T
    del train_data
    #load train data in target domian
    train_target_data = TrainingData_File_tardomain
    Tt = train_target_data[:,0].T
    Pt = train_target_data[:,1:train_target_data.shape[1]].T
    #load test data
    test_data = TestingData_File
    TVT = test_data[:,0].T
    TE0 = test_data[:,0].T
    TVP = test_data[:,2:test_data.shape[1]].T
    del test_data
    
    NumberofTrainingData = P.shape[1]
    NumberofTrainingData_Target = Pt.shape[1]
    NumberofTestingData = TVP.shape[1]
    NumberofInputNeurons = P.shape[0]
    
    if Elm_Type is not "REGRESSION":
        sorted_target = np.sort(np.hstack((T ,  TVT)))
        label = np.zeros((1,1))
        label[0,0] = sorted_target[0,0]
        j = 0
        for i in range(2,(NumberofTrainingData+NumberofTestingData+1)):
            if sorted_target[0,i-1] != label[0,j-1]:
                j=j+1
                label[0,j-1] = sorted_target[0,i-1]
                
        number_class = j+1
        NumberofOutputNeurons = number_class
        # Processing the targets of training
        temp_T = np.zeros(NumberofOutputNeurons , NumberofTrainingData)
        for i in range(1,NumberofTrainingData+1):
            for j in range(1,number_class+1):
                if label(0,j-1) == T(0,i-1):
                    break
            temp_T[j-1 , i-1] = 1
        T = temp_T*2-1
        # Processing the targets of training in target domain
        temp_Tt = np.zeros(NumberofOutputNeurons , NumberofTrainingData_Target)
        for i in range(1,NumberofTrainingData_Target+1):
            for j in range(1 , number_class+1):
                if label[0,j-1] == Tt[0,i-1]:
                    break
            temp_Tt[j-1 , i-1] = 1
        Tt = temp_Tt*2-1
        #Processing the targets of testing
        temp_TV_T = np.zeros(NumberofOutputNeurons,NumberofTestingData)
        for i in range(1,NumberofTestingData):
            for j in range(1,number_class+1):
                if label(0,j-1) == TVT(0,i-1):
                    break
            temp_TV_T[j-1 , i-1] = 1
        TVT = temp_TV_T*2-1
        
    #begin compute
    InputWeight = np.random.rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1
    BiasofHiddenNeurons = np.random.rand(NumberofHiddenNeurons ,1)
    tempH = InputWeight*P
    tempHt = InputWeight*Pt
    del P
    del Pt
    
    ind = np.ones(1,NumberofTrainingData)
    indt = np.ones(1,NumberofTrainingData_Target)
    BiasMatrix = BiasofHiddenNeurons[:,ind-1]
    BiasMatrixT = BiasofHiddenNeurons[:,indt-1]
    tempH = tempH + BiasMatrix
    tempHt=tempHt+BiasMatrixT
    
    if ActivationFunction == "sig":
        H = 1/(1+np.exp(-tempH))
        Ht = 1/(1+np.exp(-tempHt))
    if ActivationFunction == "sin":
        H = np.sin(tempH)
        Ht = np.sin(tempHt)
    if ActivationFunction != "sig" and ActivationFunction!="sin":
        pass
    
    del tempH
    del tempHt
    
    n = NumberofHiddenNeurons
    
    #DAELM
    H=H.T
    Ht=Ht.T
    T=T.T
    Tt=Tt.T
    
    if NL == 0:
        A = Ht*H.T
        B = Ht*Ht.T+np.eye(NumberofTrainingData_Target)/Ct
        C=H*Ht.T
        D=H*H.T+np.eye(NumberofTrainingData)/Cs
        AlphaT=np.linalg.inv(B)*Tt-np.linalg.inv(B)*A*np.linalg.inv(C*np.linalg.inv(B)*A-D)*(C*np.linalg.inv(B)*Tt-T)
        AlphaS=inv(C*np.linalg.inv(B)*A-D)*(C*np.linalg.inv(B)*Tt-T)
        OutputWeight=H.T*AlphaS+Ht.T*AlphaT
    else:
        OutputWeight=np.linalg.inv(np.eye(n)+Cs*H.t*H+Ct*Ht.T*Ht)*(Cs*H.T*T+Ct*Ht.T*Tt)
    #  Calculate the accuracy
    
    Y=(H * OutputWeight).T
    
    tempH_test=InputWeight*TVP
    ind = np.ones(1,NumberofHiddenNeurons)
    BiasMatrix=BiasofHiddenNeurons[:,ind-1]
    tempH_test = tempH_test+BiasMatrix
    if ActivationFunction == "sig":
        H_test = 1/(1+np.exp(-tempH_test))
    if ActivationFunction == "sin":
        H_test = np.sin(tempH_test)
        
    TY = (H_test.T*OutputWeight).T
    # return the result of test but you need a sig to get prop
    if Elm_Type =="CLASSIFIER":
        return TY
    else:
        pass
    
    
                          
                          
                          
          







                