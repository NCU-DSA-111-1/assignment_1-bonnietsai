#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "function.h"

#define NUMINPUTS 2    
#define NUMHIDDENNODES 2
#define NUMOUTPUTS 1
#define LR 0.1f 
#define TRAIN_TIME 10000 //訓練次數
#define NUMTRAININGSETS 4 //訓練之資料量


/*****************************************************Main Begin*******************************************************/
int main() {

    //動態記憶體配置hiddenLayer、outputLayer
    double *hiddenLayer=(double *)malloc( NUMHIDDENNODES * sizeof(double) );
    double *outputLayer=(double *)malloc( NUMOUTPUTS * sizeof(double) );

    //動態記憶體配置hiddenLayerBias、outputLayerBias
    double *hiddenLayerBias=(double *)malloc( NUMHIDDENNODES * sizeof(double) );
    double *outputLayerBias=(double *)malloc( NUMOUTPUTS * sizeof(double) );

    //動態記憶體配置hiddenWeights、outputWeights
    double **hiddenWeights = (double **)malloc(NUMINPUTS * sizeof(void *));
    create_2D_array(hiddenWeights ,NUMINPUTS ,NUMHIDDENNODES);//將hiddenWeights變為2D array

    double **outputWeights = (double **)malloc(NUMHIDDENNODES * sizeof(void *));
    create_2D_array(outputWeights ,NUMHIDDENNODES ,NUMOUTPUTS);//將outputWeights變為2D array


    //動態記憶體配置training_inputs、training_outputs
    double **training_inputs = (double **)malloc(NUMTRAININGSETS * sizeof(void *));
    create_2D_array(training_inputs ,NUMTRAININGSETS ,NUMINPUTS);//將training_inputs變為2D array

    double **training_outputs = (double **)malloc(NUMTRAININGSETS * sizeof(void *));
    create_2D_array(training_outputs ,NUMTRAININGSETS ,NUMOUTPUTS);//將training_outputs變為2D array

    initial_data( training_inputs , training_outputs ); //初始化訓練資料

    initial_weight(hiddenWeights, hiddenLayerBias, outputWeights, outputLayerBias ); //初始化權重
    
    //動態記憶體配置 訓練順序
    int *trainingSetOrder=(int *)malloc( NUMTRAININGSETS * sizeof(int) );

    Change_TrainingSetOrder(trainingSetOrder); //改變訓練順序

    Training_Process(trainingSetOrder, hiddenLayerBias, training_inputs, 
                     hiddenWeights,    hiddenLayer,     outputLayerBias,
                     outputWeights,    outputLayer,     training_outputs  ); //訓練過程
    
    Print_Final_Weights(hiddenWeights,hiddenLayerBias,outputWeights,outputLayerBias); //印出最後之權重

     
    prediction_result( hiddenWeights, hiddenLayerBias, outputWeights, outputLayerBias ); //結果預測

    //釋放記憶體空間
    free_2D_array(hiddenWeights,   NUMINPUTS ,      NUMHIDDENNODES);
    free_2D_array(outputWeights,   NUMHIDDENNODES , NUMOUTPUTS);
    free_2D_array(training_inputs, NUMTRAININGSETS ,NUMINPUTS);
    free_2D_array(training_outputs,NUMTRAININGSETS ,NUMOUTPUTS);
    free(hiddenWeights);
    free(outputWeights);
    free(training_inputs);
    free(training_outputs);
    free(hiddenLayer);
    free(outputLayer);
    free(hiddenLayerBias);
    free(outputLayerBias);
    free(trainingSetOrder);

    return 0;
}

/***************************************************Main End****************************************************************/