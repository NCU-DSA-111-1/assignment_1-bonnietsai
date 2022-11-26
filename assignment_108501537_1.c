#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUMINPUTS 2    
#define NUMHIDDENNODES 2
#define NUMOUTPUTS 1
#define LR 0.1f 
#define TRAIN_TIME 10000 //訓練次數
#define NUMTRAININGSETS 4 //訓練之資料量

/***********************************************function Begin*************************************************/

double sigmoid(double x) { return 1 / (1 + exp(-x)); } //activation function
double dSigmoid(double x) { return x * (1 - x); } //differential activation function
double init_weight() { return ((double)rand())/((double)RAND_MAX); } //初始化權重
void initial_data(double **training_inputs , double **training_outputs ) {

    //訓練之資料(輸入)
    *(*(training_inputs + 0) + 0)=0;
    *(*(training_inputs + 0) + 1)=0;
    *(*(training_inputs + 1) + 0)=0;
    *(*(training_inputs + 1) + 1)=1;
    *(*(training_inputs + 2) + 0)=1;
    *(*(training_inputs + 2) + 1)=0;
    *(*(training_inputs + 3) + 0)=1;
    *(*(training_inputs + 3) + 1)=1;
    //訓練之資料(輸出)
    *(*(training_outputs + 0) + 0)=0;
    *(*(training_outputs + 1) + 0)=1;
    *(*(training_outputs + 2) + 0)=1;
    *(*(training_outputs + 3) + 0)=0;

}

void initial_weight(double **hiddenWeights,double *hiddenLayerBias,double **outputWeights,double *outputLayerBias )
{//初始化權重 begin
    
    for (int i=0; i<NUMINPUTS; i++) {
        for (int j=0; j<NUMHIDDENNODES; j++) {
            *(*(hiddenWeights + i) + j) = init_weight();
        }
    }
    for (int i=0; i<NUMHIDDENNODES; i++) {
        *(hiddenLayerBias+i) = init_weight();
        for (int j=0; j<NUMOUTPUTS; j++) {
            //*(*(training_inputs + i) + j) = init_weight();
            *(*(outputWeights + i) + j) = init_weight();
        }
    }
    for (int i=0; i<NUMOUTPUTS; i++) {
        *(outputLayerBias+i) = init_weight();
    }
    

}//初始化權重 end

//改變資料訓練順序
void Change_TrainingSetOrder(int *trainingSetOrder){

    for(int i=0; i < NUMTRAININGSETS; ++i){
        *(trainingSetOrder+i)=i;       
    }
}

//將訓練資料做隨機排列
void shuffle(int* array, size_t n) 
{
    if (n > 1)
    {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = *(array+j); 
            *(array+j) = *(array+i);
            *(array+i) = t;
        }
    }
}//shuffle end


//將Duuble Ptr 指向連續的記憶體空間
void create_2D_array(double** doublePtr,int row,int colume){


    for(int i=0; i < row; ++i)
        *(doublePtr+i) = (double *)malloc(colume * sizeof(double *));

}//create_2D_array end

void Training_Process(int *trainingSetOrder,  double *hiddenLayerBias, double **training_inputs, 
                      double **hiddenWeights, double *hiddenLayer,     double *outputLayerBias,
                      double **outputWeights, double *outputLayer,     double **training_outputs)
{//訓練過程 begin
    
    double Loss_function=0 ;

    for (int n=0; n < TRAIN_TIME; n++) {

        shuffle(trainingSetOrder,NUMTRAININGSETS);
        printf("Train Time : %d\n", n); 
        for (int x=0; x<NUMTRAININGSETS; x++) {
            
            int i = *(trainingSetOrder+x);
            
            // Forward pass
            
            for (int j=0; j<NUMHIDDENNODES; j++) {
                double activation=*(hiddenLayerBias+j);
                 for (int k=0; k<NUMINPUTS; k++) {
                    activation+=(*(*(training_inputs + i) + k))*(*(*(hiddenWeights + k) + j));
                }
                *(hiddenLayer+j) = sigmoid(activation);
            }
            
            for (int j=0; j<NUMOUTPUTS; j++) {
                double activation=*(outputLayerBias+j);
                for (int k=0; k<NUMHIDDENNODES; k++) {
                    activation += *(hiddenLayer+k) * (*(*(outputWeights + k) + j));
                }
                *(outputLayer+j) = sigmoid(activation);
            }
            printf("Input:");                  printf("%.0f %.0f", *(*(training_inputs + i) + 0) , *(*(training_inputs + i) + 1) );
            printf("    Output:");             printf("%.6f", *outputLayer );
            printf("    Expected Output: ");   printf("%.0f ",   *(*(training_outputs + i) + 0)   );
            printf("    Loss Function MSE: ");   printf("%.6f \n", Loss_function );
           
            
            // Backprop
            double *deltaOutput=(double *)malloc( NUMOUTPUTS * sizeof(double) );


            for (int j=0; j<NUMOUTPUTS; j++) {
                double errorOutput = (*(*(training_outputs + i) + j)-*(outputLayer + j));
                *(deltaOutput+j) = errorOutput*dSigmoid(*(outputLayer+j));
                Loss_function=0.5 * errorOutput * errorOutput;
            }

            double *deltaHidden=(double *)malloc( NUMHIDDENNODES * sizeof(double) );


            for (int j=0; j<NUMHIDDENNODES; j++) {
                double errorHidden = 0.0f;
                for(int k=0; k<NUMOUTPUTS; k++) {
                    errorHidden += *(deltaOutput+k) *  *(*(outputWeights + j) + k);
                }
                *(deltaHidden+j) = errorHidden*dSigmoid(*(hiddenLayer+j));
            }
            
            for (int j=0; j<NUMOUTPUTS; j++) {
                *(outputLayerBias+j) += *(deltaOutput+j)*LR;
                for (int k=0; k<NUMHIDDENNODES; k++) {
                    *(*(outputWeights + k) + j) += *(hiddenLayer+k) * (*(deltaOutput+j))* LR ;
                }
            }
            
            for (int j=0; j<NUMHIDDENNODES; j++) {
                *(hiddenLayerBias+j) += *(deltaHidden+j)*LR;
                for(int k=0; k<NUMINPUTS; k++) {
                    *(*(hiddenWeights + k) + j)+=*(*(training_inputs + i) + k)*  *(deltaHidden+j) *LR;
                }
            }

        }
    }
}//訓練過程 end

void Print_Final_Weights(double **hiddenWeights,double *hiddenLayerBias,double **outputWeights,double *outputLayerBias){
    // Print weights
    printf("Final Hidden Weights\n[ ");  
    for (int j=0; j<NUMHIDDENNODES; j++) {
        printf("[ ");  
        for(int k=0; k<NUMINPUTS; k++) {
            printf("%f ",*(*(hiddenWeights + k) + j));              
        }
        printf(" ]");
    }
    printf(" ]\n");

    printf("Final Hidden Biases\n[ ");
    for (int j=0; j<NUMHIDDENNODES; j++) {
        printf("%f ",*(hiddenLayerBias+j)); 
    }
    printf(" ]\n");
    printf("Final Output Weights");
    for (int j=0; j<NUMOUTPUTS; j++) {
        printf("[ ");
        for (int k=0; k<NUMHIDDENNODES; k++) {
            printf("%f ",*(*(outputWeights + k) + j)); 
        }
        printf("]\n");
    }
    printf("Final Output Biases\n[ ");
    for (int j=0; j<NUMOUTPUTS; j++) {
        printf("%f ",*(outputLayerBias+j)); 
    }
    printf("]\n");

}// Print weights End

//將訓練好之權重 換作成NN架構之公式，對輸入之數值做XOR的判斷
void prediction_result(  double **hiddenWeights,
                         double *hiddenLayerBias,
                         double **outputWeights,
                         double *outputLayerBias)
{   
    int first_bit, second_bit;

    printf("\n **** Please input the two bits number Separately for predict XOR Logic (ex:00,01,10,11) : ****"); 

    printf("\n First bit : ");   
    scanf("%d", &first_bit);

    printf(" Second bit : "); 
    scanf("%d", &second_bit);

    double neuron1 , neuron_temp1;
    double neuron2 , neuron_temp2;

    neuron_temp1 = first_bit * *(*(hiddenWeights + 0) + 0) +  second_bit * *(*(hiddenWeights + 1) + 0) + *(hiddenLayerBias+0) ;
    neuron_temp2 = first_bit * *(*(hiddenWeights + 0) + 1) +  second_bit * *(*(hiddenWeights + 1) + 1) + *(hiddenLayerBias+1) ;

    double outputLayer_temp;
    outputLayer_temp = sigmoid(neuron_temp1) * *(*(outputWeights + 0) + 0) + sigmoid(neuron_temp2) * *(*(outputWeights + 1) + 0) + *(outputLayerBias+0) ;

    printf(" output result : %.0f !!!\n\n\n", sigmoid(outputLayer_temp) );

}//prediction_result end

void free_2D_array(double** doublePtr,int row,int colume){


    for(int i=0; i < row; ++i)
        free(*(doublePtr+i));

}//create_2D_array end


/****************************************************function End******************************************************/


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