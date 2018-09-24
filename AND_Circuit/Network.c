// 参考: 東大松尾研lecture の chapter 4
// AND回路の学習

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

// -------------------------------------------------------------------------------
// 使用関数
// -------------------------------------------------------------------------------
void sigmoid(int N, double X[N], double Z[N]);
void deriv_sigmoid(int N, double X[N], double Z[N]);

void forward(int N1, int N2, double W[N1][N2], double b[N1], double X[N2], double Y[N1]);
double train(int N1, int N2, double W[N1][N2], double b[N1], double X[N2], double T[N1], double learning_rate);
double test(int N1, int N2, double W[N1][N2], double b[N1], double X[N2], double T[N1], double Y[N1]);

void MV(int N1, int N2, double M[N1][N2], double X[N2], double Y[N1]);
void input_V(int N, double X[N], double Y[N]);



// -------------------------------------------------------------------------------
// main関数
// -------------------------------------------------------------------------------
void main(void){
				printf("Hello Deep!\n");

				int MAX_EPOCH = 1000; // 学習の回数
				double lr = 0.1;	// 学習率

				// トレーニングデータとテストデータを用意
				double train_X[4][2] = {{0.,0.},{0.,1.},{1.,0.},{1.,1.}};
				double train_Y[4][1] = {{0.},{0.},{0.},{1.}};
				double test_X[4][2] = {{0.,0.},{0.,1.},{1.,0.},{1.,1.}};
				double test_Y[4][1] = {{0.},{0.},{0.},{1.}};
				// 重みWとバイアスbを生成
				double W[1][2] = {{(double)rand()/RAND_MAX,(double)rand()/RAND_MAX}};
				double b[1] = {0.0};
				double out[2];
				double cost;
				double pred_Y[1];


				// 学習開始
				for(int epoch=1; epoch<MAX_EPOCH+1; epoch++){
								for(int l=0; l<4; l++){
												cost = train(1,2,W,b,train_X[l], train_Y[l],lr);
								}
								printf("Epoch: %5d  | Cost: %3.5f\n",epoch,cost);
								
				
				}


				// テスト
				printf("-----------------------------------------------------\n");
				printf("Final Result\n");
				for(int l=0; l<4; l++){
								cost = test(1,2,W,b,test_X[l], test_Y[l],pred_Y);
								printf("input(%1.0f,%1.0f) -> %1f\n", test_X[l][0], test_X[l][1], pred_Y[0]);
				}
}


// -------------------------------------------------------------------------------
// 活性化関数
// -------------------------------------------------------------------------------
void sigmoid(int N, double X[N], double Z[N]){
				double buf[N];
				for(int i=0; i<N; i++){
								buf[i] = 1.0 / (1.0+exp(-X[i]));
				}
				input_V(N, buf, Z);
}

void deriv_sigmoid(int N, double X[N], double Z[N]){
				double buf[N];
				for(int i=0; i<N; i++){
								buf[i] = exp(-X[i]) / ((1.0+exp(-X[i])) * (1.0+exp(-X[i])));
				}
				input_V(N, buf, Z);
}



// -------------------------------------------------------------------------------
// ニューラルネットワークの関数定義
// コスト関数: クロスエントロピー
// 最適化法: 確立勾配法
// -------------------------------------------------------------------------------

void forward(int N1, int N2, double W[N1][N2], double b[N1], double X[N2], double Y[N1]){
				double buf[N1];

				MV(N1,N2,W,X,buf);
				for(int i=0; i<N1; i++){buf[i] += b[i];}
				sigmoid(N1,buf,Y);
}

double train(int N1, int N2, double W[N1][N2], double b[N1], double X[N2], double T[N1], double learning_rate){
				double Y[N1];
				double delta[N1];
				double cost=0;


				// Forward Propagation
				forward(N1,N2,W,b,X,Y);

				// Back Propagation 
				for(int i=0; i<N1; i++){
								cost += -T[i]*log(Y[i]) - (1.0-T[i])*log(1.0-Y[i]);
								delta[i] = Y[i] - T[i];
				}

				// Updata Parameters
				for(int i=0; i<N2; i++){
								for(int j=0; j<N1; j++){
												W[j][i] -= learning_rate * X[i]*delta[j];
								}
				}
				for(int i=0; i<N1; i++){
								b[i] -= learning_rate * delta[i];
				}
				return cost;
}


double test(int N1, int N2, double W[N1][N2], double b[N1], double X[N2], double T[N1], double Y[N1]){
				double cost=0;

				// Forward Propagation
				forward(N1,N2,W,b,X,Y);

				for(int i=0; i<N1; i++){
								cost += -T[i]*log(Y[i]) - (1.0-T[i])*log(1.0-Y[i]);
				}
				return cost;
}




// -------------------------------------------------------------------------------
// 行列演算のための処理
// -------------------------------------------------------------------------------
void MV(int N1, int N2, double M[N1][N2], double X[N2], double Z[N1]){
				double sum[N1];
				for(int i=0; i<N1; i++){
								sum[i] = 0.0;
								for(int j=0; j<N2; j++){
												sum[i] += M[i][j]*X[j];
								}
				}
				input_V(N1,sum,Z);
}

void input_V(int N, double X[N], double Y[N]){
				for(int i=0; i<N; i++){Y[i] = X[i];}
}


