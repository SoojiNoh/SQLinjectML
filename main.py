import pandas as pd
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, Adagrad, Adadelta
import tensorboard
from tensorflow.python.framework.graph_io import write_graph
from asyncore import write

class KISIA_Edu_Test:
    def make_mlp_from_kreas(self,input_dim,learning_rate, dropout_ratio=0.4):
        model=Sequential()            
        model.add(Dense(256,input_dim=input_dim,activation='relu', kernel_initializer='he_normal'))     
        model.add(BatchNormalization())
        model.add(Dropout(dropout_ratio))        
        model.add(Dense(64,activation='relu',kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_ratio))        
        model.add(Dense(64,activation='relu',kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_ratio))        
        model.add(Dense(2,activation='softmax'))

        optimizer=Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy',optimizer=optimizer,
                    metrics=['accuracy'])
        return model
    
    def run_model_from_keras(self, model,X_train,y_train,epochs,batch_size):
        #from keras.callbacks import TensorBoard
        
        #tb_hist = Tensorboard(log_dir="logs", histogram_freq=10,batch_size=30, write_graph=True, write_images = True)
        
         
        #model.fit(X_train,y_train,epochs=epochs, batch_size=batch_size,verbose=1, validation_split=0.1, callbacks=[tb_hist])
        model.fit(X_train,y_train,epochs=epochs, batch_size=batch_size,verbose=2)

        return model
     
    def eval_model_from_keras(self,model,X_test,y_test):
        _loss, _acc=model.evaluate(X_test,y_test)
        print('loss: {:.3f}, accuracy: {:.3f}'.format(_loss,  _acc))
        return model

    def predict_model_from_keras(self,model,X_test):
        y_pred_res=model.predict([X_test])
        return y_pred_res
        
    def replace_column_with_one_hot_encoding_from_pandas(self,df,column_idx):
        one_hot=pd.get_dummies(df.iloc[:,column_idx])
        df=one_hot
        return df
    
if __name__ == "__main__":
    train_data_csv_fname='train.csv'
    test_data_csv_fname='test.csv'    
    # csv 파일 분리
    ket=KISIA_Edu_Test()
    X_var_no=200  # 독립변수 x는 200개
#     y변수는 2개 값 one-hot encoding 필요 
    _epoch = 200  #반복회수
    YClassName='Attack'    
    # 학습 데이터셋에서 dataframe 생성
    train_df=pd.read_csv(train_data_csv_fname)
    print(train_df.head())  #train_df.count (841,201)  
#     df.reset_index(drop=True)
    #테스트 데이터셋에서 dataframe 생성
    test_df=pd.read_csv(test_data_csv_fname)
    print(test_df.head()) #train_df.count (93,201)
#     test_df.columns=list(range(201))
#     df.reset_index(drop=True)
    # 학습 데이셋에서 X 변수들 가져오기 200개
    X_train_df=train_df.iloc[:,1:]
    # 테스트 데이터셋에서 X변수 가져오기 200개
    X_test_df=test_df.iloc[:,1:]
    print(X_train_df.head())
    print(X_test_df.head())
    y_train_seri=train_df.iloc[:,0]
    # 
    y_test_seri=test_df.iloc[:,0]
    print(y_train_seri.head())
    print(y_test_seri.head())
    
    y_train_df=pd.DataFrame(y_train_seri)
    y_train_df=ket.replace_column_with_one_hot_encoding_from_pandas(y_train_df, 0)
    print(y_train_df.head())
    
        
    y_test_df=pd.DataFrame(y_test_seri)
    y_test_df=ket.replace_column_with_one_hot_encoding_from_pandas(y_test_df,0)
    print(y_test_df.head())
    
    model=ket.make_mlp_from_kreas(input_dim=X_var_no,learning_rate=1e-6,
          dropout_ratio=0.6)
    model=ket.run_model_from_keras(model,X_train_df,y_train_df,
                               epochs=_epoch,batch_size=100)
    model=ket.eval_model_from_keras(model,X_test_df,y_test_df)    
    y_pred_res=ket.predict_model_from_keras(model,X_test_df)
    print(y_pred_res)
    
    
#     Epoch 198/200
# 9/9 - 0s - loss: 0.6868 - accuracy: 0.6643
# Epoch 199/200
# 9/9 - 0s - loss: 0.6882 - accuracy: 0.6329
# Epoch 200/200
# 9/9 - 0s - loss: 0.6870 - accuracy: 0.6643
# 
# 1/3 [=========>....................] - ETA: 0s - loss: 0.6803 - accuracy: 0.9062
# 2/3 [===================>..........] - ETA: 0s - loss: 0.6808 - accuracy: 0.8906
# 3/3 [==============================] - 0s 28ms/step - loss: 0.6817 - accuracy: 0.8611
# loss: 0.682, accuracy: 0.861
# [[0.50797933 0.49202067]
#  [0.50797933 0.49202067]
#  [0.50797933 0.49202067]
#  [0.50797933 0.49202067]
#  [0.50797933 0.49202067]
#  [0.50797933 0.49202067]
#  [0.50797933 0.49202067]
#  