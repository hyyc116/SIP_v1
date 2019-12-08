#coding:utf-8
'''
本文件使用LSTM进行预测

'''
import sys
sys.path.extend(['..','.'])
from paths import PATH
from basic_config import *
from tensorflow.keras import layers
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
import time

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

tf.keras.backend.set_floatx('float64')
from dataset.datasets_construction import construct_RNN_cat_datasets
from dataset.datasets_construction import unscale_dataset
from ED_model import Encoder
from ED_model import Decoder


def gru(units):
    # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
    # the code automatically does that.

    
    return tf.keras.layers.GRU(units, 
                                   return_sequences=True, 
                                   return_state=True, 
                                   recurrent_activation='tanh', 
                                   recurrent_initializer='glorot_uniform')


class JointModel(tf.keras.Model):

    def __init__(self,units,vocab_size,dropout_rate=0.5):

        super(JointModel,self).__init__()


        self._units = units

        ## encoder进行特征抽取
        self._encoder = Encoder(self._units)

        ## decoder进行序列解析
        self._decoder = Decoder(self._units)

        ## 对静态数据进行向量化
        self._static_fc = tf.keras.layers.Dense(self._units,activation = 'sigmoid')

        self._drp = tf.keras.layers.Dropout(rate=dropout_rate)

        ## 加一层中间层
        self._fc1 = tf.keras.layers.Dense(self._units,activation='sigmoid')
        self._drp2 = tf.keras.layers.Dropout(rate=dropout_rate)

        ## 进行label的预测
        self._fc = tf.keras.layers.Dense(vocab_size)



    def call(self,dynamic_features,static_features,Y,L,predict=False):

        batch_size = dynamic_features.shape[0]

        ## 首先需要encoder对数据进行特征抽取
        initial_state = tf.zeros((batch_size,self._units),tf.float64)
        ## 首先对输入进行encode
        enc_output, enc_hidden = self._encoder(dynamic_features,initial_state,predict=predict)

        ## regression decoder
        dec_input = tf.cast(tf.expand_dims([0]*batch_size,1),tf.float64)
        dec_hidden = enc_hidden
        all_predictions = []
        for t in range(Y.shape[1]):

            predictions,dec_hidden = self._decoder(dec_input,dec_hidden,static_features,predict=predict)

            all_predictions.append(predictions)
            dec_input = predictions

        regression_result = tf.concat(all_predictions,1)

        ## classification
        static_emd = self._static_fc(static_features)

        if not predict:
            static_emd = self._drp(static_emd)

        static_emd = self._fc1(static_emd)

        if not predict:
            static_emd = self._drp2(static_emd)

        # print(enc_output.shape)
        enc_output = tf.reshape(enc_output,(enc_output.shape[0],-1))
        # print(enc_output.shape)
        classification_result = self._fc(tf.concat([enc_output,static_emd],axis=1))

        return regression_result,classification_result


def regress_loss(real,pred):
    if real.shape!=pred.shape:
        print('ERROR: NOT SAME SHAPE IN RESULT.')

    loss = tf.keras.losses.MSE(real, pred)
    return tf.reduce_mean(loss)

def classi_loss(real,logits):
    loss = tf.keras.losses.sparse_categorical_crossentropy(real,logits,from_logits=False)
    return tf.reduce_mean(loss)



class S2SM:

    def __init__(self,pathObj,m,n,beta1=5,beta2=1):

        self._m = m
        self._n = n
        self._pathObj = pathObj

        scale = True
        ## 加载数据
        self._train_dynamic_X,self._train_static_X,self._train_Y,self._test_dynamic_X,\
        self._test_static_X,self._test_Y,self._valid_dynamic_X,self._valid_static_X,\
        self._valid_Y,self._test_sorted_ids,self._dx_mean,self._dx_std,\
        self._sx_mean,self._sx_std,self._y_mean,self._y_std,\
        self._train_L,self._test_L,self._valid_L= construct_RNN_cat_datasets(pathObj,m,n,scale)

        ## 数据集
        ## 超参数
        self._units = 256
        self._batch_sz = 512
        self._buffer_size = len(self._train_Y)
        self._n_batchs =self._buffer_size//self._batch_sz

        self._dataset = tf.data.Dataset.from_tensor_slices((self._train_dynamic_X,self._train_static_X,self._train_Y,self._train_L)).shuffle(self._buffer_size)
        self._dataset = self._dataset.batch(self._batch_sz, drop_remainder=True)

        print('train model on dataset sip-m{}n{}.'.format(m,n))

        ## label的数量，0,1,2,3,4,5

        self._vocab_size = 6
        ## 初始化encoder以及decoder
        self._model = JointModel(self._units,self._vocab_size)
        self._model_name = 'jont_model'


        ##
        self._beta1 = beta1
        self._beta2 = beta2

        ## optimizer
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

        ## 模型的保存位置
        self._checkpoint_dir = './trainning_checkpoints_{}_{}_{}'.format(self._model_name, m,n)
        self._avg_checkpoint_dir = './trainning_checkpoints_avg_{}_{}_{}'.format(self._model_name, m,n)


        self._checkpoint_prefix = os.path.join(self._checkpoint_dir, "ckpt")

        self._trackables = {}
        self._trackables['optimizer']=self._optimizer
        self._trackables['jointmodel']=self._model
        self._checkpoint = tf.train.Checkpoint(**self._trackables)


    def reload_latest_checkpoints(self):
        print('reload latest Checkpoint.....')
        self._checkpoint.restore(tf.train.latest_checkpoint(self._checkpoint_dir))

        # pass

    def train_step(self,dynamic_features,static_features,Y,L):


        with tf.GradientTape() as tape:

            regression_result, classification_result = self._model(dynamic_features,static_features,Y,L)

            loss = self._beta1*regress_loss(Y,regression_result)+ self._beta2*classi_loss(L,classification_result)

            batch_loss = (loss/int(Y.shape[1]))

            variables = self._model.trainable_variables

            gradients =  tape.gradient(loss,variables)

            self._optimizer.apply_gradients(zip(gradients,variables))

        return batch_loss

    def train(self):

        EPOCHS = 1000

        early_stop_count = 0
        best_mae = 100
        best_mse = 100
        best_r2 =0
        best_score = 0

        train_losses = []
        valid_losses = []
        test_result = {}

        test_result['IDS'] = self._test_sorted_ids

        for epoch in range(EPOCHS):
            start = time.time()


            total_loss = 0

            for (batch,(dynamic_features,static_features,targ,L)) in enumerate(self._dataset.take(self._n_batchs)):

                batch_loss  = self.train_step(dynamic_features,static_features,targ,L)

                total_loss+=batch_loss

                if (batch+1)%50==0 or (batch+1)==self._n_batchs:

                    print('sip-m{}n{}, Epoch {} Batch {}/{} Loss {:.4f}'.format(self._m,self._n,epoch+1,batch+1,self._n_batchs,batch_loss.numpy()))

            total_loss = total_loss/self._n_batchs

            ## 每一个回合结束对模型在valid上面的结果进行评价
            r2,mae,mse,all_predictions,acc = self.predict(self._valid_dynamic_X,self._valid_static_X,self._valid_Y,self._valid_L)
            
            train_losses.append(float(total_loss))
            valid_losses.append(float(mae))

            logging.info('sip-m{}n{}---beta1:{},beta2:{}, Epoch {}, training Loss {:.4f},validation mae:{}, mse:{},r2:{},acc:{},score:{:.3f},best_score:{:.3f}.'.format(self._m,self._n,self._beta1,self._beta2,epoch+1,total_loss,mae,mse,r2,acc,r2/(mae+mse),best_score))
            ## 每一回合的loss小于best_loss那么就将模型存下来
            ### 在实际的使用中并不能保存下最好的模型，
            ### 我们需要使用三个评价指标共同完成
            ### mae mse的前三位小数相同，并且r2更大
            if epoch>1 and is_better_result(mae,mse,r2,best_score,best_r2,best_mse,best_mae):

                best_mae = mae if mae<best_mae else best_mae
                best_mse = mse if mse<best_mse else best_mse
                best_r2 = r2 if r2>best_r2 else best_r2

                best_score = r2/(mae+mse)

                self._checkpoint.save(file_prefix = self._checkpoint_prefix)
                ## 使用保存的模型对test数据进行验证
                ## 在验证的时候 需要对数据进行unscale
                r2,mae,mse,all_predictions,acc = self.predict(self._test_dynamic_X,self._test_static_X,self._test_Y,self._test_L)

                logging.info('sip-m{}n{}, saved model, TEST MAE:{}, MSE:{},R2:{},acc:{}.'.format(self._m,self._n,mae,mse,r2,acc))

                test_result['summary'] = 'sip-m{}n{},beta1:{},beta2:{},{},{},{},{}'.format(self._m,self._n,self._beta1,self._beta2,self._model_name,r2,mae,mse)

                test_result['predictions'] = all_predictions.numpy().tolist()
                early_stop_count=0

            else:
                early_stop_count+=1

            print('Time taken for 1 epoch {} sec\n'.format(time.time()-start))

            ## 如果连续10个回合验证集的loss都不能大于目前保存最好的模型，停止训练
            if early_stop_count>=15:
                break

        ## 将训练过程中loss进行保存
        loss_obj = {}
        loss_obj['sip-m{}-n{}'.format(self._m,self._n)] = [list(train_losses),list(valid_losses)]
        self.save_losses(loss_obj,self._model_name)

        ## 最好的模型在TEST上面的结果进行保存
        result_path = self._pathObj.deep_result_prediction_path(self._model_name,self._m,self._n)
        self._pathObj.save_json(result_path,test_result)

        ## summary保存到文件里面
        with open(self._pathObj._deep_result_summary,'a') as f:
            summary = test_result['summary']
            f.write(summary+'\n')
            logging.info('Final performace on test is {}.'.format(summary))


    def save_losses(self,loss_obj,model):
        self._pathObj.save_json(self._pathObj.losses_file(self._m,self._n,model),loss_obj)


    def predict(self,dynamic_X,static_X,Y,L):

        dynamic_X = tf.convert_to_tensor(dynamic_X)
        static_X = tf.convert_to_tensor(static_X)
        Y = tf.convert_to_tensor(Y)
        L = tf.convert_to_tensor(L)

        ## validation set 进行验证
        regression_result,classification_result = self._model(dynamic_X,static_X,Y,L,True)

        Y = unscale_dataset(Y,self._y_mean,self._y_std)
        all_predictions = unscale_dataset(regression_result,self._y_mean,self._y_std)


        r2 =  r2_score(Y, all_predictions)
        mae = mean_absolute_error(Y, all_predictions)
        mse = mean_squared_error(Y, all_predictions)

        # acc = tf.keras.metrics.sparse_categorical_accuracy(L,classification_result)
        m = tf.keras.metrics.SparseCategoricalAccuracy()
        # print('L shape',L.shape)
        # print('Clas result：',classification_result.shape)
        m.update_state(tf.expand_dims(L,1), classification_result)

        mae = float('{:.3f}'.format(mae))
        mse = float('{:.3f}'.format(mse))
        r2 = float('{:.3f}'.format(r2))
        acc = float('{:.3f}'.format(m.result().numpy()))

        return r2,mae,mse,all_predictions,acc



def is_better_result(mae,mse,r2,best_score,best_r2,best_mse,best_mae):

    if r2/(mae+mse)>=best_score*0.95:

        if mae<best_mae or mse<best_mse or r2>best_r2:
            return True
    
    return False


def R_squared(y, y_pred):
  residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
  total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
  r2 = tf.subtract(1.0, tf.div(residual, total))
  return r2

if __name__ == '__main__':

    field = 'computer science'
    tag = 'cs'

    pathObj = PATH(field,tag)

    # mn_list=[(5,10),(3,10),(3,5),(5,5),(5,3),(3,3),(5,1),(3,1)(3,10),(3,5),]

    mn_list=[(3,3),(3,1)]
    # mn_list = [(3,3)]
    for m,n in mn_list:

        for beta1,beta2 in [(5,1),(5,5),(1,5),(10,1),(1,10)]:
            s2sm = S2SM(pathObj,m,n,beta1,beta2)
            s2sm.train()

            time.sleep(5)

