#coding:utf-8
'''
将回归问题转化为分类问题

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
from beam_search import greedy_search


def gru(units):
    # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
    # the code automatically does that.
    return tf.keras.layers.GRU(units, 
                                   return_sequences=True, 
                                   return_state=True, 
                                   recurrent_activation='sigmoid', 
                                   recurrent_initializer='glorot_uniform')

class Decoder(tf.keras.Model):

    def __init__(self,dec_units,vocab_size, dropout_rate=0.5):

        super(Decoder,self).__init__()

        self._dec_units = dec_units

        self._gru = gru(self._dec_units)

        ## 每一个Y即是位置也是对应的值，vocab size对应的是Y的最大值
        self._vocab_size = vocab_size

        ## 对每一个Y进行encoding,size和units数量一致
        self._y_embedding =  tf.keras.layers.Embedding(vocab_size,dec_units)

        # self._predict = predict

        ## 对静态特征进行抽取
        self._static_fc = tf.keras.layers.Dense(self._dec_units)

        self._rnn_dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self._static_fc_dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        ## 回归 每一步输出一个数字
        self._fc = tf.keras.layers.Dense(self._vocab_size)


    def call(self,decoder_input,enc_output,static_features,predict=False):

        ## decoder input进行embedding
        decoder_input = tf.squeeze(self._y_embedding(decoder_input))

        # print(decoder_input.shape)

        ## 将static feture的shape从（batch_size,2) 变成（batch_size,units)
        static_features = self._static_fc(static_features)

        if not predict:
            static_features = self._static_fc_dropout(static_features)

        x = tf.concat([enc_output,decoder_input,static_features],axis=-1)

        x = tf.expand_dims(x,1)

        output,state = self._gru(x)

        if not predict:
            output = self._rnn_dropout(output)

        output = tf.reshape(output,(-1,output.shape[2]))

        x = self._fc(output)


        return x,state

    def initialize_hidden_state(self,size):
        return tf.zeros((size,self._dec_units),tf.float64)

# huber = tf.keras.losses.Huber(delta=10)
def loss_function(real,pred):

    # if real.shape!=pred.shape:
    #     print('ERROR: NOT SAME SHAPE IN RESULT.')

    ## 使用交叉上作为损失函数
    loss = tf.keras.losses.sparse_categorical_crossentropy(real,pred,from_logits=False)

    return tf.reduce_mean(loss)

# def rl_loss(real,pred):

#     loss = tf.keras.losses.MSE(real, pred)

#     return tf.reduce_mean(loss)

class S2SM:

    def __init__(self,pathObj,m,n):

        self._m = m
        self._n = n
        self._pathObj = pathObj

        scale = False
        ## 加载数据
        self._train_dynamic_X,self._train_static_X,self._train_Y,self._test_dynamic_X,\
        self._test_static_X,self._test_Y,self._valid_dynamic_X,self._valid_static_X,\
        self._valid_Y,self._test_sorted_ids,self._dx_mean,self._dx_std,\
        self._sx_mean,self._sx_std,self._y_mean,self._y_std,\
        self._train_ins_Y,self._test_ins_Y,self._valid_ins_Y,\
        self._y_id,self._id_y,self._train_last_Y,self._test_last_Y,self._valid_last_Y = construct_RNN_cat_datasets(pathObj,m,n,scale)

        ## 数据集
        ## 超参数
        self._units = 256
        self._batch_sz = 256
        self._buffer_size = len(self._train_Y)
        self._n_batchs =int(self._buffer_size//self._batch_sz*0.6)
        # self._n_batchs =50

        self._valid_n_batchs = len(self._valid_Y)//self._batch_sz
        self._test_n_batchs = len(self._test_Y)//self._batch_sz


        self._dataset = tf.data.Dataset.from_tensor_slices((self._train_dynamic_X,self._train_static_X,self._train_Y,self._train_ins_Y,self._train_last_Y)).shuffle(self._buffer_size)
        self._dataset = self._dataset.batch(self._batch_sz, drop_remainder=True)


        self._valid_dataset = tf.data.Dataset.from_tensor_slices((self._valid_dynamic_X,self._valid_static_X,self._valid_Y,self._valid_ins_Y,self._valid_last_Y)).shuffle(len(self._valid_Y))
        self._valid_dataset = self._valid_dataset.batch(self._batch_sz,drop_remainder=False)
            
        self._test_dataset = tf.data.Dataset.from_tensor_slices((self._test_dynamic_X,self._test_static_X,self._test_Y,self._test_ins_Y,self._test_last_Y)).shuffle(len(self._test_Y))
        self._test_dataset = self._test_dataset.batch(self._batch_sz,drop_remainder=False)


        print('train model on dataset sip-m{}n{}.'.format(m,n))


        ## 初始化encoder以及decoder
        self._encoder = Encoder(self._units)


        ## 加一个start的标志
        # length = len(self._y_id)
        # print(self._y_id)
        # print(self._id_y)

        ## 
        # self._sorted_ys =[self._id_y[_id] for _id in sorted(self._id_y.keys())]

        self._start_tag = -1
        # self._y_id[ self._start_tag] = length
        # self._id_y[length]= self._start_tag

        ## 需要最大的Y,所有位置的最大值
        vocab_size = len(self._y_id)
        print('vocab size:',vocab_size)

        self._vocab_size = vocab_size

        self._decoder = Decoder(self._units,vocab_size)
        self._model_name = 'ED_CAT_model'

        ## optimizer
        # self._optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001,clipvalue=1)
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        ## 模型的保存位置
        self._checkpoint_dir = './trainning_checkpoints_{}_{}_{}'.format(self._model_name, m,n)

        self._avg_checkpoint_dir = './trainning_checkpoints_avg_{}_{}_{}'.format(self._model_name, m,n)


        self._checkpoint_prefix = os.path.join(self._checkpoint_dir, "ckpt")

        self._trackables = {}
        self._trackables['optimizer']=self._optimizer
        self._trackables['encoder']=self._encoder
        self._trackables['decoder']=self._decoder
        self._checkpoint = tf.train.Checkpoint(**self._trackables)


    def reload_latest_checkpoints(self):
        print('reload latest Checkpoint.....')
        self._checkpoint.restore(tf.train.latest_checkpoint(self._checkpoint_dir))

        # pass

    def get_teacher_forcing_rate(self,epoch):
        rate = np.exp(-epoch/50)
        return rate if rate>0.1 else 0.1

    def train_step(self,dynamic_features,static_features,targ,ins_targ,last_y,enc_hidden,teacher_forcing_rate):


        with tf.GradientTape() as tape:

            # print('feature shape:{},data type:{}.'.format(dynamic_features.shape,dynamic_features.dtype))

            enc_output, enc_hidden = self._encoder(dynamic_features,enc_hidden)

            ## 需要对enc_output的shape进行查看,并列输入了8种序列特征
            # print('Shape of enc output:{}'.format(enc_output.shape))
            # print('Shape of enc hidden:{}'.format(enc_hidden.shape))

            dec_input = tf.expand_dims([self._y_id[self._start_tag]]*self._batch_sz,1)

            # print('===dec input shape {}'.format(dec_input.shape))

            # print('===dec hidden shape {}'.format(dec_hidden.shape))
            # print('===enc output shape {}'.format(enc_output.shape))

            # print('target shape {},data type {}'.format(targ.shape,targ.dtype))
            # print('targ shape {}'.format(targ.shape))

            dec_hidden = enc_hidden

            loss = 0

            greedy_predictions = []

            sample_predictions = []

            length = targ.shape[1]

            for t in range(0,length):

                # print(dec_input.numpy())

                predictions,dec_hidden = self._decoder(dec_input,dec_hidden,static_features)

                probs = tf.log_softmax(predictions)

                # print(targ[:,t].shape,predictions.shape)
                ## MLE进行计算
                loss += loss_function(tf.expand_dims(ins_targ[:,t],1),predictions)
                
                
                ## sampling from result
                # tf.random.categorical(probs,k) 



                greedy_predictions.append(tf.expand_dims(tf.argmax(predictions,axis=1),1))

                # 时间t的标准结果作为t+1的x
                # if rn<teacher_forcing_rate:
                dec_input = tf.expand_dims(ins_targ[:,t],1)
                    # print(dec_input.shape)
                # else:
                    #如果不适用teacher forcing
                    # dec_input = tf.expand_dims(tf.argmax(predictions,axis=1),1)

                    # print(dec_input.shape)

                ## 根据predictions的概率分布进行随机抽样K个

            # beam_predictions,probs,last_costs = beam_search(self._decoder,dec_input,dec_hidden,static_features,self._vocab_size,10,length)

            # mae,mse = self.rl_loss(greedy_predictions,beam_predictions,last_y,targ)

            # print(mse.shape)

            # loss = 0.9*loss - 0.1*tf.reduce_mean(-last_costs*mse)

            batch_loss = (loss/int(targ.shape[1]))

            variables = self._encoder.trainable_variables + self._decoder.trainable_variables

            gradients =  tape.gradient(loss,variables)

            self._optimizer.apply_gradients(zip(gradients,variables))

        return batch_loss

    def train(self):

        EPOCHS = 1000

        early_stop_count = 0
        # best_mae = 100
        # best_mse = 100
        # best_r2 =0
        best_score = 0

        train_losses = []
        valid_losses = []
        test_result = {}

        test_result['IDS'] = self._test_sorted_ids

        for epoch in range(EPOCHS):
            start = time.time()

            enc_hidden = self._encoder.initialize_hidden_state(self._batch_sz)

            total_loss = 0

            teacher_forcing_rate = self.get_teacher_forcing_rate(epoch)

            for (batch,(dynamic_features,static_features,targ,ins_targ,last_y)) in enumerate(self._dataset.take(self._n_batchs)):
                ## 训练的时候使用相对增益
                batch_loss  = self.train_step(dynamic_features,static_features,targ,ins_targ,last_y,enc_hidden,teacher_forcing_rate)

                total_loss+=batch_loss

                if (batch+1)%50==0 or (batch+1)==self._n_batchs:

                    print('sip-m{}n{}, Epoch {} Batch {}/{} Loss {:.4f}'.format(self._m,self._n,epoch+1,batch+1,self._n_batchs,batch_loss.numpy()))

            total_loss = total_loss/self._n_batchs

            ## 每一个回合结束对模型在valid上面的结果进行评价
            ## 预测的时候使用的是绝对值
            r2,mae,mse,all_predictions = self.predict(self._valid_dataset,self._valid_n_batchs)
            
            train_losses.append(float(total_loss))
            valid_losses.append(float(mae))

            logging.info('sip-m{}n{}, Epoch {}, training Loss {:.4f},validation mae:{}, mse:{},r2:{},score:{:.3f},best_score:{:.3f}.'.format(self._m,self._n,epoch+1,total_loss,mae,mse,r2,r2/(mae+mse),best_score))
            ## 每一回合的loss小于best_loss那么就将模型存下来
            ### 在实际的使用中并不能保存下最好的模型，
            ### 我们需要使用三个评价指标共同完成
            ### mae mse的前三位小数相同，并且r2更大
            if epoch>10 and is_better_result(mae,mse,r2,best_score):

                # best_mae = mae if mae<best_mae else best_mae
                # best_mse = mse if mse<best_mse else best_mse
                # best_r2 = r2 if r2>best_r2 else best_r2

                best_score = r2/(mae+mse)

                self._checkpoint.save(file_prefix = self._checkpoint_prefix)
                ## 使用保存的模型对test数据进行验证
                ## 在验证的时候 需要对数据进行unscale
                r2,mae,mse,all_predictions = self.predict(self._test_dataset,self._test_n_batchs)

                logging.info('sip-m{}n{}, saved model, TEST MAE:{}, MSE:{},R2:{}.'.format(self._m,self._n,mae,mse,r2))

                test_result['summary'] = 'sip-m{}n{},{},{},{},{}'.format(self._m,self._n,self._model_name,r2,mae,mse)

                test_result['predictions'] = all_predictions.numpy().tolist()
                early_stop_count=0

            else:
                early_stop_count+=1

            print('Time taken for 1 epoch {} sec\n'.format(time.time()-start))

            ## 如果连续10个回合验证集的loss都不能大于目前保存最好的模型，停止训练
            if early_stop_count>=50:
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


    def predict(self,dataset,n_batch,isTest=False,name='valid'):

        ## validation set 进行验证
        # self._valid_static_X,self._valid_dynamic_X,self._valid_Y
        print('Predicting, batch size: {} ...'.format(n_batch))
        mae,mse,r2 = 0.0,0.0,0.0
        for (batch,(dynamic_X,static_X,targ,ins_targ,last_Y)) in enumerate(dataset.take(n_batch)):

            if (batch+1)%10==0 or (batch+1)==n_batch:
                print('predicting progress {}/{}..'.format(batch+1,n_batch))

            valid_size = dynamic_X.shape[0]
            # 初始化encoder的hideen state
            initial_state = tf.zeros((valid_size,self._units),tf.float64)
            ## 输入encoder
            enc_output,enc_hidden = self._encoder(dynamic_X,initial_state,True)
            dec_input = tf.expand_dims([self._y_id[self._start_tag]]*valid_size,1)

            ## 循环predict
            dec_hidden = enc_hidden
            last_Y = tf.expand_dims(last_Y,1)
            length = targ.shape[1]
            all_predictions,probs,last_costs = beam_search(self._decoder,dec_input,dec_hidden,static_X,self._vocab_size,10,length)
            # print(all_predictions.shape)
            # print(probs.shape)
            # all_predictions = tf.cast([[[self._id_y[int(predicted_id)]] for prediction_ids in all_predictions.numpy() for predicted_id in prediction_ids ]],dtype=tf.int32)

            all_predictions = tf.cast([[ self._id_y[int(predicted_id)] for predicted_id in batch_prediction] for batch_prediction in all_predictions],dtype=tf.int32)


            # print(all_predictions.shape)
            # print(last_Y.shape)

            all_predictions+=last_Y

            all_predictions = tf.clip_by_value(all_predictions,0,100000)

            r2+=r2_score(targ, all_predictions)
            mae+=mean_absolute_error(targ, all_predictions)
            mse+=mean_squared_error(targ, all_predictions)

        mae = float('{:.3f}'.format(mae/n_batch))
        mse = float('{:.3f}'.format(mse/n_batch))
        r2 = float('{:.3f}'.format(r2/n_batch))

        return r2,mae,mse,all_predictions


    def get_eval_score_by_id(self,all_predictions,last_Y,targ):

        all_predictions = tf.cast([[ self._id_y[int(predicted_id)] for predicted_id in batch_prediction] for batch_prediction in all_predictions],dtype=tf.int32)
        last_Y = tf.expand_dims(last_Y,1)

        all_predictions+=last_Y

        all_predictions = tf.clip_by_value(all_predictions,0,100000)

        # r2=r2_score(targ, all_predictions)
        mae=tf.keras.losses.MAE(targ, all_predictions)
        mse=tf.keras.losses.MSE(targ, all_predictions)

        return tf.cast(mae,dtype=tf.float64),tf.cast(mse,dtype=tf.float64)


    def rl_loss(self,greedy_predictions,beam_predictions,last_Y,targ):

        greedy_mae,greedy_mse = self.get_eval_score_by_id(greedy_predictions,last_Y,targ)

        beam_mae,beam_mse = self.get_eval_score_by_id(beam_predictions,last_Y,targ)


        return beam_mae-beam_mae,beam_mse-greedy_mse


def is_better_result(mae,mse,r2,best_score):

    if r2/(mae+mse)>=best_score:
        return True
    else:
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

    # mn_list=[(5,10),(3,10),(3,5),(5,5),(5,3),(3,3),(5,1),(3,1)]

    mn_list=[(3,10),(3,5),(3,3),(3,1)]

    for m,n in mn_list:
        s2sm = S2SM(pathObj,m,n)
        s2sm.train()

        time.sleep(5)

