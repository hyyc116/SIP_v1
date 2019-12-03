import numpy as np
import tensorflow as tf

def beam_search(decoder,dec_input,dec_state, static_features,vocab_size, beam_width=4,  max_length=50,predict=True):

    ''' 
    :decoder的输出为predictions logits的预测，hideen 上一步的隐藏变量，probs 每一步预测的概率
    ''' 

    batch_size = dec_input.shape[0]

    # print('batch size,',batch_size)

    ## (batch_size,beam_width,1)
    dec_inputs = tf.concat([tf.expand_dims(dec_input,1) for _ in range(beam_width)],axis=1)
    ## (batch_size,beam_width,units)
    dec_states = tf.concat([tf.expand_dims(dec_state,1) for _ in range(beam_width)],axis=1)
    ## (batch_size,beam_width,len(features))
    static_features = tf.concat([tf.expand_dims(static_features,1) for _ in range(beam_width)],axis=1)
    static_features = tf.reshape(static_features,(batch_size*beam_width,-1))

    ## (batch_size,beam_width,1)
    costs = tf.cast(tf.zeros_like(dec_inputs),dtype=tf.float64)
    tokens = None
    logits = None

    for step in range(max_length):

        ## 传输到deocder的参数size为(batch_size*beam_width,-1)，每个batch的所有beam进行同时计算
        dec_inputs = tf.reshape(dec_inputs,(batch_size*beam_width,-1))
        dec_states = tf.reshape(dec_states,(batch_size*beam_width,-1))

        ## 上述特征进行decoder,输出shape (batch_size*beam_width,-1)
        predictions, states = decoder(dec_inputs, dec_states, static_features,predict=predict)
        ## 将结果重新reshape为（batch_size,beam_width,-1)
        predictions = tf.reshape(predictions,(batch_size,beam_width,-1))

        probs = tf.nn.log_softmax(predictions)
        ## 将cost与probs相加 (batch_size,beam_width) + (batch_size,beam_with,len(vocab))
        costs = costs+probs

        # print(tf.sort(costs,axis=-1)[:,:,-beam_width:][0][0])

        ## 如果是第一步，每个beam列的结果应该相同，应该只使用一个作为结果
        if step==0:

            states = tf.reshape(states,(batch_size,beam_width,-1))
            ## 第一步输入相同，输出states也相同
            dec_states = states
            ## 概步的结果是一个维度的beam_size的输出,（batch_size,beam_width,1)
            dec_inputs = tf.expand_dims(tf.argsort(costs,axis=-1)[:,0,-beam_width:],-1)

            # print('step 0,dec_input',dec_inputs.shape)
            # print('step 0 probs',probs.shape)

            ## 将结果保存到tokens以及logits,(batch_size,beam_with,1)
            # tokens[:,:,step].assign(dec_input)
            tokens = dec_inputs
            ## beam_size的id对应的logits是相同的，也就是probs,只需要将prob变形为(batch_size,beam_size,1,-1)
            logits = tf.expand_dims(predictions,-2)

        else:

            ## 将tokens beam_width同样扩展beam_width倍, batch size, beam width, beam width, max_length
            last_tokens = tf.concat([tf.expand_dims(tokens,-2) for _ in range(beam_width)],axis=-2)
            last_logits = tf.concat([tf.expand_dims(logits,-3) for _ in range(beam_width)],axis=-3) # (batch_size,beam_width,beam_width,max_length,vocab_size)
            ## states同样需要 (batch_size*beam_width,beam_width,units)
            new_states = tf.concat([tf.expand_dims(states,-2) for _ in range(beam_width)],axis=-2)
            ## 每个结果对应的beam size
            new_logits = tf.concat([tf.expand_dims(predictions,-2) for _ in range(beam_width)],axis=-2)
            # new_states = np.reshape(new_states,(batch_size,beam_width*beam_width,-1))

            ## 根据概率获得每一个beam的最大的概率(batch_size,beam_width*beam_width)
            new_predited_ids = tf.reshape(tf.argsort(costs,axis=-1)[:,:,-beam_width:],(batch_size,-1))
            ## (batch_size,beam_width*beam_width)
            new_predited_probs =tf.reshape(tf.sort(costs,axis=-1)[:,:,-beam_width:],(batch_size,-1))
            ## 得到每一个样本最好的beam_width (batch_size,beam_width)的预测结果对应的位置,每一行对应beam_width个最优的位置，在beam_width*beam_width的长度内
            gather_indices = tf.argsort(new_predited_probs,axis=-1)[:,-beam_width:]

            ## 这里更新costs，也就是累计概率
            costs = tf.expand_dims(tf.sort(new_predited_probs,axis=1)[:,-beam_width:],-1)

            range_size = beam_width*beam_width
            dec_inputs = _tensor_gather_helper(gather_indices,new_predited_ids,batch_size,range_size,[-1],[batch_size,beam_width,1])
            dec_states = _tensor_gather_helper(gather_indices,new_states,batch_size,range_size,[batch_size*range_size,-1],[batch_size,beam_width,-1])

            ## 获得对应的token以及logits
            tokens =  _tensor_gather_helper(gather_indices,last_tokens,batch_size,range_size,[batch_size*range_size,-1],[batch_size,beam_width,-1])
            logits =  _tensor_gather_helper(gather_indices,last_logits,batch_size,range_size,[batch_size*range_size,-1,vocab_size],[batch_size,beam_width,-1,vocab_size])

            present_logits = _tensor_gather_helper(gather_indices,new_logits,batch_size,range_size,[batch_size*range_size,-1],[batch_size,beam_width,1,-1])

            # print('last token shape:',tokens.shape)
            # print('dec_input:',dec_inputs.shape)

            # print('last_logits',logits.shape)
            # print('present logits',present_logits.shape)

            tokens = tf.concat([tf.cast(tokens,tf.int32),tf.cast(dec_inputs,tf.int32)],axis=-1)
            logits = tf.concat([logits,present_logits],axis=-2)

    ## 根据costs的值进行排序,(batch_size,1)
    gather_indices = tf.argsort(costs,axis=1)[:,-1]
    last_costs = tf.sort(costs,axis=1)[:,-1]

    # print(gather_indices)

    tokens = _tensor_gather_helper(gather_indices,tokens,batch_size,beam_width,[batch_size*beam_width,max_length],[batch_size,max_length])
    logits = _tensor_gather_helper(gather_indices,logits,batch_size,beam_width,[batch_size*beam_width,max_length,-1],[batch_size,max_length,-1])

    return tokens,logits,last_costs

def greedy_search(decoder,dec_input,dec_state, static_features,length):

    all_predictions = []
    logits = []
    for _ in range(length):

        predictions,dec_state = decoder(dec_input,dec_state,static_features)

        dec_input = tf.argmax(predictions)

        all_predictions.append(tf.expand_dims(dec_input,1))

        logits.append(tf.softmax(predictions))

    return tf.concat(all_predictions,axis=1),tf.concat(logits,axis=1)

## gather helper from tensorflow
def _tensor_gather_helper(gather_indices,gather_from,batch_size,range_size,gather_shape,output_shape):
    ## 将indeices以及gather_from 需要取值的维度转化为一维

    ## 首先根据batch_size，range size将多维的序列转化为一维的序列
    range_ = tf.expand_dims(tf.range(batch_size) * range_size, 1)
    gather_indices = tf.reshape(gather_indices + range_, [-1])

    ## 根据转化为一维的序列，在gather_from中取值，gather_shape的第一维的长度应该是batch_size*range_size

    ## 如果维度是二维直接转化为一维向量，如果是三维或者多维需要将前两位转化为batch_size*range_size
    output = tf.gather(tf.reshape(gather_from, gather_shape), gather_indices)

    ## 最终获取的
    output = tf.reshape(output,output_shape)

    return output


