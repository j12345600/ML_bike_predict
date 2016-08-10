import os
import csv
import sys
import numpy as np
import mxnet as mx
import logging
import utilities as utl
Path_base="./dataset/61/"
Path_train_label=Path_base + "train_label.csv"
Path_train=Path_base + "train.csv"
Path_test_label=Path_base + "val_label.csv"
Path_test=Path_base + "val.csv"

def get_mlp():
    data=mx.sym.Variable('data')
    fc1=mx.sym.FullyConnected(data, name="fc1", num_hidden=6)
    act1=mx.sym.Activation(fc1, name="relu1", act_type="relu")
    fc2=mx.sym.FullyConnected(act1, name="fc2", num_hidden=36)
    act2=mx.sym.Activation(fc2, name="relu2", act_type="relu")
    fc3=mx.sym.FullyConnected(act2, name="fc3", num_hidden=6)
    # act3=mx.sym.Activation(fc3, name="relu3", act_type="relu")
    # fc4=mx.sym.FullyConnected(act3, name="fc4", num_hidden=50)
    mlp=mx.sym.SoftmaxOutput(fc3, name="softmax")
    # mlp.infer_shape(out_shapes=[(1,1)])
    arg,out,aux=mlp.infer_shape(data=(1,6),softmax_label=(1,))
    print(arg)
    return mlp

if __name__=="__main__":
    num_epoch=5
    batch_size=1
    #val
    val_ndary=utl.csv2ndary(Path_test)
    val_label_ndary=utl.csv2ndary(Path_test_label)
    #train
    train_ndary=utl.csv2ndary(Path_train)
    train_label_ndary=utl.csv2ndary(Path_train_label)

    print(train_label_ndary.shape)
    print(train_ndary.shape)

    # train_dataiter=mx.io.NDArrayIter(
    #                     train_ndary,
    #                     label=train_label_ndary,
    #                     batch_size=batch_size,
    #                     last_batch_handle='pad')
    # val_dataiter=mx.io.NDArrayIter(
    #                 val_ndary,
    #                 label=val_label_ndary,
    #                 batch_size=batch_size,
    #                 last_batch_handle='pad')
    train_dataiter=mx.io.CSVIter(
            data_csv=Path_train,
            data_shape=(6),
            label_csv=Path_train_label,
            label_shape=(1,),
            batch_size=1)
    val_dataiter=mx.io.CSVIter(
            data_csv=Path_test,
            data_shape=(6),
            label_csv=Path_test_label,
            label_shape=(1,),
            batch_size=1)
    mlp=get_mlp()
    print(train_dataiter.getdata().asnumpy())
    print("#########\nNdarray info: {}\n\n".format(train_dataiter.provide_data))
    print("#########\nNdarray info: {}\n\n".format(train_dataiter.provide_label))
    input('')
    #logging
    kv = mx.kvstore.create('local')
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments')
    save_model_prefix = "save_model"
    if save_model_prefix is None:
        save_model_prefix = model_prefix
    checkpoint = None if save_model_prefix is None else mx.callback.do_checkpoint(save_model_prefix)
    #building a model
    model = mx.model.FeedForward(
             get_mlp(),
             ctx=mx.gpu(0),
             num_epoch=num_epoch,
             learning_rate=0.1,
             momentum           = 0.9,
             wd                 = 0.00001,
             initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34),
             numpy_batch_size   = batch_size)
    #setting fit parameters
    eval_metrics = ['accuracy']
    ## TopKAccuracy only allows top_k > 1
    for top_k in [5, 10, 20]:
        eval_metrics.append(mx.metric.create('top_k_accuracy', top_k = top_k))
    batch_end_callback=None
    if batch_end_callback is not None:
        if not isinstance(batch_end_callback, list):
            batch_end_callback = [batch_end_callback]
    else:
        batch_end_callback = []
    batch_end_callback.append(mx.callback.Speedometer(batch_size, 50))

    #start training
    model.fit(
                X                  = train_dataiter,
                eval_data          = val_dataiter,
                eval_metric        = eval_metrics,
                kvstore            = kv,
                batch_end_callback = batch_end_callback,
                #epoch_end_callback = checkpoint
                )

    #save model
    prefix = 'mymodel'
    iteration = 100
    model.save(prefix, iteration)

    #use the model just trained to predict
    test_ndarray=utl.csv2ndary("./dataset/61/testset61_label.csv")
    label_ndarray=utl.csv2ndary("./dataset/61/testset61_trim.csv")
    # val=mx.io.NDArrayIter(
    #         test_ndarray,
    #         label=label_ndarray,
    #         batch_size=1)
    val=mx.io.CSVIter(
            data_csv="./dataset/61/testset61_label.csv",
            data_shape=(1,6),
            label_csv="./dataset/61/testset61_trim.csv",
            label_shape=(1,1),
            batch_size=batch_size)
    [prob, data1, label1] = model.predict(val,num_batch=batch_size, return_data=True)
    print(prob)
    print(data1)
    print(label1)
    # result=model.predict(X = val_dataiter,return_data = True)
    # for x in result:
    #     print(x.astype(int))
    # print(type(result[0]))
    # with open('predict.txt','w',newline='') as f:
    #     f.write()
    # mod=mx.mod.Module(mlp)
    # #bind and init_params
    # mod.bind(data_shapes=train_dataiter.provide_data, label_shapes=train_dataiter.provide_label)
    # mod.init_params()
    #
    # mod.init_optimizer(optimizer_params={'learning_rate':0.01, 'momentum': 0.9})
    # metric = mx.metric.create('acc')
    #
    # for i_epoch in range(num_epoch):
    #     for i_iter, batch in enumerate(train_dataiter):
    #         mod.forward(batch)
    #         mod.update_metric(metric, batch.label)
    #
    #         mod.backward()
    #         mod.update()
    #
    #     for name, val in metric.get_name_value():
    #         print('epoch %03d: %s=%f' % (i_epoch, name, val))
    #     metric.reset()
    #     train_dataiter.reset()
