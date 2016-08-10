import mxnet as mx
import logging
import numpy as np
import utilities as utl

Path_test_label="./dataset/61/testset61_label.csv"
Path_test="./dataset/61/testset61_trim.csv"
prefix = './mymodel'
iteration = 100
model_load = mx.model.FeedForward.load(prefix, iteration,ctx=mx.gpu(0))
data_shape = (3, 224, 224)


test_ndarray=utl.csv2ndary(Path_test)
label_ndarray=utl.csv2ndary(Path_test_label)
# 数据准备  batch_size = 1.
val=mx.io.NDArrayIter(
        test_ndarray,
        label=label_ndarray,
        batch_size=1)

val_dataiter=mx.io.CSVIter(
        data_csv=Path_test,
        data_shape=(6),
        label_csv=Path_test_label,
        label_shape=(1,),
        batch_size=1)
# val = mx.io.ImageRecordIter(
#         path_imgrec = '/xxx/xxx/' + "xxx.rec",
#         mean_img    = '/xxx/xxx/' + "xxx.bin",
#         rand_crop   = False,
#         rand_mirror = False,
#         data_shape  = data_shape,
#         batch_size  = 1)
print(model_load)
[prob, data1, label1] = model_load.predict(val_dataiter,num_batch=1, return_data=True)
print(prob)
print(data1)
print(label1)
