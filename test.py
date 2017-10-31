import caffe
import lmdb

lmdb_env = lmdb.open('data/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    label = datum.label
    data = caffe.io.datum_to_array(datum)
    n += 1
    print key, label

print n
