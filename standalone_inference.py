import tensorflow as tf
import tensorflow_datasets as tfds
import os
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--load_folder', type=str, default='./SavedModel')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--dataset_name', type=str, default='cifar100')
p = parser.parse_args()

imported = tf.saved_model.load(p.load_folder)

if p.dataset_name.lower() == 'cifar10':
    mean = tf.constant([[[0.4914, 0.4822, 0.4465]]], dtype=tf.float32)
    stddev = tf.constant([[[0.2023, 0.1994, 0.2010]]], dtype=tf.float32)        
    input_size = [p.batch_size, 32, 32, 3]
    num_labels = 10
elif p.dataset_name.lower() == 'cifar100':
    mean = tf.constant([[[125.3/255., 123.0/255., 113.9/255.]]], dtype=tf.float32)
    stddev = tf.constant([[[63.0/255., 62.1/255., 66.7/255.]]], dtype=tf.float32)        
    input_size = [p.batch_size, 32, 32, 3]
    num_labels = 100
elif p.dataset_name.lower() == 'mnist':
    mean = tf.constant([0.1307354092654539], dtype=tf.float32)
    stddev = tf.constant([0.30819553701697117], dtype=tf.float32)
    input_size = [p.batch_size, 28, 28, 1]
    num_labels = 10
else:
    assert False, 'this dataset is not supported'

data, _ = tfds.load(p.dataset_name, with_info=True)
def _parsefunc(D):
    D['image'] = tf.image.convert_image_dtype(D['image'], dtype=tf.float32)        
    D['label'] = tf.cast(tf.one_hot(D['label'], num_labels), tf.float32)
    return D
def subtract_mean(x):
    y = {'image': x['image'] - mean, 'label':x['label']}
    if 'coarse_label' in x.keys():
        y['coarse_label'] = x['coarse_label']
    return y
def normalize_stddev(x):
    y = {'image': x['image'] / stddev , 'label':x['label']}
    if 'coarse_label' in x.keys():
        y['coarse_label'] = x['coarse_label']
    return y    
test_set = data['test']
test_set = test_set.map(_parsefunc, num_parallel_calls=2)
test_set = test_set.map(subtract_mean, num_parallel_calls=2)
test_set = test_set.map(normalize_stddev, num_parallel_calls=2)
test_set = test_set.batch(p.batch_size, drop_remainder=True)

acc = tf.keras.metrics.Accuracy()

for x in test_set:
    image,label = x['image'],x['label']

    logits = imported(image)

    #Changed this to account for possibility of mixup-ed labels
    acc.update_state(tf.argmax(label,axis=1),tf.argmax(logits,axis=1))

print(acc.result().numpy())