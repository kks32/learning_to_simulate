# Import modules and this file should be outside learning_to_simulate code folder
import functools
import os
import json
import pickle

import tensorflow.compat.v1 as tf
import numpy as np

from learning_to_simulate import reading_utils

# Set datapath and validation set
data_path = './datasets/WaterRamps'
filename = 'valid.tfrecord'

# Read metadata
def _read_metadata(data_path):
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        return json.loads(fp.read())

# Fetch metadata
metadata = _read_metadata(data_path)

print(metadata)

# Read TFRecord
ds_org = tf.data.TFRecordDataset([os.path.join(data_path, filename)])
ds = ds_org.map(functools.partial(reading_utils.parse_serialized_simulation_example, metadata=metadata))

# Convert to list
lds = list(ds)

particle_types = []
keys = []
positions = []
for _ds in ds:
    context, features = _ds
    particle_types.append(context["particle_type"].numpy().astype(np.int64))
    keys.append(context["key"].numpy().astype(np.int64))
    positions.append(features["position"].numpy().astype(np.float32))

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Write TF Record
with tf.python_io.TFRecordWriter('test.tfrecord') as writer:
    
    for step, (particle_type, key, position) in enumerate(zip(particle_types, keys, positions)):
        seq = tf.train.SequenceExample(
                context=tf.train.Features(feature={
                    "particle_type": _bytes_feature(particle_type.tobytes()),
                    "key": _int64_feature(key)
                }),
                feature_lists=tf.train.FeatureLists(feature_list={
                    'position': tf.train.FeatureList(
                        feature=[_bytes_feature(position.flatten().tobytes())],
                    ),
                    'step_context': tf.train.FeatureList(
                        feature=[_bytes_feature(np.float32(step).tobytes())]
                    ),
                })
            )

        writer.write(seq.SerializeToString())


dt = tf.data.TFRecordDataset(['test.tfrecord'])
dt = dt.map(functools.partial(reading_utils.parse_serialized_simulation_example, metadata=metadata))


for ((_ds_context, _ds_feature), (_dt_context, _dt_feature)) in zip(ds, dt):
    if not np.allclose(_ds_context["key"].numpy(), _dt_context["key"].numpy()):
        break

    if not np.allclose(_ds_context["particle_type"].numpy(), _dt_context["particle_type"].numpy()):
        break

    if not np.allclose(_ds_feature["position"].numpy(), _dt_feature["position"].numpy()):
        break

else:
    print("TFRecords are similar!")
