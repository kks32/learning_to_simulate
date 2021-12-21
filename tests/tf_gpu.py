import tensorflow as tf


from tensorflow.python.client import device_lib
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print(f"get_available_devices() {get_available_devices()}")

print(f"tf.config.experimental.list_physical_devices() = {tf.config.experimental.list_physical_devices()}")

print(f"tf.test.is_built_with_cuda() = {tf.test.is_built_with_cuda()}")

print(f"tf.test.is_built_with_gpu_support() = {tf.test.is_built_with_gpu_support()}")

isgpu = tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)
print(f"is_gpu_available = {isgpu}")
