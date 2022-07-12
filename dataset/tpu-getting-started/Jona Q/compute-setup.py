def setup_compute():
    import tensorflow as tf
    # Check for TPU
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print(f'TPU detected: {tpu.master()}')
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()

    # If no TPU detected, check for GPU
    if tpu is None:
        device_name = tf.test.gpu_device_name()
        if 'GPU' in device_name:
            print(f'GPU detected: {device_name}')
        else:
            print(f'Using CPU')
    print(f'Replicas: {strategy.num_replicas_in_sync}')