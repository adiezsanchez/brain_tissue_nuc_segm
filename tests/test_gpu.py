import tensorflow as tf
import time

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")

if not gpus:
    print("❌ No GPU detected by TensorFlow.")
else:
    print("✅ GPUs detected:")
    for i, gpu in enumerate(gpus):
        print(f"  ID {i}: {gpu}")

    # Explicitly set device to GPU:0
    print("\nRunning GPU stress test on /device:GPU:0 ...")
    with tf.device("/GPU:0"):
        for i in range(5):
            a = tf.random.normal([10000, 10000])
            b = tf.random.normal([10000, 10000])
            c = tf.matmul(a, b)
            print(f"Iteration {i+1}: result shape {c.shape}, device: {c.device}")
            time.sleep(1)  # short pause to make GPU activity visible in nvidia-smi
