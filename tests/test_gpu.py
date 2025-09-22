import time
import ctypes.util
import tensorflow as tf
import pyopencl as cl


def check_opencl_runtime():
    """Check if OpenCL ICD loader and vendor runtimes are available."""
    # Step 1: check for loader
    loader = ctypes.util.find_library("OpenCL")
    if loader is None:
        raise RuntimeError(
            "OpenCL ICD loader (libOpenCL.so) not found.\n"
            "👉 Install it on Ubuntu with:\n"
            "   sudo apt install ocl-icd-opencl-dev clinfo\n"
            "Or install `ocl-icd-system` in your Pixi/conda env."
        )

    # Step 2: check for platforms
    try:
        platforms = cl.get_platforms()
    except cl._cl.LogicError as e:
        raise RuntimeError(
            "OpenCL loader found, but no vendor ICD runtime detected.\n"
            "For NVIDIA GPUs, please install the NVIDIA driver with OpenCL support "
            "(e.g. `libnvidia-compute-<version>` on Ubuntu).\n"
            "👉 To fix inside Pixi, run:\n"
            "   pixi run fix_opencl"
        ) from e

    if not platforms:
        raise RuntimeError(
            "No OpenCL platforms detected. "
            "This usually means the NVIDIA OpenCL ICD (libnvidia-opencl.so) is missing."
        )

    return platforms


def run_tensorflow_gpu_test():
    """Run a simple GPU stress test with TensorFlow."""
    print("\nTensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        print("❌ No GPU detected by TensorFlow.")
        return

    print("\n✅ GPUs detected:")
    for i, gpu in enumerate(gpus):
        print(f"  ID {i}: {gpu}")

    print("\nRunning GPU stress test on /device:GPU:0 ...")
    with tf.device("/GPU:0"):
        for i in range(5):
            a = tf.random.normal([10000, 10000])
            b = tf.random.normal([10000, 10000])
            c = tf.matmul(a, b)
            print(f"Iteration {i+1}: result shape {c.shape}, device: {c.device}")
            time.sleep(1)  # pause so you can see activity in nvidia-smi


if __name__ == "__main__":
    # Step 1: Check OpenCL
    try:
        plats = check_opencl_runtime()
        print("✅ OpenCL platforms available:")
        for p in plats:
            print(" -", p.name)
    except RuntimeError as err:
        print("❌", err)

    # Step 2: Run TensorFlow GPU test
    run_tensorflow_gpu_test()

