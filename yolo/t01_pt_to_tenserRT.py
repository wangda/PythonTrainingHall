# 模型量化示例
from ultralytics import YOLO

# Load the YOLO11 model
# model = YOLO("best.pt")

# Export the model to TensorRT format
# model.export(format="engine", half=True, device="0")  # creates 'yolo11n.engine'

# Load the exported TensorRT model
tensorrt_model = YOLO("best.engine")

# Run inference
results = tensorrt_model("hxq_gjbs_14.jpg")

# Save the inference results to the current directory
results[0].save()

print(results)
