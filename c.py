import os
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# ========================================
# 1. KET NOI ROBOFLOW
# ========================================

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"]
).configure(
    InferenceConfiguration(
        api_key_transport="header"
    )
)

# ========================================
# 2. ANH CAN KIEM TRA
# ========================================

IMAGE_PATH = "led.jpg"

# ========================================
# 3. GOI WORKFLOW
# ========================================

result = client.run_workflow(
    workspace_name="yolodongphuc",
    workflow_id="general-segmentation-api",
    images={
        "image": IMAGE_PATH
    },
    parameters={
        "classes": "LED"
    },
    use_cache=True
)

# ========================================
# 4. HIEN THI KET QUA
# ========================================

print("KET QUA NHAN DIEN:")
print(result)