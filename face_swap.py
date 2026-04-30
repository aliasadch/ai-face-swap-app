import cv2
import insightface
import os

# Make sure model file exists
assert os.path.exists("inswapper_128.onnx"), "inswapper_128.onnx not found!"

app = insightface.app.FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)  # will use CPU if GPU unavailable

# ✅ LOAD LOCAL FILE ONLY
swapper = insightface.model_zoo.get_model("inswapper_128.onnx")

def swap_faces(source_path, target_path):
    source = cv2.imread(source_path)
    target = cv2.imread(target_path)

    source_faces = app.get(source)
    target_faces = app.get(target)

    if len(source_faces) == 0 or len(target_faces) == 0:
        return None

    result = swapper.get(
        target,
        target_faces[0],
        source_faces[0],
        paste_back=True
    )

    return result
