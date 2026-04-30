import cv2
import insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

swapper = insightface.model_zoo.get_model(
    "https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx"
)

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
