import cv2
import insightface

app = insightface.app.FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

swapper = insightface.model_zoo.get_model("inswapper_128.onnx")

def swap_faces(source_path, target_path):
    source = cv2.imread(source_path)
    target = cv2.imread(target_path)

    source_faces = app.get(source)
    target_faces = app.get(target)

    if len(source_faces) == 0 or len(target_faces) == 0:
        return None

    target_face = target_faces[0]

    result = swapper.get(
        target,
        target_face,
        source_faces[0],
        paste_back=True
    )

    return result
