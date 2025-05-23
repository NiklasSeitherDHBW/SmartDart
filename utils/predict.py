import ultralytics

class Predictor:
    def __init__(self, model_path="yolo8n.pt"):
        self.model = ultralytics.YOLO(model_path)

    def predict(self, frame):
        results = self.model.predict(frame)
        return results
