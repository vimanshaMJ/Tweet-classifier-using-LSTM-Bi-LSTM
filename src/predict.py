from tensorflow.keras.models import load_model
from utils import prepare_input

class PHMPredictor:
    def __init__(self, model_type='lstm'):
        self.model = load_model(f'models/{model_type}_model.h5')
        self.max_length = 30
        
    def predict(self, text):
        processed_input = prepare_input(text, self.max_length)
        prediction = self.model.predict(processed_input)[0][0]
        return {
            'prediction': int(prediction > 0.5),
            'confidence': float(prediction),
            'model': 'Bi-LSTM' if 'bilstm' in self.model.name.lower() else 'LSTM'
        }
    
    def print_result(result, model_name):
        label = "Personal Health Mention" if result['prediction'] else "Not a Personal Health Mention"
        emoji = "✅" if result['prediction'] else "❌"
        print(f"{model_name}: {label} {emoji} (Confidence: {result['confidence']*100:.2f}%)")

