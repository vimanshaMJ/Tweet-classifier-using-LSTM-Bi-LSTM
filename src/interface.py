import gradio as gr
from predict import PHMPredictor

def create_interface():
    lstm_predictor = PHMPredictor('lstm')
    bilstm_predictor = PHMPredictor('bilstm')
    
    def analyze(text):
        lstm_result = lstm_predictor.predict(text)
        bilstm_result = bilstm_predictor.predict(text)
    
        def format_output(result):
            label = "Personal Health Mention" if result['prediction'] else "Not a Personal Health Mention"
            emoji = "✅" if result['prediction'] else "❌"
            return {
                "Label": label,
                "Confidence": f"{result['confidence']*100:.2f}%",
                "Emoji": emoji
            }
    
        return {
            "Input": text,
            "LSTM Prediction": format_output(lstm_result),
            "Bi-LSTM Prediction": format_output(bilstm_result)
        }

    
    return gr.Interface(
        fn=analyze,
        inputs=gr.Textbox(label="Enter Tweet"),
        outputs=gr.JSON(label="Model Predictions"),
        title="Personal Health Mention Classifier",
        description="Compare LSTM vs Bi-LSTM predictions for health mentions in tweets"
    )

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()
