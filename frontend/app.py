import gradio as gr
import requests

API_URL = "http://localhost:8000"  # Backend API URL

def upload_file_and_chat(file, question):
    if file is not None:
        # Upload the PDF file to the backend
        files = {'file': (file.name, file)}
        upload_response = requests.post(f"{API_URL}/upload", files=files)

        if upload_response.status_code == 200:
            chat_response = requests.post(
                f"{API_URL}/chat",
                json={"question": question, "file_id": upload_response.json()['file_id']}
            )
            if chat_response.status_code == 200:
                return chat_response.json().get('answer', 'No response from GPT')
    return "Please upload a valid file and ask a question."

# Create the Gradio interface
file_input = gr.File(label="Upload Paper (PDF only)")
question_input = gr.Textbox(label="Ask a Question")
output = gr.Textbox(label="Answer")

gr.Interface(
    fn=upload_file_and_chat,
    inputs=[file_input, question_input],
    outputs=output,
    title="Chat with Your Paper",
    description="Upload a PDF paper and ask GPT questions about it.",
).launch()
