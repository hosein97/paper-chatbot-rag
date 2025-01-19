import gradio as gr
import requests
from typing import List, Tuple

# Base URL of your FastAPI app
BASE_URL = "http://127.0.0.1:8000"

# Global variables
class ChatState:
    def __init__(self):
        self.uploaded_filename = None
        self.chat_history: List[Tuple[str, str]] = []  # Changed to list of tuples
        self.is_processing = False

chat_state = ChatState()

def upload_pdf(file):
    """Handle PDF file upload"""
    if file is None:
        return (
            chat_state.chat_history,
            "⚠️ No file uploaded. Please upload a PDF.",
            gr.update(interactive=False),
            None
        )
    
    try:
        with open(file.name, "rb") as f:
            response = requests.post(
                f"{BASE_URL}/upload",
                files={"file": f}
            )
            
        if response.status_code == 200:
            chat_state.uploaded_filename = response.json().get("filename")
            chat_state.chat_history = []  # Reset chat history
            return (
                [],  # Clear chat
                "✅ File uploaded successfully! You can now start chatting.",
                gr.update(interactive=True),
                None
            )
        else:
            error_message = response.json().get("detail", "Error occurred during file upload.")
            return (
                chat_state.chat_history,
                f"❌ Error: {error_message}",
                gr.update(interactive=False),
                None
            )
    except Exception as e:
        return (
            chat_state.chat_history,
            f"❌ Error: {str(e)}",
            gr.update(interactive=False),
            None
        )

def chat_with_pdf(message: str, history: List[Tuple[str, str]]):
    """Handle chat interaction"""
    if chat_state.is_processing:
        return
    
    chat_state.is_processing = True
    
    if not chat_state.uploaded_filename:
        yield [], "⚠️ Please upload a file first."
        chat_state.is_processing = False
        return
    
    if not message.strip():
        yield [], "⚠️ Please enter a valid question."
        chat_state.is_processing = False
        return

    # Add user message to chat history
    chat_state.chat_history = history + [(message, None)]
    yield chat_state.chat_history, ""

    try:
        # Send request to backend
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"filename": chat_state.uploaded_filename, "question": message}
        )

        if response.status_code == 200:
            answer = response.json().get("answer", "No response.")
            
            # Update the last message with the complete response
            chat_state.chat_history = history + [(message, answer)]
            yield chat_state.chat_history, ""
                
        else:
            error_message = response.json().get("detail", "Error occurred during chat.")
            chat_state.chat_history = history + [(message, f"❌ Error: {error_message}")]
            yield chat_state.chat_history, f"Error: {error_message}"
            
    except Exception as e:
        chat_state.chat_history = history + [(message, f"❌ Error: {str(e)}")]
        yield chat_state.chat_history, f"Error: {str(e)}"
    
    chat_state.is_processing = False

def clear_chat():
    """Clear chat history"""
    chat_state.chat_history = []
    return [], "Chat cleared."

css = """
.contain { display: flex; flex-direction: column; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; overflow-y: auto; }
#chatbot .message { padding: 1rem; margin: 0.5rem; border-radius: 0.5rem; }
#chatbot .user { background-color: #f0f0f0; }
#chatbot .bot { background-color: #e6f3ff; }
.upload-box { padding: 1rem; border: 2px dashed #ccc; border-radius: 0.5rem; }
.footer { text-align: center; padding: 1rem; }
"""

def create_gradio_app():
    """Create and configure the Gradio interface"""
    with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
        # Header
        with gr.Row():
            gr.Markdown(
                """
                # 📚 Chat with Your PDF
                Upload a PDF file and start asking questions about its content.
                """
            )
        
        # Main chat interface
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    bubble_full_width=False,
                    avatar_images=("🧑", "🤖"),
                    height=600
                )
                
                # Input area
                with gr.Row():
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder="Type your question here...",
                        container=False
                    )
                    send_btn = gr.Button("Send", variant="primary")
        
                # Status message
                status_msg = gr.Markdown("")
        
            # Sidebar
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 📄 Upload PDF")
                    file_upload = gr.File(
                        label="",
                        file_types=[".pdf"],
                        interactive=True
                    )
                    clear_btn = gr.Button("🗑️ Clear Chat")
        
        # Event handlers
        send_btn.click(
            chat_with_pdf,
            [txt, chatbot],
            [chatbot, status_msg],
            api_name="chat"
        ).then(
            lambda: gr.update(value=""),
            None,
            [txt]
        )
        
        txt.submit(
            chat_with_pdf,
            [txt, chatbot],
            [chatbot, status_msg],
            api_name="chat"
        ).then(
            lambda: gr.update(value=""),
            None,
            [txt]
        )
        
        file_upload.upload(
            upload_pdf,
            inputs=[file_upload],
            outputs=[chatbot, status_msg, txt, file_upload],
            api_name="upload"
        )
        
        clear_btn.click(
            clear_chat,
            None,
            [chatbot, status_msg],
            api_name="clear"
        )
        
        # Footer
        gr.Markdown(
            """
            <div class="footer">
                <p>Made with ❤️ using Gradio</p>
            </div>
            """,
            elem_classes=["footer"]
        )
    
    return demo

if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )