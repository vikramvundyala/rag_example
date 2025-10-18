import gradio as gr
from rag_implementation import RAGSystem
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the RAG system
def init_rag_system():
    rag = RAGSystem()

    # Load and process documents
    documents = rag.load_documents("./documents", "pdf")
    if not documents:
        raise ValueError("No documents found in the 'documents' directory")

    processed_docs = rag.process_documents(documents)
    rag.create_vector_store(processed_docs)
    rag.set_llm()
    rag.setup_rag_chain()
    return rag

# Initialize the RAG system
rag_system = init_rag_system()

def get_response(question, history):
    try:
        result = rag_system.query(question)
        response = f"**Answer:** {result['answer']}\n\n"

        if result['sources']:
            response += "**Sources:**\n"
            for i, source in enumerate(result['sources'], 1):
                source_name = source['metadata'].get('source', 'Unknown')
                response += f"\n**Source {i}** ({source_name}):\n"
                response += f"{source['content'][:300]}...\n"

        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Document Q&A System
    Ask questions about your documents and get answers based on the provided context.
    """)

    with gr.Row():
        chatbot = gr.Chatbot(height=500)

    with gr.Row():
        msg = gr.Textbox(
            label="Type your question",
            placeholder="Ask me anything about the documents...",
            scale=8
        )
        clear = gr.Button("Clear")

    def respond(message, chat_history):
        bot_message = get_response(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7861)