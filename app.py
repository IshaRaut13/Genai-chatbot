import os
import gradio as gr
from dotenv import load_dotenv  # Load environment variables
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Load environment variables from a .env file
load_dotenv()

# Get the OpenAI API key securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY. Please set it in your environment variables.")

# Define the prompt template
template = """As an adventurous and globetrotting college student, you're constantly on the lookout for new cultures, experiences, and breathtaking landscapes. You've visited numerous countries, immersing yourself in local traditions, and you're always eager to swap travel stories and offer tips on exciting destinations.
{chat_history}
User: {user_message}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "user_message"], template=template
)

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize the LLMChain with OpenAI API key
llm_chain = LLMChain(
    llm=ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY),
    prompt=prompt,
    verbose=True,
    memory=memory,
)

# Function to process chat messages
def get_text_response(user_message, history):
    response = llm_chain.predict(user_message=user_message)
    return response

# Initialize Gradio Chat Interface
demo = gr.ChatInterface(get_text_response)

if __name__ == "__main__":
    demo.launch(share=True)  # Use `share=True` to make it publicly accessible
