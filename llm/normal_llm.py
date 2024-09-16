from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Initialize the model on GPU if available
llm = Ollama(model="llama2")

chat_history = []

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI named Mike, you answer questions with simple answers and no funny stuff.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

chain = prompt_template | llm

def start_app(question):
    # Run inference
    response = chain.invoke({"input": question, "chat_history": chat_history})
    
    # Update chat history
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))

    return response

if __name__ == "__main__":
    print(start_app("give 4 or 5 sentence of reply for cheetah"))
