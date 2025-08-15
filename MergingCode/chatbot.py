from typing import TypedDict
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

class QueryHandler(BaseModel):
    query: str = Field(..., description='The query after language detect, will be in Japanese')

class LanguageDetector:
    def __init__(self, model):
        self.model = model 
        detector_prompt = """
        You are a language detection and transformation model. 
        Your task:
        1. Detect the language of the input text.
        2. If the language is Japanese → Keep the text exactly as it is.
        3. If the language is NOT Japanese → Translate it into Japanese, ensuring the translation is clear, explicit, and preserves all meaning.

        Your output MUST be a single, valid JSON object that conforms to the 'QueryHandler' Pydantic model.
        """

        self.system_message = SystemMessage(
            content=detector_prompt
        )

        self.model = self.model.with_structured_output(QueryHandler)

    def ask(self, query: str):
        return self.model.invoke([
            self.system_message,
            HumanMessage(content=query)
        ])
    
class RAGRetriever:
    def __init__(self, model, retriver):
        self.model = model
        self.retriever = retriver

        retriever_prompt = """
        You are an AI assistant that answers questions in Vietnamese using ONLY the retrieved information provided. 
        The retrieved information is your sole source of truth for answering the user's query in Japanese.

        Instructions:
        1. Read the retrieved information carefully and use it to answer the question as accurately and comprehensively as possible.
        2. Do NOT add information from outside the retrieved context.
        3. If multiple pieces of relevant information are found, combine them into a coherent and complete answer.
        4. If the retrieved information does not contain enough details to answer the question, respond with:
        "Xin lỗi, tôi không đủ thông tin." and the explanation of what information is missing or what make you can't answer.
        5. The final answer must be in clear, natural Vietnamese and easy to understand.
        """

        self.system_message = SystemMessage(content=retriever_prompt)

    def ask(self, query: str):
        retrieved_docs = self.retriever.get_relevant_documents(query)
        if not retrieved_docs:
            return "I'm sorry, but I do not have enough information to answer that. Retrieved_docs: {}".format(retrieved_docs)

        print(retrieved_docs)
        print(query)

        humanQuery = f"""
        Answer the following question based on the context provided:

        Context:
        {retrieved_docs}

        Question:
        {query}
        """


        print(humanQuery)

        return self.model.invoke([
            self.system_message.content.format(retrieved_docs=retrieved_docs, query=query),
            HumanMessage(content=humanQuery)
        ])
    
class GraphState(TypedDict):
    """Represents the state of the graph at a given point in time."""
    nodes: list[str]
    edges: list[tuple[str, str]]
    current_node: str
    user_query: str
    final_answer: str


from typing import Callable, Dict, List, Tuple, Optional
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

def visuallize_graph(graph):
    from IPython.display import Image, display

    try:
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception:
        pass

class Chatbot:
    def __init__(
        self,
        chat_model_id: Optional[str] = None,
        retriever = None,
        embedding_model_id: str = "amazon.titan-embed-text-v2:0",
    ):
        """
        chat_model_id: optional id for the chat model (if None, init_chat_model will pick default)
        retriever: your FAISS retriever instance (or None)
        embedding_model_id: required by you (set as attribute so you can wire to retriever)
        """
        self.embedding_model_id = embedding_model_id

        if chat_model_id:
            self.model = init_chat_model(model=chat_model_id)
        else:
            self.model = init_chat_model(
            "gemini-2.0-flash", 
            model_provider="google_genai", 
            google_api_key='AIzaSyApW1tlHyEXjBMItv6TN7V-8U9L7azedWI'
        )
        self.retriever = retriever

        self.lang_detector = LanguageDetector(self.model)

        self.rag_retriever = RAGRetriever(self.model, self.retriever)

        self.workflow = StateGraph(GraphState)
        self.create_workflow()
        self.app = self.workflow.compile()

        visuallize_graph(self.app)

    def _node_language_detect(self, state: GraphState) -> GraphState:
        # Detect the language of the user 
        user_query = state['user_query']
        print('User query:', user_query)
        user_query = self.lang_detector.ask(user_query)
        state['user_query'] = user_query.query
        return state

    def _node_final_answer(self, state: GraphState) -> GraphState:
        # Generate the final answer using the chat model
        user_query = state['user_query']
        print('User query:', user_query)
        final_answer = self.rag_retriever.ask(user_query)
        state['final_answer'] = final_answer
        return state

    def create_workflow(self):
        self.workflow.add_node("language_detect", self._node_language_detect)
        self.workflow.add_node("final_answer", self._node_final_answer)

        self.workflow.add_edge("language_detect", "final_answer")
        self.workflow.add_edge("final_answer", END)

        self.workflow.set_entry_point("language_detect")

    # ---------- Public ask API ----------
    def ask(self, user_query: str) -> str:
        """
        Run the workflow/app with the given user query and return the final_answer string.
        """
        inputs = {"user_query": user_query}
        result_state = self.app.invoke(inputs)
        return result_state.get("final_answer") or "I'm sorry, but something went wrong."

