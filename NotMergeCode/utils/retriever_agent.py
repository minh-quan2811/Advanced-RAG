import os
import json
import uuid
import logging
from dotenv import load_dotenv
from typing import List, TypedDict, Annotated, Any
import textwrap

from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, RunnableParallel
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI

from storage_handler import NodeStorageHandler
from retrieval_pipeline import QueryPipeline, QueryFilters


load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SearchParameters(BaseModel):
    """The structured parameters for querying the Qdrant vector store."""
    search_query: str = Field(description="An optimized, self-contained search query for semantic search.")
    filters: QueryFilters = Field(description="Metadata filters (categories and keywords) to apply to the search.")

PIPELINE: QueryPipeline = None

def get_available_filters_str() -> str:
    """Helper to return a valid JSON string of available filters."""
    # This would query to PostgreSQL database
    # SELECT DISTINCT "Category" FROM ...; SELECT DISTINCT "Keywords" FROM ...;
    valid_filters = {
        "categories": ["Customer Value Analysis", "Customer Segmentation", "Loyalty & Retention", "Branding & Positioning", "Retail Marketing", "Promotional Tactics", "Performance Measurement", "Case Study"],
        "keywords": ["Customer Lifetime Value (CLV)", "Churn Rate Analysis", "Retention Rate", "Customer Investment Management (CIM)", "Marketing ROI", "Cross-sell & Up-sell", "Decile Analysis", "Customer Profiling", "Targeted Promotion", "Sales Forecasting", "High-Value Customer Identification", "Marketing Framework (Big Picture)", "STP (Segmentation, Targeting, Positioning)", "Customer-centric Service", "Brand Storytelling", "Moment of Truth (真実の瞬間)", "Marketing Mix (4Ps)", "Relationship Marketing", "Lifestyle Segmentation", "Demand Creation (需要創造)", "Marketing Myopia (マーケティング近視眼)"]
    }

    return json.dumps(valid_filters)

# Tools
@tool
def plan_retrieval_tool(question: str, error_feedback: str = "", plan: str = "") -> str:
    """
    Generates or corrects a structured retrieval plan based on the user's question.
    The plan includes an optimized search query and any relevant metadata filters.
    """
    logger.info("Tool called: plan_retrieval_tool")

    if error_feedback:
        error_context = f"The previous attempt failed with this error: '{error_feedback}'. Please analyze the user's question again and generate a new, corrected plan."
    else:
        error_context = ""

    system_prompt_template = textwrap.dedent(f"""
        You are an expert at creating structured query plans for a vector database and finding the correct filters based on the query.
        Your task is to analyze the user's question and generate a JSON object that matches the required `SearchParameters` format.
        The JSON object must contain:
        1. A `search_query`: An optimized, self-contained query that captures the core semantic intent of the user's question. MUST be written in Japanese.
        2. A nested `filters` object, which contains `categories` and `keywords`.

        RULES:
        - Only use filter values from the provided 'Available' lists.
        - If no filters are relevant, the `categories` and `keywords` lists should be empty.
        - Values inside 'categories' MUST come ONLY from Available Categories.
        - Values inside 'keywords' MUST come ONLY from Available Keywords.
        - Do NOT invent new values.
        {error_context}

        --- Available Filters ---
        Categories: {{categories}}
        Keywords: {{keywords}}
        """)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template),
        ("human", "User Question: {question}")
    ])

    structured_llm = LLM.with_structured_output(SearchParameters)
    chain = prompt | structured_llm

    try:
        # Get the valid filters as separate lists
        valid_filters = json.loads(get_available_filters_str())

        # Invoke the chain
        search_params = chain.invoke({
            "question": question,
            "categories": valid_filters["categories"],
            "keywords": valid_filters["keywords"]
        })

        # The output `search_params` will be a Pydantic object that matches SearchParameters
        plan_json = search_params.model_dump_json(indent=2)
        logger.info(f"Retrieval plan generated:\n{plan_json}")
        return f"RETRIEVAL_PLAN_GENERATED: {plan_json}"

    except Exception as e:
        logger.error(f"Plan generation failed: {e}", exc_info=True)
        return f"ERROR: Failed to generate a retrieval plan. Reason: {e}"

@tool
def validate_plan_tool(plan: str) -> str:
    """
    Validates if the retrieval plan is well-formed and uses valid filters.

    Args:
        plan: The retrieval plan JSON string generated by plan_retrieval_tool.
    """
    logger.info("Tool called: validate_plan_tool")
    
    try:
        # Extract the actual JSON from the tool's output string
        plan_json_str = plan.replace("RETRIEVAL_PLAN_GENERATED:", "").strip()
        plan_data = json.loads(plan_json_str)
        
        # Validate the structure
        validated_plan = SearchParameters(**plan_data)
        
        # Validation for filter values
        valid_filters = json.loads(get_available_filters_str())
        for category in validated_plan.filters.categories:
            if category not in valid_filters["categories"]:
                error = f"INVALID: Category '{category}' is not a valid filter."
                logger.warning(error)
                return error
        for keyword in validated_plan.filters.keywords:
            if keyword not in valid_filters["keywords"]:
                error = f"INVALID: Keyword '{keyword}' is not a valid filter."
                logger.warning(error)
                return error
        
        logger.info("Plan validation successful.")
        return "VALIDATION_RESULT: VALID"
    except Exception as e:
        logger.error(f"Plan validation failed: {e}")
        return f"INVALID: Plan is not a valid JSON or has an incorrect structure. Reason: {e}"

@tool
def execute_retrieval_tool(plan: str) -> str:
    """
    Executes a validated retrieval plan using the Qdrant pipeline.
    """
    logger.info("Tool called: execute_retrieval_tool")

    try:
        plan_json_str = plan.replace("RETRIEVAL_PLAN_GENERATED:", "").strip()
        plan_data = json.loads(plan_json_str)
        search_params = SearchParameters(**plan_data)

        documents = PIPELINE.retrieve(
            query_str=search_params.search_query,
            filters=search_params.filters
        )

        if not documents:
            return "EXECUTION_SUCCESS: No documents were found matching the criteria."
        
        # Format the result for the agent
        formatted_docs = "\n\n---\n\n".join([
            f"Document (Score: {node.score:.4f}):\n{node.get_content()}" 
            for node in documents
        ])
        logger.info(f"Retrieval executed successfully. Found {len(documents)} documents.")
        return f"EXECUTION_SUCCESS:\n{formatted_docs}"
    except Exception as e:
        logger.error(f"Retrieval execution failed: {e}")
        return f"EXECUTION_ERROR: {e}"


# LangGraph
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class RetrievalAssistant:
    """A wrapper for the LLM that ensures it continues until a tool is called or content is generated."""
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State):
        while True:
            result = self.runnable.invoke(state)
            if not result.tool_calls and not result.content:
                # If the LLM returns an empty response, re-prompt it
                messages = state["messages"] + [("user", "Please respond with a tool call or a final answer.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": [result]}

def handle_tool_error(state: State) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(content=f"Error: {repr(error)}\nPlease fix your mistakes.", tool_call_id=tc["id"])
            for tc in tool_calls
        ]
    }

# Build Graph with Agent
def create_qdrant_agent_graph() -> StateGraph:
    """Builds and returns the LangGraph for the Qdrant agent."""

    assisstant_prompt = """
        You are a helpful assistant that answers questions by retrieving information from a document database
        Your task is to analyze the user's question and retrieve the most relevant information from the document database, 
        then provide a clear and concise answer based strictly on the retrieved content.

        IMPORTANT CONTEXT INFORMATION:
        - All tools operate within this context.

        WORKFLOW:
        1. Use `plan_retrieval_tool` to create a search plan (query and filters) from the user's question
        2. Use `validate_plan_tool` to check if the plan is valid.
        3. If validation fails, use `plan_retrieval_tool` again with the error feedback to fix the plan.
        4. Once validated, use `execute_retrieval_tool` to fetch the documents.
        5. If execution fails, analyze the error and use `plan_retrieval_tool` to create a better plan.
        6. After successfully retrieving documents, synthesize the information and provide a final, comprehensive answer in english to the user. 
        Do not call any more tools at this stage.

        IMPORTANT:
        - When you retrieve documents and synthesize answer from them, you MUST show me the exact text you took it from.
        - If the retrieved documents is unrelated to the question, do NOT try to generate answer.
        """
    
    retrieval_assistant_prompt = ChatPromptTemplate.from_messages([
        ("system", assisstant_prompt),
        ("placeholder", "{messages}"),
    ])

    tools = [plan_retrieval_tool, validate_plan_tool, execute_retrieval_tool]
    llm_with_tools = LLM.bind_tools(tools)

    assistant_runnable = retrieval_assistant_prompt | llm_with_tools

    builder = StateGraph(State)
    builder.add_node("retrieval_assistant", RetrievalAssistant(assistant_runnable))
    builder.add_node("tools", ToolNode(tools).with_fallbacks([RunnableLambda(handle_tool_error)], exception_key="error"))

    builder.add_edge(START, "retrieval_assistant")
    builder.add_conditional_edges("retrieval_assistant", tools_condition)
    builder.add_edge("tools", "retrieval_assistant")

    return builder.compile()

def _print_event(event: dict, _printed: set):
    """Helper function to print events from the graph stream."""
    for message in event.get("messages", []):
        if message.id not in _printed:
            logger.info(message.pretty_repr())
            _printed.add(message.id)

def main():
    """Main function to initialize and run the interactive agent session."""
    global PIPELINE, LLM
    try:
        # Initialize dependencies
        logger.info("Initializing dependencies...")
        LLM = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        storage_handler = NodeStorageHandler(collection_name="sailing_test")
        storage_handler.build_automerging_index()

        PIPELINE = QueryPipeline(storage_handler)

        # Create the LangGraph agent
        agent_graph = create_qdrant_agent_graph()

        print("\n" + "="*50)
        print("=== LangGraph Qdrant Tool-Using Agent ===")
        print("Ask me about your marketing documents. Type 'quit' to exit.")
        print("="*50)

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        _printed = set()

        while True:
            user_query = input("\nYour question: ").strip()
            if user_query.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            if not user_query:
                continue

            # Stream the agent's thought process
            events = agent_graph.stream({"messages": [("user", user_query)]}, config, stream_mode="values")
            for event in events:
                _print_event(event, _printed)

    except Exception as e:
        logger.error(f"A critical error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()