from typing import Dict, List, Any, Optional, TypedDict, Literal
from langgraph.graph import Graph, StateGraph, END
from langgraph.prebuilt import ToolNode
import json
import psycopg2
import os
from dotenv import load_dotenv

from dataclasses import dataclass
from enum import Enum
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

class AgentState(TypedDict):
    """State object that flows through the graph"""
    user_id: str
    conversation_id: str

    messages: List[Dict[str, Any]]
    user_input: str
    
    detected_intents: List[str]
    current_goal: Optional[str]
    plan: Dict[str, Any]
    retrieved_data: Dict[str, List[Dict]]
    final_response: str
    needs_reask: bool
    reask_reason: str

class ConversationAnalysis(BaseModel):
    """Structured LLM output capturing user intents, inferred goal, and reasoning."""
    intents: List[str] = Field(description="A list of detected intents from the user's input.")
    goal: str = Field(description="A single high-level goal type inferred from the conversation.")
    reasoning: str = Field(description="A brief explanation of why these intents and goal were selected.")


def get_conversation_context_from_db(conversation_id: str, max_messages: int = 3) -> str:
    """
    Fetches and formats the recent conversation history from the PostgreSQL database.
    """
    if not conversation_id:
        return "No conversation ID provided."

    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT"),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
        )
        cur = conn.cursor()

        query = """
            SELECT role, content 
            FROM messages
            WHERE conversation_id = %s
            ORDER BY created_at ASC
            LIMIT %s;
        """

        cur.execute(query, (conversation_id, max_messages))
        messages = cur.fetchall()

        cur.close()
        conn.close()

        if not messages:
            return "No previous conversation history found for this session."

        formatted_lines = []
        for role, content in messages:
            formatted_lines.append(f"{role.capitalize()}: {content}")
            
        return "\n".join(formatted_lines)

    except (Exception, psycopg2.Error) as error:
        print(f"Error while fetching from PostgreSQL: {error}")
        return "Error: Could not retrieve conversation history."

def intent_detection_node(state: AgentState) -> AgentState:
    """Intent Detection and Goal Extraction using LLM"""
    
    user_input = state["user_input"]
    conversation_id = state["conversation_id"]

    conversation_context = get_conversation_context_from_db(conversation_id)

    intent_detection_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert intent detection system. 
    Your task is to analyze the user's input and conversation context to identify:
    1. Current message intents (can be multiple)
    2. Overall conversation goal based on the flow of messages

    Available intents:
    - information_seeking: User wants to know facts, get explanations, or understand concepts
    - data_retrieval: User wants to find, search, or access specific data/records
    - explanation_request: User wants detailed explanations or clarifications  
    - problem_solving: User has a problem/issue they need help resolving
    - general_assistance: General help or casual conversation

    Available goals:
    - learning_assistance: User's overall goal is to learn or understand something
    - problem_resolution: User's overall goal is to solve a specific problem
    - information_retrieval: User's overall goal is to find specific information
    - general_assistance: General conversation or unclear goal

    Respond in valid JSON format:
    {{
    "intents": ["intent1", "intent2"],
    "goal": "goal_type",
    "reasoning": "Brief explanation of why these intents and goal were selected",
    }}"""),
    ("human", """Current message: {current_message}

    Conversation context:
    {conversation_context}

    Analyze the current message and conversation flow to determine intents and overall goal.""")
    ])

    prompt = intent_detection_prompt.format_messages(
        current_message=user_input,
        conversation_context=conversation_context
    )

    structured_llm = llm.with_structured_output(ConversationAnalysis)
    response = prompt | structured_llm

    result = json.loads(response.content)

    intents = result.get("intents", [])
    goal = result.get("goal", "general_assistance")

    return {
        **state,
        "detected_intents": intents,
        "current_goal": goal
    }

