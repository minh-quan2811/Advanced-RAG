from importlib.metadata import metadata
from pydantic import BaseModel, Field
from langchain.schema import SystemMessage, HumanMessage

class Premeta(BaseModel):
    is_relevant: bool = Field(..., description="True if the document is relevant to marketing/economics, False otherwise.")
    relevance_reason: str = Field(..., description="Reasoning for the relevance check. If not relevant, explain why. If relevant, briefly state the core topic.")
    summary: str = Field(..., description="A detailed summary of the document's core concepts, focusing on the 'What' question (What is this about? What knowledge does it provide?). Max 1000 words.")
    candidate_categories: list[str] = Field(..., description="A preliminary list of categories extracted directly from the document.")
    candidate_keywords: list[str] = Field(..., description="A preliminary list of keywords extracted directly from the document.")

class Metadata(BaseModel):
    categories: list[str] = Field(..., description="Final, standardized list of categories (max 3), aligned with the database.")
    keywords: list[str] = Field(..., description="Final, curated list of the most important keywords (max 6), aligned with the database.")


class PremetaParser:
    def __init__(self, model, document_content):
        self.model = model 
        prompt_template = f"""
        You are a highly capable AI document analyst. Your task is to perform a two-step analysis on the provided document.

        ### Step 1: Domain Check and Relevance Assessment

        First, critically assess the document's primary subject matter. Your goal is to determine if the document is **substantially and directly about Marketing or Economics**.

        - A document is "Relevant" ONLY IF its core topic is marketing strategies, economic theories, market analysis, business models, consumer behavior analysis, or similar subjects. It must provide actionable insights, frameworks, or in-depth analysis on these topics.
        - A document is "Not Relevant" if it only tangentially mentions these topics, or if its core subject is something else (e.g., technology, science, art, law) and one would need to "stretch" the imagination to connect it to marketing/economics.

        Based on this assessment, you will determine two fields:
        1.  `is_relevant` (boolean): `true` if relevant, `false` otherwise.
        2.  `relevant_reason` (string):
            - If `true`, briefly explain WHY the document is relevant (e.g., "The document details a SWOT analysis framework for new market entry.").
            - If `false`, briefly explain WHAT the document is actually about (e.g., "The document is a technical paper on AI for fossil identification and does not discuss marketing or economic applications.").

        ### Step 2: Detailed Analysis (Conditional)

        **ONLY IF `is_relevant` is `true`**, proceed with the following analysis. If `is_relevant` is `false`, leave the corresponding JSON fields empty or null.

        1.  **Insightful Summary**: Create a detailed summary (not too short, but should be under 1000 words) of the core marketing/economic concepts, frameworks, and valuable knowledge shared in the documents, what if you think valuable, mark it on the summary.
        2.  **Candidate Extraction**:
            -   **Categories**: Extract broad topics.
            -   **Keywords**: Extract specific tools or terms (e.g., "SWOT Analysis", "Customer Journey Map").
        """

        self.system_message = SystemMessage(
            content = prompt_template
        )

        self.model = self.model.with_structured_output(Premeta)

    def ask(self, document_content: str) ->Premeta:
        return self.model.invoke([
            self.system_message,
            HumanMessage(content=document_content)
        ])
    

class MetadataExtractor:
    def __init__(self, model = None):
        self.model = model
        self.prompt_template = """
        You are a meticulous data curator specializing in standardizing marketing and economic metadata.
        Your task is to refine a set of candidate categories and keywords based on a document's summary and align them with our existing database.

        **Existing Database:**
        -   Existing Categories: {existing_categories}
        -   Existing Keywords: {existing_keywords}

        **Input from Previous Analysis:**
        -   Summary
        -   Candidate Categories: {candidate_categories}
        -   Candidate Keywords: {candidate_keywords}

        **Your Refinement Process:**

        1.  **Refine Categories:**
            -   Review the `candidate_categories`.
            -   For each candidate, check if it matches, is a synonym of, or is a sub-concept of a category in {existing_categories} If so, use the standard category from the database. (e.g., 'Customer analysis' should map to 'CRM Strategy').
            -   If a candidate is new, decide if it's a broad, important concept. If it's too specific, map it to a more general, existing category.
            -   Consolidate related candidates into a single, higher-level category, see the scope of those categories: {existing_categories} as knowledge to know about the scope of categories.
            -   The final list must be the most accurate, standard, and concise representation.
            -   The maximum categories the system have is about 30, counted the exist, so be concise when create new categories

        2.  **Refine and Select Keywords (Max 6):**
            -   Review the `candidate_keywords`.
            -   Select the 5-6 MOST IMPORTANT keywords that represent specific tools, methods, or core concepts from the summary.
            -   For each selected keyword, check if a standard version exists in `existing_keywords` and use it.
            -   Ensure keywords provide specific value not already captured by the category labels.

        **Final Output:**
        Produce the final, alphabetized lists of categories and keywords in a structured JSON format that match Metadata class. 
        """

    def format_prompt(self, db_categories, db_keywords, expected_categories, expected_keywords):
        self.system_message = SystemMessage(
            self.prompt_template.format(
                existing_categories=db_categories or [],
                existing_keywords=db_keywords or [],
                candidate_categories=expected_categories or [],
                candidate_keywords=expected_keywords or []
            )
        )

        self.model = self.model.with_structured_output(Metadata)

    def ask(self, summary: str, db_categories: list[str], db_keywords: list[str], expected_categories: list[str], expected_keywords: list[str]) -> Metadata:

        self.format_prompt(db_categories, db_keywords, expected_categories, expected_keywords)

        response = self.model.invoke([
            self.system_message,
            HumanMessage(content='This is the summary, based on this summary and the provided context, do your job: \n ' + summary)
        ])
        return response


    def extract_metadata(self, document_text: str, db_categories=None, db_keywords=None) -> Metadata:
        metadata = None

        from langchain_aws import ChatBedrock
        model = ChatBedrock(
            model="us.anthropic.claude-3-haiku-20240307-v1:0",
            region_name="us-west-2", 
            temperature=0,
            verbose=True
        )

        self.model = model

        if not document_text:
            print("Document is empty")
            return None

        premeta_extractor = PremetaParser(model, document_content=document_text)
        premetadata = premeta_extractor.ask(document_text)

        if not premetadata.is_relevant:
            print("Premetadata is not relevant")
            return None

        # Use defaults if db_categories or db_keywords are empty or None
        default_categories = [
            "Branding & Positioning",
            "Case Study",
            "Customer Segmentation",
            "Customer Value Analysis",
            "Loyalty & Retention",
            "Performance Measurement",
            "Promotional Tactics",
            "Retail Marketing"
        ]
        default_keywords = [ 
            "Decile Analysis",
            "Customer Profiling",
            "High-Value Customer Identification",
            "Targeted Promotion",
            "Sales Forecasting",
            "STP",
            "Marketing Mix (4Ps)",
            "Relationship Marketing",
            "Lifestyle Segmentation",
            "Demand Creation",
            "Marketing Myopia",
            "Marketing Framework",
            "Customer-centric Service",
            "Brand Storytelling",
            "Moment of Truth",
            "Marketing ROI",
            "Customer Lifetime Value",
            "Churn Rate Analysis",
            "Retention Rate",
            "Customer Investment Management",
            "Marketing ROI",
            "Cross-sell & Up-sell",
            "Thank You Letter",
            "RFM Analysis",
        ]

        use_categories = db_categories if db_categories else default_categories
        use_keywords = db_keywords if db_keywords else default_keywords

        metadata = self.ask(
            premetadata.summary,
            use_categories,
            use_keywords,
            premetadata.candidate_categories,
            premetadata.candidate_keywords
        )
        
        print(f"Extracted Metadata: {metadata}")

        return {
            "categories": metadata.categories,
            "keywords": metadata.keywords
        }
