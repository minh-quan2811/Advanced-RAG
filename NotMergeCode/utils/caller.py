# import docx
# import jaconv
# import re
# from environment_setup import EnvironmentSetup
from data_chunker import DocumentChunker
from langchain_experimental.text_splitter import SemanticChunker
from langchain_aws.embeddings import BedrockEmbeddings


class MarketingDocs:
    def __init__(self):
        self.embedding = BedrockEmbeddings(
            region_name="us-west-2",
            model_id="amazon.titan-embed-text-v2:0"
        )
        self.chunker = DocumentChunker()

        # File paths
        self.docx_1_path = r'C:\Users\Admin\Desktop\Advanced-RAG\client data\JCAI３０１ＭＫ.docx'
        self.docx_2_path = r'C:\Users\Admin\Desktop\Advanced-RAG\client data\ＪＣＡＩ３０２ＭＫ構造.docx'
        self.docx_3_path = r'C:\Users\Admin\Desktop\Advanced-RAG\client data\ＪＣＡＩ２０２デシル顧客.docx'
        self.docx_4_path = r'C:\Users\Admin\Desktop\Advanced-RAG\client data\ＪＣＡＩ２０４CRM.docx'

        # Metadata definitions
        self.docx_1_metadata = {
            "Category": [
                "Customer Segmentation",
                "Branding & Positioning",
                "Loyalty & Retention",
                "Retail Marketing",
                "Case Study"
            ],
            "Keywords": [
                "STP (Segmentation, Targeting, Positioning)",
                "Marketing Mix (4Ps)",
                "Relationship Marketing",
                "Lifestyle Segmentation",
                "Demand Creation (需要創造)",
                "Marketing Myopia (マーケティング近視眼)"
            ]
        }

        self.docx_2_metadata = {
            "Category": [
                "Customer Segmentation",
                "Branding & Positioning",
                "Loyalty & Retention",
                "Performance Measurement",
                "Case Study"
            ],
            "Keywords": [
                "Marketing Framework (Big Picture)",
                "STP (Segmentation, Targeting, Positioning)",
                "Customer-centric Service",
                "Brand Storytelling",
                "Moment of Truth (真実の瞬間)",
                "Marketing ROI"
            ]
        }

        self.docx_3_metadata = {
            'Category': [
                'Customer Value Analysis',
                'Customer Segmentation',
                'Loyalty & Retention',
                'Promotional Tactics',
                'Performance Measurement',
                'Case Study'
            ],
            'Keywords': [
                'Decile Analysis',
                'Customer Profiling',
                'Targeted Promotion',
                'Sales Forecasting',
                'High-Value Customer Identification'
            ]
        }

        self.docx_4_metadata = {
            "Category": [
                "Customer Value Analysis",
                "Loyalty & Retention",
                "Performance Measurement",
                "Promotional Tactics"
            ],
            "Keywords": [
                "Customer Lifetime Value (CLV)",
                "Churn Rate Analysis",
                "Retention Rate",
                "Customer Investment Management (CIM)",
                "Marketing ROI",
                "Cross-sell & Up-sell"
            ]
        }

    def get_nodes(self):
        docx_1_nodes = self.chunker.chunk_from_file(self.docx_1_path, self.docx_1_metadata)
        docx_2_nodes = self.chunker.chunk_from_file(self.docx_2_path, self.docx_2_metadata)
        # docx_3_nodes = self.chunker.chunk_from_file(self.docx_3_path, self.docx_3_metadata)
        # docx_4_nodes = self.chunker.chunk_from_file(self.docx_4_path, self.docx_4_metadata)

        all_nodes = []

        all_nodes.extend(docx_1_nodes)
        all_nodes.extend(docx_2_nodes)
        # all_nodes.extend(docx_3_nodes)
        # all_nodes.extend(docx_4_nodes)

        return all_nodes



# Example usage
if __name__ == "__main__":
    # md = MarketingDocs()
    # all_nodes = md.get_nodes()
    pass