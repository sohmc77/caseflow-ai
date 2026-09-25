"""Print the LangGraph graphs as Mermaid, generated from the compiled graphs themselves.

python -m agents.visualize
"""

from .document_review import DocumentReviewAgent
from .pipeline import REVIEW_GRAPH
from .providers import MockProvider


def main() -> None:
    print("%% Case review pipeline")
    print(REVIEW_GRAPH.get_graph().draw_mermaid())
    print("%% Agent loop (shared by every agent)")
    print(DocumentReviewAgent(MockProvider()).graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
