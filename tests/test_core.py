import json
import tempfile
import unittest
from pathlib import Path

from src.retrieval.answer_generator import generate_extractive_answer
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.evaluator import recall_at_k, reciprocal_rank, token_f1
from src.retrieval.evidence_trace import build_evidence_trace, validate_evidence_trace


class CorePipelineTest(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "chunk_id": f"DOC-test-CH-{index:03d}",
                "doc_id": "DOC-test",
                "title": "테스트 사건",
                "section": "주문" if index == 1 else "이유",
                "section_type": "order" if index == 1 else "legal_reasoning",
                "text": text,
            }
            for index, text in enumerate(
                [
                    "피심인에게 시정명령을 부과한다.",
                    "피심인은 가격을 공동으로 결정하였다.",
                    "해당 행위는 공정거래법을 위반하였다.",
                    "관련 매출액을 기준으로 과징금을 산정하였다.",
                    "피심인은 위반 사실을 통지해야 한다.",
                ],
                1,
            )
        ]

    def test_bm25_returns_existing_unique_ids(self):
        results = BM25Retriever(self.chunks).search("가격 공동 결정", top_n=5)
        ids = [result["chunk_id"] for result in results]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(set(ids) <= {chunk["chunk_id"] for chunk in self.chunks})

    def test_metrics(self):
        self.assertEqual(recall_at_k(["a", "b"], ["b"]), 1.0)
        self.assertEqual(reciprocal_rank(["a", "b"], ["b"]), 0.5)
        self.assertGreater(token_f1("시정명령을 부과한다", "시정명령을 부과한다"), 0.99)

    def test_grounding_trace(self):
        results = [
            {**chunk, "score": 1.0 / rank}
            for rank, chunk in enumerate(self.chunks, 1)
        ]
        answer = generate_extractive_answer("시정명령은?", results)
        trace = build_evidence_trace("시정명령은?", results, answer)
        self.assertTrue(validate_evidence_trace(trace))
        self.assertTrue(
            set(answer["evidence_chunk_ids"]) <= set(trace["top5_chunk_ids"])
        )


if __name__ == "__main__":
    unittest.main()
