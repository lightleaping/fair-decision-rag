# Day 15 Submission Checklist

## 1. Core Result Files

- [x] `outputs/results/day10_bm25_top5.json`
- [x] `outputs/results/day11_chunk_validation_report.json`
- [x] `outputs/results/day12_hybrid_top5.json`
- [x] `outputs/results/day13_eval_report.json`
- [x] `outputs/results/day14_answer_trace.json`

## 2. Core Code Files

- [x] `scripts/build_chunks_jsonl.py`
- [x] `src/retrieval/bm25_retriever.py`
- [x] `src/retrieval/query_classifier.py`
- [x] `src/retrieval/section_boost.py`
- [x] `src/retrieval/topk_selector.py`
- [x] `src/retrieval/day10_a_runner.py`
- [x] `src/retrieval/day10_bm25_pipeline.py`
- [x] `src/retrieval/chunk_validator.py`
- [x] `src/retrieval/score_fusion.py`
- [x] `src/retrieval/hybrid_retriever.py`
- [x] `src/retrieval/evaluator.py`
- [x] `src/retrieval/answer_generator.py`
- [x] `src/retrieval/evidence_trace.py`
- [x] `main_day12_hybrid.py`
- [x] `main_day14_answer_trace.py`

## 3. Execution Commands

```bash
python scripts/build_chunks_jsonl.py
python -m src.retrieval.day10_bm25_pipeline
python -m src.retrieval.chunk_validator
python main_day12_hybrid.py
python -m src.retrieval.evaluator
python main_day14_answer_trace.py