class DenseRetriever:
    """
    Dense Retrieval 연결용 뼈대.
    실제 임베딩/FAISS 구현은 2주차에 A와 연결한다.
    """

    def __init__(self, chunks: list[dict], model_name: str | None = None):
        self.chunks = chunks
        self.model_name = model_name
        self.index = None
        self.embeddings = None

    def build_index(self):
        raise NotImplementedError(
            "Dense Retrieval 실제 인덱싱은 2주차 작업으로 분리합니다."
        )

    def save_index(self, index_path: str):
        raise NotImplementedError(
            "FAISS 인덱스 저장은 2주차 작업으로 분리합니다."
        )

    def load_index(self, index_path: str):
        raise NotImplementedError(
            "FAISS 인덱스 로드는 2주차 작업으로 분리합니다."
        )

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        raise NotImplementedError(
            "Dense 검색은 2주차에 임베딩 모델 확정 후 구현합니다."
        )