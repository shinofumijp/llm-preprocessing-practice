from src.minhash import deduplicate_documents, create_minhash_lsh, create_minhash_index


class TestMinhash:
    def test_deduplicate_documents(self):
        documents = {
            "1": "This is a test document",
            "2": "This is a test document",
            "3": "This is the test document",
            "4": "これはテストドキュメントです",
            "5": "これもテストドキュメントです",
        }

        lsh = create_minhash_lsh(threshold=0.5)
        create_minhash_index(lsh, documents)

        deduplicated = deduplicate_documents(documents, lsh)
        assert deduplicated == {
            "2": "This is a test document",
            "3": "This is the test document",
            "5": "これもテストドキュメントです",
        }
