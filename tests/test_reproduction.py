import tempfile
import unittest
from pathlib import Path
from swcv.voting import predict
from swcv.reproduce import records

class IntegrityTests(unittest.TestCase):
    def test_vote_aggregates_multilabel_evidence_without_dividing(self):
        values = [dict(similarity=.7, cwe_ids=["CWE-79", "CWE-89"]), dict(similarity=.2, cwe_ids=["CWE-89"]), dict(similarity=.8, cwe_ids=["CWE-20"])]
        self.assertEqual(predict(values), "CWE-89")

    def test_lexical_tie_and_empty_evidence(self):
        self.assertEqual(predict([dict(similarity=.5, cwe_ids=["CWE-2", "CWE-10"])]), "CWE-10")
        with self.assertRaises(ValueError):
            predict([])

    def test_duplicate_question_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text('{"question_id":"1"}\n{"question_id":"1"}\n')
            with self.assertRaises(ValueError):
                records(path)
