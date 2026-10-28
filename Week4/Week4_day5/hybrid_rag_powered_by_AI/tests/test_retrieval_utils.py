import unittest

from retrieval_utils import tokenize, reciprocal_rank_fusion


class RetrievalLogicTests(unittest.TestCase):
    def test_tokenize_character_names_and_punctuation(self):
        self.assertEqual(tokenize("Tom Riddle, Harry's diary!"),
                         ["tom", "riddle", "harry's", "diary"])

    def test_merge_rewards_documents_found_by_both_methods(self):
        self.assertEqual(reciprocal_rank_fusion([5, 1], [2, 5])[0], 5)

    def test_fusion_no_duplicates(self):
        result = reciprocal_rank_fusion([4, 3, 2], [2, 3, 4])
        self.assertEqual(len(result), len(set(result)))

    def test_empty_ranking(self):
        self.assertEqual(reciprocal_rank_fusion([], []), [])


if __name__ == "__main__":
    unittest.main()
