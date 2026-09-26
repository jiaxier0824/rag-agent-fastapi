import unittest

from evaluation.run_evaluation import build_report, contains_expected


class EvaluationRulesTests(unittest.TestCase):
    def test_numeric_answer_does_not_match_course_code(self):
        self.assertFalse(contains_expected("INFS7410 的课程资料", "10"))
        self.assertTrue(contains_expected("共计 10 分", "10"))

    def test_english_term_uses_word_boundaries(self):
        self.assertFalse(contains_expected("encoder", "encode"))
        self.assertTrue(contains_expected("The model can encode text.", "encode"))

    def test_tool_only_case_does_not_inflate_knowledge_scores(self):
        results = [
            {
                "category": "knowledge",
                "keyword_correct": False,
                "tool_correct": True,
                "source_correct": False,
                "case_pass": False,
                "elapsed_seconds": 3.0,
            },
            {
                "category": "tool",
                "keyword_correct": True,
                "tool_correct": True,
                "source_correct": True,
                "case_pass": True,
                "elapsed_seconds": 1.0,
            },
        ]
        summary = build_report(results)["summary"]
        self.assertEqual(summary["knowledge_cases"], 1)
        self.assertEqual(summary["tool_cases"], 1)
        self.assertEqual(summary["keyword_accuracy"], 0.0)
        self.assertEqual(summary["source_attribution_accuracy"], 0.0)
        self.assertEqual(summary["tool_selection_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
