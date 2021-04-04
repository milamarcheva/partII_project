# import unittest
# from src.analysis.concreteness_utils import get_concreteness_score, lex
#
# class ConcretenessTest(unittest.TestCase):
#     def test_get_concretness_score(self):
#         story_tokens = ['the','day', 'start', 'nicely', 'with', 'a', 'slice','of','freshly','baked', 'banana','bread']
#         score_function = get_concreteness_score(story_tokens, lex)
#         print(score_function)
#         score_correct =35.47/12
#         self.assertEqual(score_function, score_correct)
#
#
# if __name__ == '__main__':
#     unittest.main()
# #
# # def test_get_concretness_score(self):
# #         story_tokens = ['the','day', 'start', 'nicely', 'with', 'a', 'slice','of','freshly','baked', 'banana','bread']
# #         score_function = get_tokens_concreteness(story_tokens)
# #         score_correct =35.47/12
# #         self.assertEqual(score_function, score_correct)
# #
# # test_get_concretness_score()