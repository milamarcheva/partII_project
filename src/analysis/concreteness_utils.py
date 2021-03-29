# def get_concreteness_score(story, lex):
#     n = len(story)
#     count = 0
#     sum = 0
#     i = 0
#
#     while i < n:
#         t1 = story[i]
#         if i != n - 1:
#             t1t2 = t1 + ' ' + story[i + 1]
#             if t1t2 in lex:
#                 sum += lex[t1t2]
#                 i += 2
#                 count += 2
#                 continue
#         if t1 in lex:
#             sum += lex[t1]
#             count += 1
#         i += 1
#
#     avg = sum / count
#     return avg
