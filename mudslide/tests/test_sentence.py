# def test_sample_pair():
#     from mudslide.sentence import sample
#     sentence = sample("CC", "CCO")
#     print(sentence)

def test_sample_single():
    from mudslide.sentence import sample
    sentence = sample("CC")
    print(sentence)
