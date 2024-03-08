def test_str():
    import mudslide
    from mudslide.filters.clause import sample
    clause = sample("C")
    print(clause)