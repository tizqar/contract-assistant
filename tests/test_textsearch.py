from contract_assistant.textsearch import Match, count, occurrences, pattern_of

TEXT = "Page one talks about the Customer.\nThe customer pays.\nNo CUSTOMER here? Yes, customer."
PAGE_STARTS = [0, 35, 54]


def test_occurrences_are_case_insensitive_and_in_order():
    found = occurrences(TEXT, "customer", PAGE_STARTS)
    assert [m.text for m in found] == ["Customer", "customer", "CUSTOMER", "customer"]
    assert [m.page for m in found] == [1, 2, 3, 3]
    assert found[0].start == TEXT.index("Customer")
    assert isinstance(found[0], Match)


def test_snippets_are_one_line_and_short():
    found = occurrences(TEXT, "pays", PAGE_STARTS)
    assert found[0].before == "Page one talks about the Customer. The customer"
    assert found[0].after == ". No CUSTOMER here? Yes, customer."


def test_a_query_matches_across_line_breaks_and_extra_spaces():
    assert count("the  Customer\nshall pay", "customer shall") == 1


def test_short_queries_find_nothing():
    assert occurrences(TEXT, "a") == []
    assert count(TEXT, " ") == 0
    assert pattern_of("x") is None


def test_the_limit_caps_the_matches():
    assert len(occurrences(TEXT, "customer", limit=2)) == 2
    assert count(TEXT, "customer") == 4


def test_without_page_starts_the_page_is_none():
    assert occurrences(TEXT, "pays")[0].page is None
