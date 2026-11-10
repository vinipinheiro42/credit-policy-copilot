from copilot.llm.extractive import segment


def test_table_rows_become_self_describing_units():
    text = "| Grade | Cap |\n|---|---|\n| A | 25,000 |\n| B | 15,000 |"
    assert segment(text) == ["Grade: A; Cap: 25,000.", "Grade: B; Cap: 15,000."]


def test_bullets_keep_their_lead_in():
    text = "Calls are limited as follows:\n- No calls before 8:00 a.m.\n- No more than 7 attempts in 7 days."
    units = segment(text)
    assert "Calls are limited as follows: No more than 7 attempts in 7 days." in units
