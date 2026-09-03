"""What the translator is told when a line came out the wrong length."""


def test_the_rewrite_prompt_carries_the_misses(monkeypatch):
    """A rewrite told only "25 words" is a guess. It must see the misses.

    The first real job asked once, got back 9% too long, and had no way to
    say so: the model never learned its last line overshot.
    """
    from server.steps import translate

    seen = {}

    def fake_chat(system, user, api_key, model):
        seen["system"], seen["user"] = system, user
        return "câu mới."

    monkeypatch.setattr(translate, "_chat", fake_chat)
    translate.rewrite_line(
        "câu gốc.", [("câu dài quá.", 7.42)], 5.70, 25, "Vietnamese", "key")

    assert "5.70 seconds" in seen["user"], seen["user"]
    assert "7.42s" in seen["user"], "the model must see what it really took"
    assert "30% too long" in seen["user"], seen["user"]
    assert "30% shorter" in seen["user"], "and which way to go next"
    assert "25 words" in seen["user"]
