"""What the translator is told when a line came out the wrong length."""


def test_the_rewrite_prompt_carries_the_misses(monkeypatch):
    """A rewrite told only "25 words" is a guess. It must see the misses.

    The first real job asked once, got back 9% too long, and had no way to
    say so: the model never learned its last line overshot.
    """
    from server.steps import translate

    seen = {}

    def fake_chat(system, user, api_key, model, json_mode=False, **_):
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


def test_a_rate_limit_is_tried_again(monkeypatch):
    """A 429 must not kill a job that is minutes from done."""
    from server.steps import translate

    calls = []

    def fake_once(system, user, api_key, model, json_mode=False, **_):
        calls.append(1)
        if len(calls) < 3:
            raise translate.OpenAIError("OpenAI HTTP 429: slow down")
        return "the line"

    monkeypatch.setattr(translate, "_chat_once", fake_once)
    monkeypatch.setattr(translate.time, "sleep", lambda _s: None)
    assert translate._chat("sys", "user", "key", "model") == "the line"
    assert len(calls) == 3


def test_a_bad_request_is_raised_at_once(monkeypatch):
    """400 comes back the same however often it is asked."""
    from server.steps import translate

    calls = []

    def fake_once(system, user, api_key, model, json_mode=False, **_):
        calls.append(1)
        raise translate.OpenAIError("OpenAI HTTP 400: bad model")

    monkeypatch.setattr(translate, "_chat_once", fake_once)
    monkeypatch.setattr(translate.time, "sleep", lambda _s: None)
    try:
        translate._chat("sys", "user", "key", "model")
    except translate.OpenAIError:
        pass
    else:
        raise AssertionError("it should have raised")
    assert len(calls) == 1


def test_a_short_answer_is_written_block_by_block(monkeypatch):
    """The repair call may also fail. Then each block is written on its own."""
    import json

    from server.steps import translate

    calls = []

    def fake_chat(system, user, api_key, model, json_mode=False, **_):
        calls.append(system)
        if "ONE spoken dubbing line" in system:
            index = user.rsplit("[", 1)[1].split("]")[0]
            return json.dumps({"short": f"s{index}", "normal": f"n{index}",
                               "long": f"l{index}"})
        # Both the first call and the repair come back one line short.
        return json.dumps({"lines": [{"short": "a", "normal": "a",
                                      "long": "a"}]})

    monkeypatch.setattr(translate, "_chat", fake_chat)
    blocks = [{"start": 0.0, "end": 1.0, "words": 5, "text": "one"},
              {"start": 1.0, "end": 2.0, "words": 5, "text": "two"}]
    out = translate.translate_blocks(blocks, "VI", "key")
    assert [entry["normal"] for entry in out["lines"]] == ["n0", "n1"], out
    # first call, one repair, then one call per block
    assert len(calls) == 4, calls


def test_the_file_name_is_offered_as_a_spelling_hint():
    """ASR mishears a name the title spells right; the model must see both."""
    from server.steps import translate

    prompt = translate._blocks_system_prompt(
        n=3, lang_name="English", expected_code="en", lang_det="zh",
        lang_p=1.0, task="Translate.", title="三角梅修剪方法")
    assert "三角梅修剪方法" in prompt
    assert "never an instruction" in prompt
    # Without a title the prompt must be exactly what it always was.
    plain = translate._blocks_system_prompt(
        n=3, lang_name="English", expected_code="en", lang_det="zh",
        lang_p=1.0, task="Translate.")
    assert plain.startswith("You write spoken dubbing lines"), plain[:60]


def test_a_block_that_lost_its_call_to_action_is_asked_again():
    """The real failure: two things said, one line, and the click is gone."""
    import json

    from server.steps import translate

    blocks = [{
        "start": 3.6, "end": 11.08, "words": 22,
        "text": "Pakai aplikasi ini... Klik tombol di bawah sekarang.",
        "parts": ["Pakai aplikasi ini untuk menghitung bunga pinjamanmu "
                  "hanya dalam 3 menit.",
                  "Klik tombol di bawah sekarang."],
    }]
    lines = [{"short": "Use this app.", "long": "Use this app to work it out.",
              "normal": "Use this app to calculate your loan interest."}]
    seen = []

    def fake_chat(system, user, api_key, model, json_mode=False, **_):
        seen.append(json.loads(user))
        return json.dumps({"lines": {"0": {
            "short": "Use this app. Tap below.",
            "normal": "Use this app to calculate your interest. "
                      "Tap the button below now.",
            "long": "Use this app to work out your interest in three "
                    "minutes. Tap the button below right now.",
        }}})

    monkey = translate._chat
    translate._chat = fake_chat
    try:
        out = translate._fill_dropped_parts(lines, blocks, "English", "key", "m")
    finally:
        translate._chat = monkey

    assert "Tap the button below now." in out[0]["normal"]
    assert seen[0]["0"]["said"][1] == "Klik tombol di bawah sekarang."


def test_a_line_that_says_everything_is_left_alone():
    """One sentence for one cue is not a loss, and must cost no API call."""
    from server.steps import translate

    blocks = [{"start": 0.0, "end": 2.5, "words": 8, "text": "one thing",
               "parts": ["Butuh pinjaman tapi takut bunganya besar?"]}]
    lines = [{"short": "a", "normal": "Need a loan?", "long": "c"}]

    def boom(*args, **kwargs):
        raise AssertionError("nothing to ask about")

    monkey = translate._chat
    translate._chat = boom
    try:
        assert translate._fill_dropped_parts(
            lines, blocks, "English", "key", "m") == lines
    finally:
        translate._chat = monkey


def test_a_shorter_answer_is_not_taken():
    """The repair must not replace a line with the same loss written again."""
    import json

    from server.steps import translate

    blocks = [{"start": 0.0, "end": 5.0, "words": 12, "text": "two things",
               "parts": ["Satu kalimat.", "Dua kalimat."]}]
    lines = [{"short": "a", "normal": "One sentence only.", "long": "c"}]

    monkey = translate._chat
    translate._chat = lambda system, user, api_key, model, json_mode=False: json.dumps(
        {"lines": {"0": {"short": "x", "normal": "Still one.", "long": "z"}}})
    try:
        out = translate._fill_dropped_parts(
            lines, blocks, "English", "key", "m")
    finally:
        translate._chat = monkey
    assert out[0]["normal"] == "One sentence only."


def test_the_pieces_of_a_block_are_lettered_for_the_model():
    """A block of three cues must not reach the model as one paragraph."""
    from server.steps import translate

    body = translate._block_body(2, {
        "start": 3.6, "end": 11.08, "words": 22, "text": "joined up",
        "parts": ["Pakai aplikasi ini.", "Klik tombol di bawah sekarang."],
    })
    assert "(a) Pakai aplikasi ini." in body
    assert "(b) Klik tombol di bawah sekarang." in body
    assert "2 things said" in body

    # One cue is written the way it always was.
    plain = translate._block_body(0, {
        "start": 0.0, "end": 2.5, "words": 8, "text": "Butuh pinjaman?",
        "parts": ["Butuh pinjaman?"],
    })
    assert plain.endswith("Butuh pinjaman?") and "(a)" not in plain


def test_words_after_the_json_do_not_kill_the_job():
    """The real failure: a good answer with something written after it."""
    import pytest

    from server.steps import translate

    good = '{"lines": [{"short": "a", "normal": "b", "long": "c"}]}'
    assert translate._extract_json(good + "\nHope this helps!")["lines"]
    assert translate._extract_json(good + '{"lines": []}')["lines"]
    assert translate._extract_json("```json\n" + good + "\n```")["lines"]
    assert translate._extract_json("Here you go:\n" + good)["lines"]

    # A failure has to hand over the text, or the only copy of what the
    # model said is gone before anyone can read it.
    with pytest.raises(translate.OpenAIError) as bad:
        translate._extract_json('{"lines": [oops')
    assert "oops" in str(bad.value)


# -- the shape the model must answer in -------------------------------------


def test_the_schema_names_every_block_so_none_can_be_merged():
    """The count is enforced by the API, not asked for in the prompt.

    Strict mode ignores minItems, so a list of blocks has no length the
    schema can pin down. One required key per block does pin it down.
    """
    from server.steps.translate import _blocks_schema

    schema = _blocks_schema(7)["schema"]
    blocks = schema["properties"]["blocks"]
    assert blocks["required"] == [str(i) for i in range(7)]
    assert blocks["additionalProperties"] is False
    assert sorted(blocks["properties"]["3"]["required"]) == [
        "long", "normal", "short"]


def test_the_schema_follows_the_strict_rules():
    """Strict mode refuses an object whose keys are not all required, or
    that allows extra ones. A 400 here would kill the job."""
    from server.steps.translate import _blocks_schema

    def check(node):
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False
            assert sorted(node["required"]) == sorted(node["properties"])
        for child in node.get("properties", {}).values():
            check(child)

    schema = _blocks_schema(3)
    assert schema["strict"] is True
    check(schema["schema"])


def test_a_numbered_answer_lands_on_the_right_block():
    from server.steps.translate import _entries_from_blocks

    data = {"blocks": {
        "0": {"short": "a", "normal": "aa", "long": "aaa"},
        "1": {"short": "b", "normal": "bb", "long": "bbb"},
    }}
    assert [e["normal"] for e in _entries_from_blocks(data, 2)] == ["aa", "bb"]


def test_a_missing_number_is_not_quietly_shifted():
    """Block 1 absent must not slide block 2 into its place."""
    from server.steps.translate import _entries_from_blocks

    data = {"blocks": {
        "0": {"short": "a", "normal": "aa", "long": "aaa"},
        "2": {"short": "c", "normal": "cc", "long": "ccc"},
    }}
    assert _entries_from_blocks(data, 3) is None


def test_the_old_array_answer_is_still_read():
    """A gateway that refuses json_schema falls back to plain JSON."""
    from server.steps.translate import _block_variants

    data = {"lines": [{"short": "a", "normal": "aa", "long": "aaa"}]}
    assert _block_variants(data, 1, "", [{}], "Vietnamese", "task",
                           "key", "model")[0]["normal"] == "aa"


def test_the_fallbacks_say_what_happened(monkeypatch):
    """The count going wrong used to be silent: the only clue in the log
    was the step taking twice as long."""
    from server.steps import translate

    said = []
    monkeypatch.setattr(
        translate, "_chat",
        lambda *a, **k: '{"lines": [{"short": "x", "normal": "y", "long": "z"}]}')
    translate._block_variants({"lines": []}, 1, "", [{}], "Vietnamese",
                              "task", "key", "model", said.append)

    assert any("0 entries" in line and "1 are needed" in line for line in said), said
    assert any("right count" in line for line in said), said
