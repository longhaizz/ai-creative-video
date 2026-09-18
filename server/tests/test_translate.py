"""What the translator is told when a line came out the wrong length."""

import json


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


# -- a block handed back untranslated ---------------------------------------


HINDI_LONG = "सुप्रभात आपको देखकर अच्छा लगा कल रात आपकी नींद अच्छी थी"
# The three lines a real job came back with: times of day, a couple of Hindi
# words each. Too short for the diacritic heuristic to look at.
HINDI_SHORT = ["11.26.", "रात के, 11.46.", "बोले 3.45."]


def test_a_short_untranslated_line_is_caught():
    """The bug: two Hindi words in a Vietnamese dub went through unseen."""
    from server.steps.translate import lines_wrong_language

    assert lines_wrong_language(HINDI_SHORT, "vi") == [1, 2]


def test_every_target_language_is_watched_not_only_vietnamese():
    """The old check only measured Vietnamese marks, so an English dub
    scored the same for Hindi as it did for English."""
    from server.steps.translate import lines_wrong_language

    for target in ("en", "id", "th", "tr", "vi"):
        assert lines_wrong_language([HINDI_LONG], target) == [0], target


def test_a_line_in_the_right_script_is_left_alone():
    from server.steps.translate import lines_wrong_language

    fine = {
        "vi": "Chào buổi sáng, rất vui được gặp bạn",
        "en": "Good morning, nice to see you",
        "th": "สวัสดีตอนเช้า ยินดีที่ได้พบคุณ",
        "zh": "早上好，很高兴见到你",
        "ja": "おはようございます、お会いできて嬉しいです",
        "ko": "좋은 아침입니다, 만나서 반갑습니다",
        "ru": "Доброе утро, рад вас видеть",
        "ar": "صباح الخير، سعيد برؤيتك",
    }
    for code, line in fine.items():
        assert lines_wrong_language([line], code) == [], code


def test_numbers_and_names_do_not_count_as_another_language():
    """A time or a brand name is not a translation failure."""
    from server.steps.translate import lines_wrong_language

    assert lines_wrong_language(["11.26"], "vi") == []
    assert lines_wrong_language(["Lúc 11 giờ 26."], "vi") == []
    assert lines_wrong_language(["Hãy thử Shatai miễn phí"], "vi") == []


def test_a_latin_brand_name_in_an_arabic_line_is_fine():
    """The bug: "BeFit" inside a good Arabic line stopped the whole job."""
    from server.steps.translate import lines_wrong_language

    assert lines_wrong_language(["قم بتحميل تطبيق BeFit وابدأ الآن."], "ar") == []
    assert lines_wrong_language(["BeFit アプリをダウンロード"], "ja") == []


def test_an_english_line_left_in_an_arabic_dub_is_caught():
    from server.steps.translate import lines_wrong_language

    assert lines_wrong_language(["Download the BeFit app now."], "ar") == [0]


def test_drifting_into_vietnamese_is_still_caught():
    """The diacritic heuristic still does the job the script check cannot:
    English and Vietnamese are written in the same alphabet."""
    from server.steps.translate import lines_wrong_language

    assert lines_wrong_language(["Chào buổi sáng rất vui được gặp bạn"], "en") == [0]


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


# -- the short pieces of text printed on the picture ------------------------


def _fake_chat(monkeypatch, answer):
    """Answer one _chat call and keep what it was asked."""
    from server.steps import translate
    import json as _json

    seen = {}

    def fake_chat(system, user, api_key, model, json_mode=False, schema=None):
        seen["system"], seen["user"], seen["schema"] = system, user, schema
        return _json.dumps(answer)

    monkeypatch.setattr(translate, "_chat", fake_chat)
    return seen


def test_every_piece_is_translated_in_one_ask(monkeypatch):
    """One request per video, not one per piece: they share an advert."""
    from server.steps import translate

    seen = _fake_chat(monkeypatch, {"labels": {"0": "GIẢM 50%", "1": "MUA NGAY"}})
    out = translate.translate_labels(
        ["SALE 50%", "BUY NOW"], "VI", "key")

    assert out == ["GIẢM 50%", "MUA NGAY"]
    # Sent as JSON under the keys the answer must carry, so the model copies
    # them instead of counting its way down a numbered list.
    assert json.loads(seen["user"]) == {"0": "SALE 50%", "1": "BUY NOW"}
    assert "Vietnamese" in seen["system"]


def test_the_answer_cannot_drop_or_reorder_a_piece(monkeypatch):
    """Strict mode cannot pin an array's length, so the keys are numbered."""
    from server.steps import translate

    seen = _fake_chat(monkeypatch, {"labels": {"0": "a", "1": "b", "2": "c"}})
    translate.translate_labels(["x", "y", "z"], "VI", "key")
    labels = seen["schema"]["schema"]["properties"]["labels"]
    assert labels["required"] == ["0", "1", "2"]
    assert labels["additionalProperties"] is False


def test_a_piece_the_model_left_empty_keeps_its_own_text(monkeypatch):
    """A bad answer costs the translation, not the piece."""
    from server.steps import translate

    _fake_chat(monkeypatch, {"labels": {"0": "GIẢM 50%", "1": "  "}})
    assert translate.translate_labels(
        ["SALE 50%", "BUY NOW"], "VI", "key") == ["GIẢM 50%", "BUY NOW"]


def test_the_same_language_is_not_sent_to_the_model_at_all(monkeypatch):
    from server.steps import translate

    def blow_up(*a, **kw):
        raise AssertionError("nothing to translate, so nothing to ask")

    monkeypatch.setattr(translate, "_chat", blow_up)
    assert translate.translate_labels(
        ["SALE"], "same", "key", asr_meta={"language": "en"}) == ["SALE"]


def test_no_text_asks_nothing(monkeypatch):
    from server.steps import translate

    def blow_up(*a, **kw):
        raise AssertionError("no pieces, no request")

    monkeypatch.setattr(translate, "_chat", blow_up)
    assert translate.translate_labels([], "VI", "key") == []


def test_the_prompt_says_the_text_came_from_ocr(monkeypatch):
    """Without this the model copies "AND IT.LEARNS" straight through."""
    from server.steps import translate

    seen = _fake_chat(monkeypatch, {"labels": {"0": "x"}})
    translate.translate_labels(["AND IT.LEARNS"], "VI", "key")
    assert "OCR" in seen["system"]
    assert "numbers" in seen["system"] and "brand" in seen["system"].lower()


def test_a_long_list_is_broken_into_asks_of_a_size_the_model_holds(monkeypatch):
    """One 58 piece ask drifted: every answer past the fortieth was the
    translation of the piece after it, and the tail of the video was
    covered in the wrong words."""
    from server.steps import translate

    asks = []

    def fake_chat(system, user, api_key, model, json_mode=False, schema=None):
        given = json.loads(user)
        asks.append(given)
        return json.dumps({"labels": {k: f"<{v}>" for k, v in given.items()}})

    monkeypatch.setattr(translate, "_chat", fake_chat)
    texts = [f"piece {i}" for i in range(58)]
    out = translate.translate_labels(texts, "VI", "key")

    assert out == [f"<{t}>" for t in texts], "every piece keeps its own answer"
    assert len(asks) == 3
    assert all(len(ask) <= translate.LABELS_PER_ASK for ask in asks)
    # Each ask starts its keys again at 0, and the answers still line up.
    assert list(asks[1]) == [str(i) for i in range(translate.LABELS_PER_ASK)]


def test_a_piece_with_a_picture_is_read_off_it(monkeypatch):
    """OCR read "तुम्हारे यहाँ बच्चा होगा!" as nonsense; the model reads the
    picture and says what it read, for the log."""
    from server.steps import translate
    import json as _json

    seen = {}

    def fake_chat(system, user, api_key, model, json_mode=False, schema=None,
                  images=()):
        seen["system"], seen["schema"], seen["images"] = system, schema, images
        return _json.dumps({"labels": {
            "0": "GIẢM 50%",
            "1": {"read": "तुम्हारे यहाँ बच्चा होगा!", "text": "Bạn sắp có em bé!"}}})

    monkeypatch.setattr(translate, "_chat", fake_chat)
    said = []
    ctx = type("Ctx", (), {"log": staticmethod(said.append)})()
    out = translate.translate_labels(
        ["SALE 50%", "दतुम्हारे चर्हली हब"], "VI", "key", ctx=ctx,
        images=[None, b"png"])

    assert out == ["GIẢM 50%", "Bạn sắp có em bé!"]
    assert seen["images"] == [("picture of piece 1", b"png")]
    assert "read the text off the picture" in seen["system"]
    keys = seen["schema"]["schema"]["properties"]["labels"]["properties"]
    assert keys["0"] == {"type": "string"}
    assert keys["1"]["required"] == ["read", "text"]
    assert any("तुम्हारे यहाँ बच्चा होगा!" in line for line in said), said


def test_a_failed_ask_with_pictures_is_asked_again_without(monkeypatch):
    from server.steps import translate
    import json as _json

    calls = []

    def fake_chat(system, user, api_key, model, json_mode=False, schema=None,
                  images=()):
        calls.append(images)
        if images:
            raise translate.OpenAIError("Gemini HTTP 400: bad image")
        return _json.dumps({"labels": {"0": "Em bé"}})

    monkeypatch.setattr(translate, "_chat", fake_chat)
    out = translate.translate_labels(["बबच्चा"], "VI", "key", images=[b"png"])

    assert out == ["Em bé"]
    assert calls == [[("picture of piece 0", b"png")], ()]


def test_any_provider_is_retried_on_a_rate_limit():
    from server.steps import translate

    assert translate._worth_retrying(translate.OpenAIError("Gemini HTTP 429: slow"))
    assert translate._worth_retrying(translate.OpenAIError("OpenAI HTTP 503: down"))
    assert not translate._worth_retrying(translate.OpenAIError("Gemini HTTP 400: bad"))
