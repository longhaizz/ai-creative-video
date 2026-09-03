"""Speak the script and lay it on the timeline.

The unit is a block: everything between two anchors. An anchor is a silence
of ANCHOR_SILENCE or more, or a scene cut. One TTS call speaks a whole
block, so the voice keeps one intonation across its sentences instead of
starting again at every Whisper cue.

A block is never cut, and never cut mid-sentence: when a run of speech is
too long for one take, it is split after a full stop. When a block does not
fit its room, the line is shortened, then the speed is changed by at most
SOFT_TEMPO, and only then it may push the next block later. Overrunning by a
few hundred milliseconds is always preferred to squeezing the voice harder,
because a rushed voice is what a viewer hears as wrong.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from server import config
from server.jobs import PipelineError
from server.steps import duration as duration_model
from server.steps.audio import clean_take, duration, match_tempo, place_clips
from server.steps.translate import VARIANTS, rewrite_line, translate_blocks

# A pause this long at the end of a sentence ends a block. Ads are spoken
# without real breaks — the sample clip never pauses longer than 0.22s — so a
# high threshold would make the whole video one block.
ANCHOR_SILENCE = 0.15
# A pause this long is a break whatever the punctuation says: Whisper often
# leaves the full stop out, but nobody stops for a second mid-sentence.
LONG_SILENCE = 0.8
# VoxCPM starts to wander on long takes. This ceiling is a guess until the
# per-block numbers in the log say otherwise; see the "Blocks:" line.
MAX_BLOCK_SECONDS = 10.0
# Keep this much silence between two blocks.
MIN_GAP = 0.08
# How late a block may push the next one. A scene cut allows none, because a
# cut is the moment a viewer checks the lips against the sound.
DRIFT_CAP = 0.4
SCENE_DRIFT_CAP = 0.0
# A block must start and stop with the speaker, so the speed is always
# changed by the little that is left over. These are the two bands.
#
# Inside the first one nobody hears the change, so a line that lands here is
# kept. Outside it the line is written again instead — a shorter or longer
# wording of the same thing costs an API call, while a rushed voice costs
# the viewer. Only when four tries have not found one does the wide band
# run, and that block is logged: it is a translation the wrong size, not a
# tempo problem.
FIT_LOW, FIT_HIGH = 0.98, 1.03
LAST_LOW, LAST_HIGH = 0.85, 1.25
# A change smaller than this is not worth an ffmpeg pass. 0.2% of a five
# second block is 10ms, which is the accuracy we are promising.
FIT_DEADBAND = 0.002
# Takes one block may cost before the closest one so far is kept.
MAX_FIT_TRIES = 4
# A wording already written is only worth speaking when it is roughly the
# right size. The three that come with a block are guesses made before the
# voice was heard, and a guess that is a third of the room does not become
# right by being spoken: it just burns one of the four tries. Outside this
# band we go straight to asking for a line written against the measured
# speed. The first try is exempt — something has to be said to measure.
TRY_LOW, TRY_HIGH = 0.8, 1.3
# A block shorter than this is a sliver, not a block. Whisper cuts a number
# like "11.26" across two windows and leaves 0.44s holding ".26"; no
# sentence fits that, so the dub is squeezed to the floor and still runs
# over. Slivers are folded back into the block they were torn from.
MIN_BLOCK_SECONDS = 1.0
# Takes per block. The one that says the words best wins.
TAKES = 2
# Both takes worse than this: pay for one more.
EXTRA_TAKE_ERROR = 0.35
# A gap this long between two words of one take is the model hesitating.
MAX_HESITATION = 0.7
HESITATION_PENALTY = 0.3
# A take this much longer than the estimate is the model babbling, not a
# long line. Cue 22 of the MRI clip spoke 23.68s for a 1.8s slot.
BABBLE_FACTOR = 3.0

# Silence this long while the speaker is on screen is a hole a viewer
# notices. Nothing is re-spoken for it; it is logged so the variant ratios
# in translate.py can be raised if it keeps happening.
HOLE_SECONDS = 1.2

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_SENTENCE_END = re.compile(r"[.!?…。！？][\"'”’)\]]*$")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

# Voice presets, put in front of the text for /tts. Not used when cloning.
VOICE_PRESETS = {
    "male_young": "A young man, warm, clear and energetic voice",
    "male_middle": "A middle-aged man, low-pitched, warm and authoritative voice",
    "male_old": "An elderly man, deep, slightly raspy and slow voice",
    "female_young": "A young woman, bright, gentle and sweet voice",
    "female_middle": "A middle-aged woman, warm, confident and natural voice",
    "female_old": "An elderly woman, soft, mature and slightly raspy voice",
}


def with_voice_instruction(text: str, preset: str) -> str:
    """Put the voice description in front of the line."""
    description = VOICE_PRESETS.get(preset)
    if not description:
        raise PipelineError(f"Unknown voice preset: {preset}", code="invalid_input")
    body = (text or "").strip()
    if not body:
        raise PipelineError("There is no text to speak", code="invalid_input")
    return f"({description}){body}"


class VoxCPMModel:
    """Keeps the loaded TTS model between jobs."""

    name = "voxcpm"

    def __init__(self, model_id: str = "openbmb/VoxCPM2", device: str = "cuda:0"):
        self.model_id = model_id
        self.device = device
        self._model = None

    def load(self) -> None:
        from voxcpm import VoxCPM

        self._model = VoxCPM.from_pretrained(
            self.model_id, device=self.device, load_denoiser=False
        )

    def speak(self, text: str, out_wav: Path, cfg_value: float,
              timesteps: int, reference_wav: Path | None = None) -> Path:
        """Say one line. With reference_wav it copies that voice.

        reference_wav_path is the cloning mode, the one the desktop tool
        calls through /clone, and it needs no transcript of the reference.
        Do not swap it for prompt_wav_path: that is continuation mode, it
        wants the exact words of the reference in prompt_text, and it
        sounds worse here. Both arguments need voxcpm 2.x — on 1.x this
        call raises TypeError, which is the real fault to fix.
        """
        if self._model is None:
            raise PipelineError("The voice model is not loaded")

        import soundfile

        wav = self._model.generate(
            text=text,
            reference_wav_path=str(reference_wav) if reference_wav else None,
            cfg_value=cfg_value,
            inference_timesteps=timesteps,
        )
        out_wav = Path(out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(out_wav, wav, self._model.tts_model.sample_rate)
        return out_wav


# -- blocks -----------------------------------------------------------------


def build_blocks(cues: list[dict], scenes: list[float],
                 video_seconds: float) -> list[dict]:
    """Group the cues into the takes we will speak.

    Two passes. First the runs: a run ends at a scene cut, or where the
    speaker finishes a sentence and pauses. Then the long runs are split
    into blocks that one take can hold, always after a full stop — a seam
    in the middle of a sentence is the one seam a listener notices.
    """
    speaking = [cue for cue in cues if (cue.get("text") or "").strip()]
    if not speaking:
        raise PipelineError("There is nothing to say", code="internal")

    cuts = sorted(float(t) for t in scenes or [])
    runs: list[dict] = []
    for cue in speaking:
        start = _start(cue)
        if runs:
            last = runs[-1]["cues"][-1]
            gap = start - _end(last)
            cut = _cut_between(cuts, _end(last), start)
            if cut:
                runs[-1]["hard"] = True
            ends = gap >= LONG_SILENCE or (
                gap >= ANCHOR_SILENCE and ends_sentence(last))
            if not cut and not ends:
                runs[-1]["cues"].append(cue)
                continue
        runs.append({"cues": [cue], "hard": False})

    blocks: list[dict] = []
    for run in runs:
        pieces = split_to_cap(run["cues"])
        for piece in pieces:
            blocks.append(_block_from(piece))
        # Only the piece that ends at the cut may not drift over it.
        blocks[-1]["hard"] = run["hard"]

    blocks = _fold_slivers(blocks, cuts)

    for index, block in enumerate(blocks):
        if index + 1 < len(blocks):
            block["next_start"] = blocks[index + 1]["start"]
        else:
            block["next_start"] = max(video_seconds, block["end"])
    return blocks


def _fold_slivers(blocks: list[dict], cuts: list[float]) -> list[dict]:
    """Join every block too short to hold a sentence to its neighbour."""
    folded: list[dict] = []
    for block in blocks:
        short = block["end"] - block["start"] < MIN_BLOCK_SECONDS
        if folded and short and _joinable(folded[-1], block, cuts):
            folded[-1] = _join(folded[-1], block)
            continue
        folded.append(block)
    # A sliver at the very front has nothing behind it to join.
    while len(folded) > 1 and (
            folded[0]["end"] - folded[0]["start"] < MIN_BLOCK_SECONDS
            and _joinable(folded[0], folded[1], cuts)):
        folded[1] = _join(folded[0], folded[1])
        folded.pop(0)
    return folded


def _joinable(left: dict, right: dict, cuts: list[float]) -> bool:
    """Two blocks may be one take: no cut between them, and not too long."""
    if left["hard"]:
        return False  # left ends on a scene cut; nothing may cross it
    if _cut_between(cuts, left["end"], right["start"]):
        return False
    return right["end"] - left["start"] <= MAX_BLOCK_SECONDS


def _join(left: dict, right: dict) -> dict:
    joined = _block_from(left["cues"] + right["cues"])
    joined["hard"] = right["hard"]
    return joined


def _start(cue) -> float:
    return float(cue.get("speech_start", cue["start"]))


def _end(cue) -> float:
    return max(float(cue.get("speech_end", cue["end"])), _start(cue) + 0.05)


def ends_sentence(cue) -> bool:
    return bool(_SENTENCE_END.search((cue.get("text") or "").strip()))


def split_to_cap(cues: list[dict]) -> list[list[dict]]:
    """Cut a long run into takes, after a full stop wherever possible.

    When no sentence ends inside the ceiling, the widest pause in range is
    used instead. One cue is always taken, so this ends.
    """
    if _end(cues[-1]) - _start(cues[0]) <= MAX_BLOCK_SECONDS:
        return [cues]

    fits = [
        i for i in range(1, len(cues))
        if _end(cues[i - 1]) - _start(cues[0]) <= MAX_BLOCK_SECONDS
    ]
    if not fits:
        cut_at = 1
    else:
        sentences = [i for i in fits if ends_sentence(cues[i - 1])]
        if sentences:
            cut_at = max(sentences)
        else:
            cut_at = max(fits, key=lambda i: _start(cues[i]) - _end(cues[i - 1]))
    return [cues[:cut_at]] + split_to_cap(cues[cut_at:])


def _block_from(cues: list[dict]) -> dict:
    return {
        "start": _start(cues[0]),
        "end": max(_end(cue) for cue in cues),
        "cues": list(cues),
        "text": " ".join((cue.get("text") or "").strip() for cue in cues).strip(),
        "hard": False,
        "ref_cue": cues[0],
    }


def _cut_between(cuts: list[float], left: float, right: float) -> bool:
    """Is there a scene change inside this pause?"""
    return any(left <= t <= right for t in cuts)


# -- takes ------------------------------------------------------------------


def _normalize(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", (text or "").casefold()).split())


def text_error(asked: str, heard: str) -> float:
    """0.0 when the take says the line, 1.0 when it says something else."""
    left, right = _normalize(asked), _normalize(heard)
    if not left or not right:
        return 1.0
    return 1.0 - SequenceMatcher(None, left, right).ratio()


def best_take(line: str, work: Path, name: str, speak, listen, lang: str,
              expected: float, log, cue=None) -> dict:
    """Speak the line a few times and keep the best take.

    Best means: says the words (checked by listening to it), and is not
    wildly longer than the words can be. Both are needed — a take that
    babbles for 20 seconds scores badly on length even when the first words
    are right.
    """
    takes: list[dict] = []
    for number in range(TAKES):
        takes.append(_one_take(line, work, f"{name}_t{number}", speak,
                               listen, lang, expected, cue))
        if _score(takes[-1]) < 0.1:
            break  # clean and fluent, do not pay for another take
    if min(_score(take) for take in takes) > EXTRA_TAKE_ERROR:
        log("every take so far stumbled, trying once more")
        takes.append(_one_take(line, work, f"{name}_t{len(takes)}", speak,
                               listen, lang, expected, cue))

    best = min(takes, key=lambda take: (round(_score(take), 2),
                                        abs(take["length"] - expected)))
    best["takes"] = len(takes)
    if len(takes) > 1:
        log(f"{len(takes)} takes, kept {best['length']:.2f}s "
            f"(error {best['error']:.2f}, pause {best['hesitation']:.2f}s)")
    return best


def _score(take: dict) -> float:
    """How bad a take is: wrong words, plus a fine for hesitating."""
    penalty = HESITATION_PENALTY if take["hesitation"] > MAX_HESITATION else 0.0
    return take["error"] + penalty


def _one_take(line: str, work: Path, name: str, speak, listen, lang: str,
              expected: float, cue=None) -> dict:
    """Speak the line once, clean it, and listen to what came out.

    Cleaning happens before the length is measured, so the length is speech
    and not the silence the model padded around it.
    """
    raw = speak(line, work / f"{name}.wav", cue)
    wav = clean_take(raw, work / f"{name}_clean.wav")
    length = duration(wav)
    heard = ""
    words: list[dict] = []
    if listen is not None:
        result = listen(wav, lang) or {}
        heard = (result.get("text") or "").strip()
        words = list(result.get("words") or [])
    error = text_error(line, heard) if heard else 0.0
    if length > expected * BABBLE_FACTOR:
        error = 1.0  # the model ran away, whatever the words say
    return {
        "path": wav,
        "length": length,
        "heard": heard,
        "words": words,
        "error": error,
        "hesitation": longest_pause(words),
        "text": line,
    }


def longest_pause(words: list[dict]) -> float:
    """The widest gap between two words inside a take.

    A take can say every word and still sound wrong, because the model
    stopped in the middle of the sentence. The word times come free with
    the listen-back, so this costs nothing.
    """
    biggest = 0.0
    for left, right in zip(words, words[1:]):
        biggest = max(biggest, float(right.get("start", 0.0))
                      - float(left.get("end", 0.0)))
    return biggest


# -- laying the blocks on the timeline --------------------------------------


def timed_speech(
    cues: list[dict],
    work: Path,
    video_seconds: float,
    speak,
    openai_key: str,
    target_lang: str,
    meta: dict | None = None,
    ctx=None,
    listen=None,
    scenes: list[float] | None = None,
) -> Path:
    """Translate, speak every block, and lay them on one track.

    Returns a wav as long as the video. Every block keeps the moment the
    speaker started, give or take the drift a long line needs.
    """
    if not openai_key:
        raise PipelineError(
            "An OpenAI key is needed, even for the same language: it repairs "
            "the transcript and shares the lines out between blocks",
            code="internal",
        )

    blocks = build_blocks(cues, scenes or [], video_seconds)
    if ctx is not None:
        ctx.step(f"Rewriting the script ({len(blocks)} blocks)")

    # How long a line takes is learned from every take this server has ever
    # made; how fast this one cloned voice runs is measured as we go.
    model = duration_model.load(config.DURATION_DATA)
    speed = duration_model.Speed()
    lang_code = _seed_lang(target_lang, meta)

    script = translate_blocks(
        [
            {
                "start": block["start"],
                "end": block["end"],
                "text": block["text"],
                # The room is the speech itself, not the pause after it.
                "words": model.words_for(
                    block["end"] - block["start"], lang_code),
            }
            for block in blocks
        ],
        target_lang, openai_key, asr_meta=meta or {},
    )
    lines = list(script["lines"])
    lang_code = script.get("output_lang_code") or lang_code

    if ctx is not None:
        ctx.log(f"Meaning: {script['master_meaning']}")
        ctx.log(f"Script: {script['master_translation']}")
        fit_note = (f"fitted on {len(model.rows)} takes" if model.fitted
                    else "default weights, not enough history yet")
        ctx.log(f"Duration model: {fit_note}")
        for index, block in enumerate(blocks):
            ctx.log(f"Block {index + 1} heard: {block['text']}")
            ctx.log(f"Block {index + 1} script: {lines[index]['normal']}")

    if ctx is not None:
        ctx.step(f"Making the voice ({len(blocks)} blocks)")

    clips: list[tuple[float, Path]] = []
    spoken: list[dict] = []
    report = {"stretched": 0, "max_tempo": 1.0, "max_drift": 0.0,
              "errors": [], "overruns": 0, "takes": 0, "stumbles": 0,
              "longest_block": 0.0, "silence": 0.0, "holes": 0,
              "guess_error": [], "variants": {key: 0 for key in VARIANTS},
              "rewrites": 0, "wide": 0, "fit_error": []}
    delay = 0.0

    for index, block in enumerate(blocks):
        if ctx is not None:
            ctx.check_cancel()
        entry = lines[index]

        def log(message: str, _index=index):
            if ctx is not None:
                ctx.log(f"Block {_index + 1}: {message}")

        start = block["start"] + delay
        cap = SCENE_DRIFT_CAP if block["hard"] else DRIFT_CAP
        # What the block owns is the speech, not the pause behind it. Aiming
        # at the next block's start is what let the dub keep talking after
        # the speaker had already stopped.
        target = max(block["end"] - block["start"], 0.4)
        ceiling = max(block["next_start"] - start - MIN_GAP, 0.4) + cap

        take = fit_block(
            entry, target, work, index, speak, listen, lang_code,
            script.get("output_lang_name") or lang_code, model, speed,
            openai_key, block["ref_cue"], log,
        )
        if take["label"] in report["variants"]:
            report["variants"][take["label"]] += 1
        report["rewrites"] += take["rewrites"]
        report["guess_error"].append(
            abs(take["guess"] - take["length"]) / max(take["length"], 0.2))
        log(f"{take['length']:.1f}s for {target:.1f}s of speech, "
            f"{take['tries']} line(s), {take['takes']} take(s), "
            f"error {take['error']:.2f}, longest pause "
            f"{take['hesitation']:.2f}s")

        path, tempo = fit_tempo(take, target, ceiling, work, index, log)
        length = duration(path)
        report["fit_error"].append(abs(length - min(target, ceiling)))
        if not FIT_LOW <= take["need"] <= FIT_HIGH:
            report["wide"] += 1
        if tempo != 1.0:
            report["stretched"] += 1
            report["max_tempo"] = max(report["max_tempo"], tempo)
        report["errors"].append(take["error"])
        report["takes"] += take["spoken_takes"]
        report["longest_block"] = max(report["longest_block"], take["length"])
        if take["hesitation"] > MAX_HESITATION:
            report["stumbles"] += 1

        # Silence that matters is silence while the mouth is moving: the
        # block's own speech span, not the pause after it and not the tail
        # of the video.
        silence = max(0.0, (block["end"] - block["start"]) - length)
        report["silence"] += silence
        if silence >= HOLE_SECONDS:
            report["holes"] += 1
            log(f"{silence:.1f}s of silence left while the speaker is talking")

        overrun = max(0.0, start + length + MIN_GAP - block["next_start"])
        if overrun > cap + 0.01:
            report["overruns"] += 1
            log(f"runs {overrun * 1000:.0f}ms into the next block")
        # The next block is pushed by the overrun, but never by more than the
        # cap: it has its own moment to start on. Whatever is left over lands
        # on top of it. Two voices for a moment is bad; a block that starts a
        # second late is worse, and used to be possible because the squeeze
        # had no floor. A scene cut has a cap of zero, so nothing moves there.
        delay = min(overrun, cap)
        report["max_drift"] = max(report["max_drift"], delay)

        clips.append((start, path))
        spoken.extend(_sentence_cues(take, start, length, tempo))

    model.refit()

    if not clips:
        raise PipelineError("There is nothing to say", code="internal")

    (work / "spoken_cues.json").write_text(
        json.dumps(spoken, ensure_ascii=False, indent=2), encoding="utf-8")
    script["voice_speed"] = round(speed.value, 3)
    script["duration_coef"] = [round(value, 4) for value in model.coef]
    (work / "dub_script.json").write_text(
        json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")

    if ctx is not None:
        mean_error = sum(report["errors"]) / max(len(report["errors"]), 1)
        ctx.log(
            f"Smoothness: 0 lines cut, {report['stretched']} blocks stretched "
            f"(max {report['max_tempo']:.2f}x), "
            f"max drift {report['max_drift'] * 1000:.0f}ms, "
            f"{report['overruns']} overruns, "
            f"{report['stumbles']} blocks with a long pause inside, "
            f"listen-back error {mean_error:.2f}"
        )
        guess_error = (sum(report["guess_error"])
                       / max(len(report["guess_error"]), 1))
        mix = " ".join(f"{key} {report['variants'][key]}" for key in VARIANTS)
        ctx.log(
            f"Filling: {report['silence']:.1f}s silent while the speaker "
            f"talks, {report['holes']} holes over {HOLE_SECONDS:.1f}s, "
            f"lines chosen {mix}, length guess off by "
            f"{guess_error * 100:.0f}%, voice speed {speed.value:.2f}x"
        )
        gaps = report["fit_error"] or [0.0]
        ctx.log(
            f"Fit: worst {max(gaps) * 1000:.0f}ms, "
            f"average {sum(gaps) / len(gaps) * 1000:.0f}ms, "
            f"{report['rewrites']} lines rewritten, "
            f"{report['wide']} blocks on the wide band"
        )
        ctx.log(
            f"Blocks: {len(clips)} spoken, {report['takes']} takes, "
            f"longest {report['longest_block']:.1f}s "
            f"(ceiling {MAX_BLOCK_SECONDS:.0f}s)"
        )
    return place_clips(clips, video_seconds, work / "speech_timed.wav")


def _seed_lang(target_lang: str, meta: dict | None) -> str:
    code = (target_lang or "").lower()
    if code in ("", "same"):
        return ((meta or {}).get("language") or "").lower()
    return code


def fit_block(entry: dict, target: float, work: Path, index: int, speak,
              listen, lang: str, lang_name: str, model, speed, api_key: str,
              cue, log) -> dict:
    """Speak the block until its length needs no audible change of speed.

    Every take measures this voice, so the next wording is chosen against a
    number and not a guess. The lines that came with the block are spoken
    first because they are already paid for; only when all of them have been
    heard is a new one asked for. The closest take is kept, so a block that
    never lands still gets the best of what was tried.
    """
    # Every wording spoken, with the seconds it really took. The translator
    # gets this back: a rewrite that knows the last line came out 9% long is
    # a correction, while one told only "about 25 words" is another guess.
    spoken_lines: list[tuple[str, float]] = []
    rewrites = 0
    spoken_takes = 0
    best = None
    for attempt in range(MAX_FIT_TRIES):
        choice = _next_line(entry, spoken_lines, target, model, speed, lang,
                            lang_name, api_key, log)
        if choice is None:
            break  # nothing left to say that has not been said
        line, label, guess = choice
        rewrites += label == "rewrite"
        take = best_take(line, work, f"blk_{index:03d}_{attempt}", speak,
                         listen, lang, guess, log, cue)
        spoken_takes += take["takes"]
        # Only a take that said the words teaches anything. One that babbles
        # or swallows half the line is still a real number of seconds, but
        # they are the seconds of something else — writing it down teaches
        # the model that this sentence takes that long, which it does not.
        if take["error"] <= EXTRA_TAKE_ERROR:
            model.record(lang, take["text"], take["length"], speed.value)
            speed.observe(model.seconds(take["text"], lang), take["length"])
        take["need"] = take["length"] / max(target, 0.01)
        take["label"] = label
        take["guess"] = guess
        spoken_lines.append((line, take["length"]))
        if best is None or _closer(take, best):
            best = take
        if FIT_LOW <= take["need"] <= FIT_HIGH:
            break
        log(f"{take['length']:.2f}s for {target:.2f}s needs "
            f"{take['need']:.2f}x, looking for another wording")
    if best is None:
        raise PipelineError("A block came back with no line at all",
                            code="internal")
    best["tries"] = len(spoken_lines)
    best["rewrites"] = rewrites
    best["spoken_takes"] = spoken_takes
    return best


def _closer(take: dict, best: dict) -> bool:
    """Is this take the better one to keep? Words first, length second.

    Length alone would keep a take that says the wrong words at the right
    moment, which is the one thing a viewer cannot forgive. A take that
    stumbles only wins when everything else stumbled too.
    """
    stumbled = take["error"] > EXTRA_TAKE_ERROR
    if stumbled != (best["error"] > EXTRA_TAKE_ERROR):
        return not stumbled
    return abs(take["need"] - 1.0) < abs(best["need"] - 1.0)


def _next_line(entry: dict, spoken_lines: list, target: float, model, speed,
               lang: str, lang_name: str, api_key: str, log):
    """The next wording worth speaking, or None when there is none left.

    Returning None is not a failure. It means the block has no other way of
    being said — the three lengths were the same sentence, or the rewrite
    came back with what we already had — and trying again would only spend
    money to hear the same thing.
    """
    tried = [text for text, _seconds in spoken_lines]
    best = None
    for key in VARIANTS:
        text = (entry.get(key) or "").strip()
        if not text or text in tried:
            continue
        guess = model.seconds(text, lang) * speed.value
        if best is None or abs(guess - target) < best[0]:
            best = (abs(guess - target), key, text, guess)
    if best is not None:
        fits = TRY_LOW <= best[3] / max(target, 0.01) <= TRY_HIGH
        # Nothing spoken yet: say the closest one, whatever it measures.
        # Until a take is heard, `speed` is a guess and so is this.
        if fits or not tried or not api_key:
            return best[2], best[1], best[3]
    if not api_key:
        return None
    words = model.words_for(target / max(speed.value, 0.01), lang)
    source = (entry.get("normal") or (tried[-1] if tried else "")).strip()
    if not source:
        return None
    text = rewrite_line(source, spoken_lines, target, words,
                        lang_name, api_key).strip()
    if not text or text in tried:
        return None
    log(f"asked for a line of about {words} words")
    return text, "rewrite", model.seconds(text, lang) * speed.value


def fit_tempo(take: dict, target: float, ceiling: float, work: Path,
              index: int, log):
    """Land the block on the moment the speaker stopped.

    Both directions matter. A take that runs long is squeezed, and one that
    ends early is stretched — leaving it short is what left holes of silence
    while the speaker was still talking.

    `ceiling` is the only thing allowed to beat the target: a block may not
    be stretched into the one behind it.
    """
    goal = min(target, ceiling)
    need = take["length"] / max(goal, 0.01)
    if FIT_LOW <= need <= FIT_HIGH:
        slowest, fastest = FIT_LOW, FIT_HIGH
    else:
        slowest, fastest = LAST_LOW, LAST_HIGH
        log(f"still needs {need:.2f}x after every try, using the wide band")
    out, tempo = match_tempo(
        take["path"], goal, work / f"blk_{index:03d}_fit.wav",
        slowest=slowest, fastest=fastest, deadband=FIT_DEADBAND,
    )
    if tempo != 1.0:
        log(f"speed {tempo:.3f}x to land on {goal:.2f}s")
    return out, tempo


def _sentence_cues(take, start: float, length: float, tempo: float) -> list[dict]:
    """One subtitle per sentence, timed from the take we actually used."""
    text = take["text"].strip()
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [{"start": round(start, 3),
                 "end": round(start + length, 3),
                 "text": sentences[0]}]

    spans = _word_spans(sentences, take["words"], take["length"])
    out = []
    for sentence, (left, right) in zip(sentences, spans):
        out.append({
            "start": round(start + left / max(tempo, 0.01), 3),
            "end": round(min(start + right / max(tempo, 0.01), start + length), 3),
            "text": sentence,
        })
    return out


def _word_spans(sentences: list[str], words: list[dict],
                length: float) -> list[tuple[float, float]]:
    """Where each sentence sits inside the take.

    Word times come from listening to the take itself. When the listener
    heard a different number of words than we asked for, fall back to
    sharing the time out by text length: a wrong subtitle time is better
    than a wrong one that pretends to be exact.
    """
    counts = [len(s.split()) for s in sentences]
    if words and abs(len(words) - sum(counts)) <= max(2, sum(counts) // 4):
        spans = []
        cursor = 0
        for count in counts:
            first = words[min(cursor, len(words) - 1)]
            last = words[min(cursor + count - 1, len(words) - 1)]
            spans.append((float(first.get("start", 0.0)),
                          float(last.get("end", length))))
            cursor += count
        return spans

    total = sum(len(s) for s in sentences) or 1
    spans = []
    cursor = 0.0
    for sentence in sentences:
        share = length * len(sentence) / total
        spans.append((cursor, cursor + share))
        cursor += share
    return spans


def _selfcheck():
    """Block building and variant choice, without ffmpeg or a GPU."""
    def cue(start, end, text="hello there"):
        return {"start": start, "end": end, "speech_start": start,
                "speech_end": end, "text": text}

    # A short pause keeps one block, a long one ends it.
    blocks = build_blocks(
        [cue(0.0, 2.0), cue(2.2, 4.0), cue(6.0, 8.0)], [], 10.0)
    assert len(blocks) == 2, blocks
    assert blocks[0]["cues"] == blocks[0]["cues"]
    assert blocks[0]["next_start"] == 6.0
    assert blocks[1]["next_start"] == 10.0

    # A scene cut in the pause ends the block, and marks it hard.
    blocks = build_blocks([cue(0.0, 2.0), cue(2.2, 4.0)], [2.1], 6.0)
    assert len(blocks) == 2
    assert blocks[0]["hard"] is True

    # A very long run is split even without a pause.
    long_cues = [cue(i * 2.0, i * 2.0 + 1.9) for i in range(8)]
    blocks = build_blocks(long_cues, [], 20.0)
    assert len(blocks) > 1, "16s of speech must not be one take"
    assert all(b["end"] - b["start"] <= MAX_BLOCK_SECONDS + 2 for b in blocks)

    # The widest room takes the long line, the tightest takes the short one.
    model = duration_model.Model()
    speed = duration_model.Speed()
    entry = {
        "short": "mua ngay đi",
        "normal": "hãy mua ngay hôm nay",
        "long": "hãy mua ngay hôm nay để không bỏ lỡ điều gì cả",
    }
    _, variant, _ = pick_variant(entry, 5.0, 5.4, model, speed, "vi")
    assert variant == "long", variant
    _, variant, _ = pick_variant(entry, 1.0, 1.4, model, speed, "vi")
    assert variant == "short", variant

    # A line guessed to overrun is only taken when all three would.
    tight = {"short": "a b c d e f g h", "normal": "a b c d e f g h i j k l",
             "long": "a b c d e f g h i j k l m n o p"}
    _, variant, _ = pick_variant(tight, 0.5, 0.5, model, speed, "vi")
    assert variant == "short", variant

    # Listening back tells a good take from a wrong one.
    assert text_error("xin chào các bạn", "xin chào các bạn") < 0.01
    assert text_error("xin chào các bạn", "hôm nay trời mưa") > 0.5

    # Sentences share the take by word times when they line up.
    words = [{"start": i * 0.5, "end": i * 0.5 + 0.4} for i in range(6)]
    spans = _word_spans(["one two three", "four five six"], words, 3.0)
    assert spans[0][0] == 0.0 and spans[1][0] == 1.5, spans

    print("synth.py self-check OK")


if __name__ == "__main__":
    _selfcheck()
