"""Shot-by-shot lip sync. LatentSync itself is not loaded."""

from pathlib import Path

from server.steps.lipsync import (
    NoFaceError,
    LipsyncModel,
    parse_scene_times,
    shot_ranges,
)


def test_shot_ranges_splits_on_cuts():
    assert shot_ranges(10.0, [3.0, 7.0]) == [
        (0.0, 3.0), (3.0, 7.0), (7.0, 10.0),
    ]


def test_shot_ranges_with_no_cuts_is_one_shot():
    assert shot_ranges(10.0, []) == [(0.0, 10.0)]


def test_shot_ranges_drops_cuts_too_close_to_an_edge():
    assert shot_ranges(10.0, [0.1, 9.95]) == [(0.0, 10.0)]


def test_parse_scene_times_reads_pts_time():
    log = (
        "[Parsed_showinfo_1 @ 0x1] n:0 pts:75 pts_time:1.50 pos:123\n"
        "[Parsed_showinfo_1 @ 0x1] n:1 pts:210 pts_time:4.20 pos:456\n"
    )
    assert parse_scene_times(log) == [1.5, 4.2]


def test_parse_scene_times_skips_cuts_closer_than_a_shot():
    log = "pts_time:1.00 x\npts_time:1.10 x\npts_time:3.00 x\n"
    assert parse_scene_times(log) == [1.0, 3.0]


def test_run_shots_keeps_the_shot_when_there_is_no_face(monkeypatch, tmp_path):
    model = LipsyncModel(tmp_path, tmp_path / "c.yaml", tmp_path / "w.pt")
    called = []

    def fake_run(video, audio, out, steps, guidance, seed=1247):
        called.append(Path(video).name)
        if "shot_001" in Path(video).name:
            raise NoFaceError("no face")
        Path(out).write_bytes(b"lip")
        return Path(out)

    monkeypatch.setattr(model, "run", fake_run)
    monkeypatch.setattr(
        "server.steps.lipsync.detect_scenes", lambda video, threshold=0.3: [2.0],
    )
    monkeypatch.setattr("server.steps.lipsync.audio.duration", lambda p: 4.0)
    monkeypatch.setattr("server.steps.lipsync.audio.video_size", lambda p: (64, 64))

    def fake_cut(src, start, end, dest):
        Path(dest).write_bytes(b"x")
        return Path(dest)

    monkeypatch.setattr("server.steps.lipsync.cut_segment", fake_cut)

    def fake_scale(src, dest, w, h):
        Path(dest).write_bytes(Path(src).read_bytes())
        return Path(dest)

    def fake_concat(paths, dest):
        Path(dest).write_bytes(b"out")
        return Path(dest)

    monkeypatch.setattr("server.steps.lipsync._scale_clip", fake_scale)
    monkeypatch.setattr("server.steps.lipsync.concat_videos", fake_concat)

    video = tmp_path / "v.mp4"
    audio = tmp_path / "a.wav"
    video.write_bytes(b"v")
    audio.write_bytes(b"a")
    out = tmp_path / "out.mp4"
    result = model.run_shots(video, audio, out, tmp_path / "shots", 20, 1.5)
    assert result == out
    assert called == ["shot_000.mp4", "shot_001.mp4"]


def test_run_shots_with_one_scene_uses_the_whole_video(monkeypatch, tmp_path):
    model = LipsyncModel(tmp_path, tmp_path / "c.yaml", tmp_path / "w.pt")

    def fake_run(video, audio, out, steps, guidance, seed=1247):
        Path(out).write_bytes(b"lip")
        return Path(out)

    monkeypatch.setattr(model, "run", fake_run)
    monkeypatch.setattr(
        "server.steps.lipsync.detect_scenes", lambda video, threshold=0.3: [],
    )
    monkeypatch.setattr("server.steps.lipsync.audio.duration", lambda p: 4.0)

    video = tmp_path / "v.mp4"
    audio = tmp_path / "a.wav"
    video.write_bytes(b"v")
    audio.write_bytes(b"a")
    out = tmp_path / "out.mp4"
    result = model.run_shots(video, audio, out, tmp_path / "shots", 20, 1.5)
    assert result == out
    assert out.read_bytes() == b"lip"
