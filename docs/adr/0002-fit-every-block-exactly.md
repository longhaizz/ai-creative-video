# A block starts and stops with the speaker

The dub used to be allowed to drift. A block aimed at the room up to the
next block, so it spoke through the pause the speaker had left; it was only
ever squeezed, never stretched, so a short line went quiet over a moving
mouth; and a line that did not fit ran over by up to 400ms on purpose,
because "a rushed voice is heard by everyone and a late line by almost no
one". Real jobs left 1.4s to 5.7s of silence while the speaker was talking.

Now every block lands on the speaker's own start and end. Fitting is done in
this order:

1. The block is spoken with the wording whose measured length is closest to
   the target. The first take measures this voice, so the next choice is not
   a guess.
2. If the speed needed to land exactly is outside 0.98–1.03 — the band
   nobody hears — another wording is spoken instead. The three lengths that
   came with the block are free, so they go first; after that a new line is
   asked for, sized in words from the speed we measured.
3. Four tries at most, then the closest take is kept and the wide band
   0.85–1.25 lands it anyway.
4. Only a line that misses even that runs over, and it pushes the next block
   by at most DRIFT_CAP. A scene cut allows none.

This reverses two earlier decisions on purpose: "there is no asking again
for a shorter line" in translate.py, and preferring an overrun to a squeeze
in synth.py. The cost is real and was accepted: a job can take about three
times longer, and a line written to a word count reads less freely than one
written for meaning alone.

"Fit: worst Xms, average Yms, N lines rewritten, M blocks on the wide band"
is logged per job. The wide-band count is the one to watch — blocks landing
there are translations of the wrong size, not a tempo problem.
