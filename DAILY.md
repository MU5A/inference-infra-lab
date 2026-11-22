# Daily runbook

This file is the contract for every day's work, whether it is done with you watching or by a
scheduled run with nobody watching. A run follows it exactly. If a step cannot be completed
safely, the run stops and says so. It never improvises around a guardrail.

## 1. Pre-flight (stop and notify if any check fails)

1. Free disk space is at least 8 GB **and** at least three times what the day will download or
   build (`df -h .`). This machine ran out of space before, and starting Docker on a
   nearly full disk wedged it. If a day needs more than a third of the free space, stop and ask.
2. `gh auth status` succeeds, and `git status` is clean.
3. Find today's step: the first row in the README status table that is not `done`. Never skip a
   row. If a day was missed, do the oldest undone day.
4. Machine state for benchmark days: on AC power, Docker Desktop quit unless the day needs it
   (16 GB is shared), and the result metadata records `pmset -g therm`, load average and free
   memory. A run that starts throttled is rescheduled, not measured.
5. GPU measurements on this machine: it is a fanless MacBook Air, and sustained load slowed it to a
   third of its speed for several minutes while repeats stayed steady (Day 1, attempt 3). Take a
   short decode probe before and after every GPU phase and refuse the phase if the probe is more
   than 20% below the Day 1 baseline (`scripts/day01_followup.py`, `reference_decode`). Re-probe
   for up to ten minutes before giving up.

## 2. Do the day

Follow that day's entry in PLAN.md. Keep `days/NN-slug/LOG.md` open and append to it **as you go**:

- every command that was run, with the key lines of its output
- every dead end and error, and what fixed it. Failures are part of the record.
- the exact version or commit of every tool, and the sha256 of every model file used
- a short note on anything that surprised you

Benchmark rules: one warm-up run, then at least three measured runs. Report the median and the
spread (min and max). Fix the seed, the prompt set and the temperature. Never report a single run.
Raw results go in `days/NN-slug/results/` as small JSON or CSV. Every number in prose must be in a
results file.

## 3. Post-flight (all must pass before anything is pushed)

1. Tests and the linter pass locally.
2. No file over 5 MB is staged, no model weights or datasets are staged, and `models/` is ignored.
3. A secret scan of the staged diff finds nothing (keys, tokens, passwords, private URLs).
4. No em dashes anywhere in the changed files.
5. Only public or synthetic data was used. Nothing from any customer, pilot, patient or employer.
6. Every claim in the write-up is checked against a results file, and gets a label (M, S, L or G).
   A claim that cannot be traced is deleted or rewritten as a question.
7. EVIDENCE.md is updated with each new claim, its label, and the file that supports it.

## 4. Publish

Commit in logical steps with `git commit --signoff`. No attribution lines of any kind in commits
or pull requests. Push to `main`, then wait for the CI run and confirm every job is green
(`gh run watch --exit-status`). Update the README status row to `done` only after CI is green.

## 5. When something fails

Do not mark the day done. Do not retry in a loop. Write `STUCK.md` in the day's folder with what
happened, send one notification, and stop. The next run does the same day again.

## Guardrails

- **No spending.** No cloud GPU, no paid API, no purchase, no new account. Days labelled G run
  only in a session with you present and after you have said yes in that session.
- No real cloud infrastructure and no real cluster. A local `kind` cluster is fine.
- No destructive commands. The one exception is deleting model files under
  this repository's `models/` directory, through `scripts/prune_models.sh`, which refuses any
  other path.
- Touch only files inside this repository.
- Do not change the plan, the labels or this runbook without being asked.
- Do not publish a number you did not measure, and do not describe a simulation as a measurement.
