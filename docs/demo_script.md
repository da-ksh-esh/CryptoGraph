# Live demo script (about 6 minutes)

What the audience sees: real Binance prices up to the last hour, the model's view of which coins are at risk and why,
a past day replayed hour by hour as if live, and the evidence that its numbers can be trusted. Every number below comes
from tracked runs (`runs/`, `runs/live/`).

## Before the session (10 minutes ahead)

```bash
cryptograph-live                                            # newest hours: fetch, rebuild, verify, predict (~5 min)
cryptograph-train --config configs/experiment/replay_1h.yaml # replay data; once is enough (~2 min, offline)
streamlit run dashboard/app.py
```

If the `cryptograph-*` commands are not on the PATH, use `python -m cryptograph.pipelines.live` and
`python -m cryptograph.pipelines.train --config ...`. To keep the page current during the session, leave
`cryptograph-live --loop` running in a second terminal (one cycle at hh:02 UTC; the page picks up new runs within a
minute). Check with `cryptograph-live --status`.

**No internet in the room?** Everything except the live update works offline: the page shows the last live run with
a "paused" note, and the Replay, Track record and Past crises tabs need no network.

## The flow

1. **The question (30 s).** "When one coin crashes, which coins follow within hours?" Point at the **● LIVE** badge:
   Binance prices up to the last full hour, hours the model never saw (it was built on data up to September 2026).
2. **Now (60 s).** *What just happened* (the latest big moves, how unusual they were, how long such moves usually
   last) → **Who's at risk** (each coin's chance of a big move within the chosen window, against a typical coin) →
   **Why** (the inputs that pushed a coin's score up or down, in plain words).
3. **Replay (90 s).** Tab **Replay**, "10 Jan 2024: US spot bitcoin ETF approval" (test period, the green banner).
   Press **▶ Play**: the clock moves one hour per second; the model sees only the past; after each hour with big moves
   it names 3 coins to watch; outcomes appear only when their time has passed; the score compares its picks with random
   picks. Use **⏭ Next big move** to skip calm hours. Over this window at 1-3 h its picks came true about twice as often
   as random (1 h: 8 vs 4.4; 3 h: 19 vs 9.9).
4. **Honesty (60 s).** Switch to "10 Oct 2025: liquidation cascade": nearly every coin moves, so any pick is right and
   the model only matches random picks (6 h: 46 vs 48.4). Then "An ordinary week": single-coin moves, where the model's
   picks came true 3.6 times as often as random at 1 h (29 vs 8.1) and about twice at 6 h (77 vs 40.3). Toggle
   **Only coins that are calm right now** to ask the harder question (who is hit next, among coins not already moving).
5. **Can we trust the numbers? (60 s).** Tab **Track record**: on 2024–2026 test data, of the 1% of coin-hours the
   model rated riskiest, 28% made a big move within 1 h (random coin: 6%) and 60% within 6 h (random: 23%); the
   calibration chart shows that a stated chance comes true at about that rate. Below: the live alerts, each checked
   once its window has passed.
6. **The method in one breath.** Every live hour is rebuilt by the same research pipeline with every fitted setting
   frozen, and each update checks that the past is unchanged: all 59,085 research hours and 1,463,031 shock events
   reproduce exactly (`runs/live/status.json`, `verify`).

## Likely questions

- *Is the future leaking in?* No: the model's memory is updated only after it has scored an hour, live hours come
  after everything used to build it, and the data checks above run every cycle.
- *Why is it no better than random during big crashes?* Shocks spread market-wide within the same hour; when almost
  every coin moves there is nobody left to pick out. The model's value is in ordinary hours and partial sell-offs.
- *Does a simpler model do as well?* Say so plainly: on the test period a logistic regression on four features of the
  receiving coin and the market matches the TGN's precision; the TGN clearly beats the network-diffusion baseline. The
  signal is mostly the receiving coin's own turbulence (notebook 04).
- *Is this financial advice?* No: a research tool; shocks include sudden jumps as well as drops.
