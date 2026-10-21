# Marvin

A chess engine that plays like a human of a chosen rating, including how long it thinks.
Marvin is a neural network trained on Lichess games. Besides the position and both players'
ratings, it is conditioned on the time control, both clocks and how long recent moves took, and
predicts the move a human would play and how long they would spend on it. So the same position is
played differently in bullet than in classical, or with ten seconds left rather than ten minutes:
faster, and with more mistakes, when short on time.

## Setup

Python 3.10+.

```
pip install -r requirements.txt
python download_weights.py
```

This downloads two models into `weights/`: `79m` (default, stronger human prediction) and `23m`
(smaller). Both run fast on CPU. Add `--fp32` for the full-precision files, which are twice the size
and give practically identical outputs. Run it again after an update: it replaces weights from before
sub-second think times (below).

Think times are predicted to 40 ms: the model predicts whole seconds (Lichess records clocks to the
second), and a small extra head trained on Lichess's centisecond clocks places the time within that
second. This matters mostly in bullet, where many human moves take well under a second.

The model's inputs and outputs are documented on the
[weights page](https://huggingface.co/holymolyyy/marvin-onnx).

## Play in the browser

```
python -m marvin.gui
```

Options: `--model 23m`, `--port 8000`. Move and think-time settings are in the page.

The board is [chessground](https://github.com/lichess-org/chessground). The sounds are Lichess's
standard set (move, capture, check, low time, game start and end). Their license is unclear, so they
aren't in the repo: the GUI fetches them from Lichess the first time and keeps them in
`marvin/static/sound/`. Turn them off with **Sound**.

With **Analysis** off you play Marvin. With it on, the board is free: you move both sides, Marvin
doesn't reply, and each move spends the model's typical think time on that side's clock. A table
shows the moves a human would play in the viewed position, with think times and expected results.
Step through the moves with ← and → (Home and End for the start and end) and play out variations.
Turning analysis off again continues the game against Marvin from the position on the board.

Tick **Debug** to see the exact model inputs and outputs behind each of Marvin's moves.

## Use as a UCI engine

Add `marvin.sh` (Linux/macOS) or `marvin.bat` (Windows) as an engine in any UCI GUI
(Cute Chess, En Croissant, Arena) or in [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot).

| Option | Default | |
|---|---|---|
| `UCI_Elo` | 1500 | Marvin's rating (1200–2500) |
| `OpponentElo` | 0 | Opponent's rating. 0 = from `UCI_Opponent` if the GUI sends it, else same as `UCI_Elo` |
| `Model` | 79m | `79m` or `23m` |
| `Temperature` | 1.0 | Move sampling temperature. 1 = the model's distribution, 0 = always the most likely move |
| `TopP` | 1.0 | Only sample from the most likely moves covering this probability |
| `ThinkTime` | sample | `sample`: random draw from the predicted think time. `typical`: close to the most likely time, with some variation. `mean`: the average predicted time. `mode`: the single most likely time. `instant`: move immediately |
| `TimeTemperature` | 1.0 | Temperature for `sample` think times |
| `TimeTopP` | 1.0 | Top-p for `sample` think times |
| `TimeScale` | 1.0 | Multiplies think times (below 1 is faster, above 1 slower) |
| `Flag` | sample | When Marvin runs out of time. `sample`: with the model's probability of a human flagging in the position, but only with 5 seconds or less left. `threshold`: whenever that probability reaches 15%, at any clock. `never` |
| `BaseTime` | 0 | Game's base time in seconds. 0 = detect at game start |
| `FallbackTimeControl` | 5+0 | Time control assumed when the GUI sends no clock (minutes+increment) |
| `MoveOverhead` | 100 | Milliseconds kept in reserve per move, for GUI or network lag |

The standard `debug on` command makes Marvin print the model's inputs and outputs for every move.

### How time is handled

With a clock (normal games), the GUI sends both clocks and increments with every move. Marvin
detects the base time at the start of the game, measures the opponent's think time from the
clock, and then waits its own sampled think time before moving. Everything the model sees
(clocks, increments, recent move times) is real, rounded to whole seconds as in its training data.
When Marvin runs out of time (see `Flag`), it holds its move until its clock has expired, so the GUI
records a loss on time.

Without a clock (`go nodes`, `go infinite`, `go movetime`), Marvin assumes `FallbackTimeControl`,
estimates the missing move times with the model and simulates both clocks. It replies
immediately, or within `movetime` if one is given.

## License

GPL-3.0, see [LICENSE](LICENSE).
