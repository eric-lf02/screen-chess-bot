# Screen Chess Bot

A single-file Python bot that watches a chess board on your screen, figures out the position, asks [Stockfish](https://stockfishchess.org/) for the best move, and plays it with your mouse via `pyautogui`.

No machine learning model and no site-specific API: the bot learns what each piece looks like in your current theme by calibrating on the starting position, then uses template matching on piece silhouettes.

> ## ⚠️ Fair play
> Use this only against bots or engines, in offline games, or in your own test setups. **Do not use it in rated or unrated games against human opponents.** It violates the terms of service of essentially every chess site and ruins the game for the other player.

## Features

- **Theme-agnostic piece recognition** – calibrates from the starting position, so it works with any board or piece set.
- **Full game tracking** with `python-chess`, including castling, en passant and promotion.
- **Stockfish-powered moves** with configurable think time and skill level.
- **Click-click or drag** piece movement.
- **Auto new game** – optionally clicks a "New game" button when a game ends and waits for the starting position to appear.
- **Promotion handling** – records where the Queen/Knight choices appear and picks the right one.
- **Self-healing** – retries moves that didn't register, and resyncs from the screen if the tracked game drifts from what is displayed.
- **HiDPI / Retina aware** screenshots.
- **Safe to abort** – press `Ctrl+C`, or slam the mouse into a screen corner (`pyautogui` failsafe).

## Requirements

- Python 3.8+
- A [Stockfish](https://stockfishchess.org/download/) binary, on your `PATH` or passed with `--stockfish`
- Python packages:

```bash
pip install pyautogui pillow numpy python-chess
```

## Quick start

```bash
python chess_bot.py
```

On the first run the bot will walk you through setup:

1. **Region** – hover over the board's top-left corner, then its bottom-right corner (or type pixel coordinates). The region should be a square covering exactly the 8×8 grid.
2. **Orientation** – tell it whether White is at the bottom of the screen. The bot always plays the side at the **bottom**.
3. **Calibration** – set up the board in the **starting position** (no highlights, arrows or overlays) and press Enter. The bot saves a template for each piece.
4. **Play** – the bot reads the board, tracks the game, and moves when it's your side's turn.

Settings are saved to `chess_bot_config.json` and `chess_bot_templates.npz` in the working directory, so later runs start playing immediately.

If you start the bot mid-game, it will ask whether it's your turn and build the position from what it sees. Castling rights are inferred from king and rook placement.

## Usage

```bash
python chess_bot.py                        # setup if needed, then play
python chess_bot.py --reselect             # choose the board region again
python chess_bot.py --recalibrate          # relearn pieces (new theme / piece set)
python chess_bot.py --setnewgame           # record the "New game" button position, then play
python chess_bot.py --setpromo             # record where Queen/Knight promotion choices appear
python chess_bot.py --stockfish /path/to/stockfish --time 0.5 --skill 20 --drag
```

### Options

| Option | Default | Description |
| --- | --- | --- |
| `--stockfish PATH` | `stockfish` | Path to the Stockfish executable |
| `--time SECONDS` | `0.5` | Thinking time per move |
| `--skill N` | engine default | Stockfish Skill Level, 0–20 |
| `--poll SECONDS` | `0.4` | Delay between screen checks while waiting for the opponent |
| `--drag` | off | Drag pieces instead of click-click |
| `--reselect` | | Choose the board region again (also triggers recalibration) |
| `--recalibrate` | | Relearn piece appearance from the starting position |
| `--setnewgame` | | Pick the screen spot to click when a game ends |
| `--clearnewgame` | | Forget the saved new-game position |
| `--newgame-delay SECONDS` | `5` | Wait after a game ends before clicking "New game" |
| `--setpromo` | | Record where the Queen and Knight promotion choices appear |
| `--clearpromo` | | Forget the saved promotion positions |
| `--promo-delay SECONDS` | `2.5` | Wait after a promotion move before choosing the piece |

## How it works

1. **Screenshot** the board region and split it into 64 squares.
2. For each square, estimate its background colour from the border pixels and build a **silhouette mask** of pixels that differ from it.
3. **Classify** non-empty squares by comparing their mask and grayscale values against the calibrated templates (IoU minus a grayscale-difference penalty).
4. Compare the recognised placement with the tracked `python-chess` board. If the opponent has moved, the bot finds the one legal move that produces the observed position.
5. On your turn, ask Stockfish for a move and play it on screen with `pyautogui`, then wait for the opponent.

### Promotion

By default the bot promotes to a **queen**. If you run `--setpromo`, it records where the Queen and Knight options appear relative to the promotion square, which allows knight underpromotions too. Bishop and rook underpromotions are always converted to queens.

To use `--setpromo`, get a pawn to the last rank in a practice or analysis game so the choice dialog is showing, then follow the prompts.

### New game automation

With `--setnewgame` saved, when a game finishes the bot waits `--newgame-delay` seconds, clicks the saved spot, and waits for the starting position to appear. It retries up to 5 times before giving up.

## Troubleshooting

| Problem | Things to try |
| --- | --- |
| Calibration says a square "should be empty but isn't" | Remove move highlights, arrows and coordinates drawn on the board, and double-check the region and orientation |
| "Can't read the board yet" | Make sure nothing overlaps the board (popups, move animations, mouse cursor) and that the region is exact |
| Pieces misrecognised after a theme change | Run with `--recalibrate` |
| Clicks don't register | Try `--drag`, or re-run with `--reselect` to check the region |
| Wrong region on a multi-monitor or scaled display | Use `--reselect` and type coordinates manually |
| Nothing happens on macOS | Grant your terminal **Screen Recording** and **Accessibility** permissions |
| Doesn't work on Linux | `pyautogui` screenshots and input generally need an X11 session; Wayland is often problematic |

## Limitations

- Always plays the side at the bottom of the screen.
- Assumes a static 2D board that isn't covered by other windows.
- Highlights of last moves or legal-move dots can confuse recognition if they weren't present during calibration. Disable them or calibrate with them in a consistent state.
- Only queen and knight promotions are supported.
- Screen-based, so speed depends on the poll interval and on-screen animations.

## License

Add your license of choice here (e.g. MIT).
