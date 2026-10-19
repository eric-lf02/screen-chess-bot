#!/usr/bin/env python3
"""
Screen chess bot: pick the board region, recognise pieces, ask Stockfish, play with pyautogui.

Install:
    pip install pyautogui pillow numpy python-chess
    + a Stockfish binary (https://stockfishchess.org/download/) on your PATH or passed via --stockfish

How it works:
  1. Region   - you hover over the board's top-left and bottom-right corners (or type pixel coordinates).
  2. Calibrate- with the board in the STARTING position, the bot learns what every piece looks like
                in the current theme (template matching on piece silhouettes, so no ML model needed).
  3. Play     - the bot assumes you play the side at the BOTTOM of the screen. It reads the board,
                tracks the game with python-chess (castling / en passant), asks Stockfish for a move
                when it is your turn, and clicks it. Move the mouse to a screen corner to abort
                (pyautogui failsafe) or press Ctrl+C in the terminal.

Usage:
    python chess_bot.py                       # first run: select region + calibrate, then play
    python chess_bot.py --reselect            # choose the board region again
    python chess_bot.py --recalibrate         # relearn pieces (new theme / piece set)
    python chess_bot.py --setnewgame          # pick the spot to click when a game ends, then play
    python chess_bot.py --newgame-delay 8     # seconds to wait before clicking it (default 5)
    python chess_bot.py --setpromo            # record where the Queen/Knight promotion choices appear
    python chess_bot.py --promo-delay 3       # seconds to wait after a promotion move (default 2.5)
    python chess_bot.py --stockfish /path/to/stockfish --time 0.5 --skill 20 --drag

Only use this against bots, engines or your own offline games - not in rated games against humans.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np
import pyautogui
from PIL import Image

CONFIG_PATH = Path("chess_bot_config.json")
TEMPLATES_PATH = Path("chess_bot_templates.npz")

SQ_PX = 48          # every square is normalised to SQ_PX x SQ_PX for matching
MARGIN = 0.06       # fraction of a square cropped away (grid lines, borders)
FG_THRESHOLD = 70   # summed RGB distance from background to count as "piece pixel"
EMPTY_FRACTION = 0.05
MIN_SCORE = 0.35    # below this a non-empty square is considered unreadable

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.02


# --------------------------------------------------------------------------- region selection
def pick_point(label, delay=5):
    input(f"\nPress Enter, then hover the mouse over the {label} and keep it still ({delay}s)... ")
    for i in range(delay, 0, -1):
        print(f"  {i}...", end="", flush=True)
        time.sleep(1)
    p = pyautogui.position()
    print(f"\n  {label}: ({p.x}, {p.y})")
    return p.x, p.y


def select_region():
    print("=== Board region selection ===")
    mode = input("Select by [h]overing the mouse or [t]yping coordinates? [h/t]: ").strip().lower()
    if mode.startswith("t"):
        x1, y1 = map(int, input("Top-left corner of the board  (x y): ").split())
        x2, y2 = map(int, input("Bottom-right corner of the board (x y): ").split())
    else:
        x1, y1 = pick_point("TOP-LEFT corner of the board (outer corner of square a8/h1)")
        x2, y2 = pick_point("BOTTOM-RIGHT corner of the board")
    left, top = min(x1, x2), min(y1, y2)
    w, h = abs(x2 - x1), abs(y2 - y1)
    size = min(w, h)
    if abs(w - h) > 8:
        print(f"  Warning: selection is {w}x{h}, not square. Using {size}x{size}.")
    if size < 80:
        sys.exit("Region too small.")
    return [left, top, size]


# --------------------------------------------------------------------------- screen reading
def grab(region):
    left, top, size = region
    img = pyautogui.screenshot(region=(left, top, size, size)).convert("RGB")
    if img.size != (size, size):  # HiDPI / Retina: screenshot is in physical pixels
        img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img)


def image_to_square(row, col, white_bottom):
    if white_bottom:
        return chess.square(col, 7 - row)
    return chess.square(7 - col, row)


def square_center(sq, region, white_bottom):
    left, top, size = region
    q = size / 8
    f, r = chess.square_file(sq), chess.square_rank(sq)
    col = f if white_bottom else 7 - f
    row = 7 - r if white_bottom else r
    return left + (col + 0.5) * q, top + (row + 0.5) * q


def square_features(arr, row, col, size):
    q = size / 8
    m = int(q * MARGIN)
    y0, y1 = int(row * q) + m, int((row + 1) * q) - m
    x0, x1 = int(col * q) + m, int((col + 1) * q) - m
    crop = Image.fromarray(arr[y0:y1, x0:x1]).resize((SQ_PX, SQ_PX), Image.BILINEAR)
    a = np.asarray(crop).astype(np.int16)

    ring = np.concatenate([
        a[:3].reshape(-1, 3), a[-3:].reshape(-1, 3),
        a[:, :3].reshape(-1, 3), a[:, -3:].reshape(-1, 3),
    ])
    bg = np.median(ring, axis=0)                      # square's own background colour
    mask = np.abs(a - bg).sum(axis=2) > FG_THRESHOLD  # piece silhouette
    gray = (a.mean(axis=2) / 255.0).astype(np.float32)

    k = SQ_PX // 5
    empty = mask[k:-k, k:-k].mean() < EMPTY_FRACTION
    return empty, mask, gray


def classify(mask, gray, templates):
    syms, t_masks, t_grays = templates
    union_arr = t_masks | mask
    union = np.maximum(union_arr.sum(axis=(1, 2)), 1)
    inter = (t_masks & mask).sum(axis=(1, 2))
    iou = inter / union
    gray_diff = (np.abs(t_grays - gray) * union_arr).sum(axis=(1, 2)) / union
    score = iou - 0.5 * gray_diff
    i = int(np.argmax(score))
    return str(syms[i]), float(score[i])


def read_board(region, white_bottom, templates):
    """Return {chess.Square: 'P'/'n'/...} or None if some square can't be recognised."""
    arr = grab(region)
    size = region[2]
    result = {}
    for row in range(8):
        for col in range(8):
            empty, mask, gray = square_features(arr, row, col, size)
            if empty:
                continue
            sym, score = classify(mask, gray, templates)
            if score < MIN_SCORE:
                return None
            result[image_to_square(row, col, white_bottom)] = sym
    return result


def read_stable(region, white_bottom, templates, tries=6, gap=0.2):
    prev = None
    for _ in range(tries):
        cur = read_board(region, white_bottom, templates)
        if cur is not None and cur == prev:
            return cur
        prev = cur
        time.sleep(gap)
    return None


# --------------------------------------------------------------------------- calibration
def symbols(board):
    return {sq: p.symbol() for sq, p in board.piece_map().items()}


def calibrate(region, white_bottom):
    print("\n=== Calibration ===")
    input("Show the board in the STARTING position (orientation as chosen), then press Enter... ")
    time.sleep(0.5)
    arr = grab(region)
    expected = symbols(chess.Board())
    syms, masks, grays = [], [], []
    for row in range(8):
        for col in range(8):
            sq = image_to_square(row, col, white_bottom)
            empty, mask, gray = square_features(arr, row, col, region[2])
            exp = expected.get(sq)
            name = chess.square_name(sq)
            if exp is None and not empty:
                sys.exit(f"Calibration failed: square {name} should be empty but isn't "
                         f"(highlights/arrows? wrong region or orientation?).")
            if exp is not None:
                if empty:
                    sys.exit(f"Calibration failed: no piece found on {name}. Check region/orientation.")
                syms.append(exp)
                masks.append(mask)
                grays.append(gray)
    np.savez(TEMPLATES_PATH, syms=np.array(syms), masks=np.array(masks), grays=np.array(grays))
    print(f"Saved {len(syms)} piece templates to {TEMPLATES_PATH}.")


def load_templates():
    d = np.load(TEMPLATES_PATH)
    return d["syms"], d["masks"].astype(bool), d["grays"]


# --------------------------------------------------------------------------- game helpers
def board_from_placement(read, turn):
    b = chess.Board(None)
    for sq, sym in read.items():
        b.set_piece_at(sq, chess.Piece.from_symbol(sym))
    b.turn = turn
    rights = 0
    K, k, R, r = (chess.Piece(chess.KING, chess.WHITE), chess.Piece(chess.KING, chess.BLACK),
                  chess.Piece(chess.ROOK, chess.WHITE), chess.Piece(chess.ROOK, chess.BLACK))
    if b.piece_at(chess.E1) == K:
        if b.piece_at(chess.H1) == R:
            rights |= chess.BB_H1
        if b.piece_at(chess.A1) == R:
            rights |= chess.BB_A1
    if b.piece_at(chess.E8) == k:
        if b.piece_at(chess.H8) == r:
            rights |= chess.BB_H8
        if b.piece_at(chess.A8) == r:
            rights |= chess.BB_A8
    b.castling_rights = rights
    return b


def park_mouse(region):
    left, top, size = region
    sw, _ = pyautogui.size()
    x = left - 30 if left > 40 else min(left + size + 30, sw - 5)
    pyautogui.moveTo(x, top + size // 2, duration=0.1)


def play_on_screen(move, region, white_bottom, drag):
    x1, y1 = square_center(move.from_square, region, white_bottom)
    x2, y2 = square_center(move.to_square, region, white_bottom)
    if drag:
        pyautogui.moveTo(x1, y1, duration=0.15)
        pyautogui.dragTo(x2, y2, duration=0.25, button="left")
    else:
        pyautogui.moveTo(x1, y1, duration=0.15)
        pyautogui.click()
        time.sleep(0.12)
        pyautogui.moveTo(x2, y2, duration=0.15)
        pyautogui.click()
    time.sleep(0.2)
    park_mouse(region)


def set_promo(cfg):
    """Record where the Queen / Knight choices appear, relative to the promotion square."""
    region = cfg["region"]
    white_bottom = cfg["white_bottom"]
    print("\n=== Promotion choice positions ===")
    print("Get a pawn to the last rank (e.g. in a practice/analysis game) so the piece-choice")
    print("squares are showing. Don't pick a piece yet - leave the dialog open while you answer.")
    print("(If it closes when you click the terminal, re-open it, or re-run this step.)")
    while True:
        f = input("On which file (a-h) is the promotion dialog open? ").strip().lower()
        if len(f) == 1 and f in "abcdefgh":
            break
    target = chess.square("abcdefgh".index(f), 7 if white_bottom else 0)  # top row on screen
    cx, cy = square_center(target, region, white_bottom)
    q = region[2] / 8
    promo = {}
    for key, label in (("q", "QUEEN"), ("n", "KNIGHT")):
        x, y = pick_point(f"{label} option in the promotion dialog")
        promo[key] = [round((x - cx) / q, 4), round((y - cy) / q, 4)]
    cfg["promo"] = promo
    CONFIG_PATH.write_text(json.dumps(cfg))
    print("Saved promotion positions (stored relative to the promotion square, so any file works).")


def click_promotion(move, cfg, args):
    promo = cfg.get("promo")
    key = "n" if move.promotion == chess.KNIGHT else "q"
    if not promo or key not in promo:
        print("No promotion position saved (use --setpromo); not clicking a piece.")
        return
    region = cfg["region"]
    white_bottom = cfg["white_bottom"]
    cx, cy = square_center(move.to_square, region, white_bottom)
    q = region[2] / 8
    dx, dy = promo[key]
    print(f"Promotion: waiting {args.promo_delay}s, then choosing {'knight' if key == 'n' else 'queen'}.")
    time.sleep(args.promo_delay)
    pyautogui.moveTo(cx + dx * q, cy + dy * q, duration=0.2)
    pyautogui.click()
    time.sleep(0.3)
    park_mouse(region)


def start_new_game(args, cfg, templates):
    """Wait, click the saved 'New game' spot, and wait until the starting position shows up."""
    region = cfg["region"]
    white_bottom = cfg["white_bottom"]
    start_syms = symbols(chess.Board())
    x, y = cfg["new_game"]
    attempts = 5
    for attempt in range(1, attempts + 1):
        print(f"Waiting {args.newgame_delay}s, then clicking the new-game spot ({x}, {y}) "
              f"[attempt {attempt}/{attempts}]...")
        time.sleep(args.newgame_delay)
        pyautogui.moveTo(x, y, duration=0.2)
        pyautogui.click()
        time.sleep(0.5)
        park_mouse(region)
        deadline = time.time() + 15
        while time.time() < deadline:
            read = read_stable(region, white_bottom, templates, tries=4, gap=0.2)
            if read == start_syms:
                print("New game started.")
                return chess.Board()
            time.sleep(0.5)
    print("The starting position never appeared after clicking. Stopping.")
    return None


def open_engine(args):
    path = shutil.which(args.stockfish) or args.stockfish
    if not Path(path).exists() and not shutil.which(path):
        sys.exit(f"Stockfish not found ('{args.stockfish}'). Install it or pass --stockfish /path/to/stockfish")
    engine = chess.engine.SimpleEngine.popen_uci(path)
    if args.skill is not None:
        try:
            engine.configure({"Skill Level": args.skill})
        except chess.engine.EngineError:
            print("Engine doesn't support 'Skill Level'; ignoring --skill.")
    return engine


# --------------------------------------------------------------------------- main loop
def run(args, cfg, templates):
    region = cfg["region"]
    white_bottom = cfg["white_bottom"]
    me = chess.WHITE if white_bottom else chess.BLACK
    start_syms = symbols(chess.Board())
    engine = open_engine(args)

    print(f"\nPlaying as {'White' if me else 'Black'} (bottom side). "
          "Abort: Ctrl+C, or slam the mouse into a screen corner.")

    read = None
    while read is None:
        read = read_stable(region, white_bottom, templates)
        if read is None:
            print("Can't read the board yet (move overlays? wrong region?). Retrying...")
            time.sleep(1)

    if read == start_syms:
        board = chess.Board()
    else:
        ans = input("Board isn't in the starting position. Is it YOUR turn to move? [Y/n]: ").strip().lower()
        board = board_from_placement(read, me if not ans.startswith("n") else (not me))
        if not board.is_valid():
            sys.exit("Recognised position is not a legal chess position - recalibrate or check the region.")

    last_own = None      # (move, placement_before, time) of our latest move awaiting confirmation
    retries = 0
    mismatch = 0

    try:
        while True:
            if board.is_game_over():
                print(f"Game over: {board.result()} ({board.outcome().termination.name})")
                if not cfg.get("new_game"):
                    break
                board = start_new_game(args, cfg, templates)
                if board is None:
                    break
                last_own = None
                retries = 0
                mismatch = 0
                continue

            # ------------------------------------------------------------ our turn
            if board.turn == me:
                result = engine.play(board, chess.engine.Limit(time=args.time))
                move = result.move
                allowed = {chess.QUEEN}
                if "n" in (cfg.get("promo") or {}):
                    allowed.add(chess.KNIGHT)          # knight position known -> allow underpromotion
                if move.promotion and move.promotion not in allowed:
                    move = chess.Move(move.from_square, move.to_square, chess.QUEEN)
                print(f"Playing {board.san(move)} ({move.uci()})")
                before = symbols(board)
                play_on_screen(move, region, white_bottom, args.drag)
                if move.promotion:
                    click_promotion(move, cfg, args)
                board.push(move)
                last_own = (move, before, time.time())
                mismatch = 0
                continue

            # ------------------------------------------------------------ opponent's turn
            time.sleep(args.poll)
            read = read_stable(region, white_bottom, templates, tries=4, gap=0.15)
            if read is None:
                continue

            if read == symbols(board):
                mismatch = 0
                if last_own:
                    last_own = (*last_own[:2], last_own[2])  # still waiting for opponent
                continue

            if read == start_syms:
                print("Starting position detected - new game.")
                board = chess.Board()
                last_own = None
                continue

            # our own move didn't register on screen? undo and retry
            if last_own and read == last_own[1] and time.time() - last_own[2] > 3:
                retries += 1
                print(f"Move didn't appear on screen (attempt {retries}/3), retrying.")
                board.pop()
                last_own = None
                if retries >= 3:
                    sys.exit("Giving up: clicks aren't reaching the board. Check region / try --drag.")
                continue

            found = None
            for mv in list(board.legal_moves):
                board.push(mv)
                ok = symbols(board) == read
                board.pop()
                if ok:
                    found = mv
                    break
            if found:
                print(f"Opponent played {board.san(found)}")
                board.push(found)
                last_own = None
                retries = 0
                mismatch = 0
                continue

            mismatch += 1
            if mismatch >= 8:
                print("Position on screen doesn't follow from the game; resyncing from the screen.")
                nb = board_from_placement(read, me)
                if nb.is_valid():
                    board = nb
                    last_own = None
                    mismatch = 0
                else:
                    print("Resync produced an illegal position; still waiting.")
                    mismatch = 0
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        engine.quit()


# --------------------------------------------------------------------------- entry point
def main():
    ap = argparse.ArgumentParser(description="Screen chess bot (pyautogui + Stockfish)")
    ap.add_argument("--stockfish", default="stockfish", help="path to the Stockfish executable")
    ap.add_argument("--time", type=float, default=0.5, help="seconds of thinking per move")
    ap.add_argument("--skill", type=int, default=None, help="Stockfish Skill Level 0-20")
    ap.add_argument("--poll", type=float, default=0.4, help="seconds between screen checks")
    ap.add_argument("--drag", action="store_true", help="drag pieces instead of click-click")
    ap.add_argument("--setnewgame", action="store_true",
                    help="choose the screen position to click when a game ends (new game button)")
    ap.add_argument("--clearnewgame", action="store_true", help="forget the saved new-game position")
    ap.add_argument("--newgame-delay", type=float, default=5.0,
                    help="seconds to wait after a game ends before clicking new game")
    ap.add_argument("--setpromo", action="store_true",
                    help="record where the Queen and Knight promotion choices appear")
    ap.add_argument("--clearpromo", action="store_true", help="forget the saved promotion positions")
    ap.add_argument("--promo-delay", type=float, default=2.5,
                    help="seconds to wait after a promotion move before choosing the piece")
    ap.add_argument("--reselect", action="store_true", help="choose the board region again")
    ap.add_argument("--recalibrate", action="store_true", help="relearn piece appearance")
    args = ap.parse_args()

    cfg = None
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            cfg = None
        if not isinstance(cfg, dict) or "region" not in cfg or "white_bottom" not in cfg:
            print(f"Ignoring invalid {CONFIG_PATH.resolve()} - starting fresh setup.")
            cfg = None

    if cfg is None or args.reselect:
        region = select_region()
        ans = input("Is WHITE at the bottom of the screen? [Y/n]: ").strip().lower()
        cfg = {"region": region, "white_bottom": not ans.startswith("n")}
        CONFIG_PATH.write_text(json.dumps(cfg))
        print(f"Saved region {region} to {CONFIG_PATH}.")

    if args.clearnewgame:
        cfg.pop("new_game", None)
        CONFIG_PATH.write_text(json.dumps(cfg))
        print("Cleared the saved new-game position.")

    if args.setnewgame:
        print("\n=== New-game click position ===")
        x, y = pick_point("NEW GAME button (the spot to click after a game ends)")
        cfg["new_game"] = [x, y]
        CONFIG_PATH.write_text(json.dumps(cfg))
        print(f"Saved new-game position ({x}, {y}).")

    if args.clearpromo:
        cfg.pop("promo", None)
        CONFIG_PATH.write_text(json.dumps(cfg))
        print("Cleared the saved promotion positions.")

    if args.setpromo:
        set_promo(cfg)

    if args.recalibrate or args.reselect or not TEMPLATES_PATH.exists():
        calibrate(cfg["region"], cfg["white_bottom"])

    run(args, cfg, load_templates())


if __name__ == "__main__":
    main()