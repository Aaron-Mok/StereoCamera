#!/usr/bin/env python3
"""
Generate a print-ready ChArUco calibration board as a vector PDF.

The pattern is rasterised by OpenCV at a bit-exact resolution (every ArUco bit
is a whole number of pixels), then decomposed into filled rectangles and written
as PDF vector geometry — so edges stay perfectly crisp at any print resolution,
which is what subpixel corner accuracy depends on.

Usage:
  python3 make_charuco_board.py [output.pdf]
"""
import sys
import os

import cv2
import numpy as np

# ─── Board specification ───────────────────────────────────────────────────────
SQUARES_X      = 10         # squares across
SQUARES_Y      = 6          # squares down
SQUARE_MM      = 80.0       # checker square size
MARKER_RATIO   = 0.75       # marker size as a fraction of the square

# 4x4 markers carry their bits in 6 cells rather than 7, so each bit is larger
# at a given marker size — that decode margin is what pays for the smaller
# squares, and smaller squares are what keep enough corners alive in the
# partial views this board exists to tolerate.
DICT_NAME      = "DICT_4X4_50"

SHEET_W_MM     = 914.4      # 36 in — landscape
SHEET_H_MM     = 609.6      # 24 in

LABEL_PT       = 11.0       # spec text in the bottom margin; 0 disables
# ───────────────────────────────────────────────────────────────────────────────

MM_PER_PT = 25.4 / 72.0


def make_board():
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_MM, SQUARE_MM * MARKER_RATIO, d)
    return board, d


def square_px():
    """Pixels per square that keep every ArUco bit an integer size.

    A marker spans (dict bits + 2 border bits) cells and sits centred in its
    square, so the square must divide evenly by the bit count *and* leave an
    even margin around the marker.
    """
    bits = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, DICT_NAME)).markerSize + 2
    for px in range(bits * 2, 4096, 2):
        marker = px * MARKER_RATIO
        if marker % bits == 0 and (px - marker) % 2 == 0:
            return int(px)
    raise ValueError("no bit-exact square size for this marker ratio")


def rasterise(board, sq_px):
    """Board as a bool mask, True where ink goes."""
    img = board.generateImage((SQUARES_X * sq_px, SQUARES_Y * sq_px),
                              marginSize=0, borderBits=1)
    return img < 128


def to_rectangles(mask):
    """Greedy maximal-rectangle decomposition of the ink mask.

    Emitting one rect per pixel would work but bloats the file and lets
    renderers show antialiasing seams between abutting fills; merging into
    maximal blocks keeps the geometry exact and the rect count small.
    """
    m = mask.copy()
    h, w = m.shape
    rects = []
    for y in range(h):
        x = 0
        while x < w:
            if not m[y, x]:
                x += 1
                continue
            x2 = x
            while x2 < w and m[y, x2]:
                x2 += 1
            run = slice(x, x2)
            y2 = y + 1
            while y2 < h and m[y2, run].all():
                y2 += 1
            m[y:y2, run] = False
            rects.append((x, y, x2 - x, y2 - y))
            x = x2
    return rects


def write_pdf(path, rects, sq_px, label):
    """Minimal single-page PDF. Coordinates are points, origin bottom-left."""
    sheet_w = SHEET_W_MM / MM_PER_PT
    sheet_h = SHEET_H_MM / MM_PER_PT
    scale   = (SQUARE_MM / sq_px) / MM_PER_PT   # mask pixels → points
    pat_w   = SQUARES_X * sq_px * scale
    pat_h   = SQUARES_Y * sq_px * scale
    ox      = (sheet_w - pat_w) / 2.0
    oy      = (sheet_h - pat_h) / 2.0
    if ox < 0 or oy < 0:
        raise ValueError(f"pattern {pat_w*MM_PER_PT:.0f}×{pat_h*MM_PER_PT:.0f} mm "
                         f"does not fit the sheet")

    ops = ["0 g"]
    for x, y, w, h in rects:
        # PDF y grows upward; the mask's y grows downward.
        px = ox + x * scale
        py = oy + pat_h - (y + h) * scale
        ops.append(f"{px:.4f} {py:.4f} {w*scale:.4f} {h*scale:.4f} re")
    ops.append("f")

    if label and LABEL_PT > 0:
        ty = max(6.0, (oy - LABEL_PT * 1.6) / 2.0)
        ops += ["BT", f"/F1 {LABEL_PT:.1f} Tf",
                f"{ox:.4f} {ty:.4f} Td", f"({label}) Tj", "ET"]

    stream = "\n".join(ops).encode("ascii")

    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        (f"<</Type/Page/Parent 2 0 R/MediaBox[0 0 {sheet_w:.4f} {sheet_h:.4f}]"
         f"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>").encode("ascii"),
        b"<</Length " + str(len(stream)).encode() + b">>\nstream\n" + stream + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"

    xref = len(out)
    out += f"xref\n0 {len(objs)+1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<</Size {len(objs)+1}/Root 1 0 R>>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()

    with open(path, "wb") as f:
        f.write(out)
    return sheet_w * MM_PER_PT, sheet_h * MM_PER_PT, ox * MM_PER_PT, oy * MM_PER_PT


def verify(board, dictionary, mask):
    """Re-detect the pattern we are about to print — catches a bad spec before
    it costs a print run."""
    img = np.where(mask, 0, 255).astype(np.uint8)
    img = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(img)
    n_corners = 0 if ids is None else len(ids)
    expected = (SQUARES_X - 1) * (SQUARES_Y - 1)
    return n_corners, expected


def main():
    out_pdf = sys.argv[1] if len(sys.argv) > 1 else "charuco_board.pdf"
    board, dictionary = make_board()
    sq_px = square_px()
    mask  = rasterise(board, sq_px)
    rects = to_rectangles(mask)

    marker_mm = SQUARE_MM * MARKER_RATIO
    label = (f"ChArUco {SQUARES_X}x{SQUARES_Y}  square {SQUARE_MM:g} mm  "
             f"marker {marker_mm:g} mm  {DICT_NAME}  -  PRINT AT 100% SCALE, DO NOT FIT TO PAGE")

    sw, sh, mx, my = write_pdf(out_pdf, rects, sq_px, label)

    png = os.path.splitext(out_pdf)[0] + "_preview.png"
    cv2.imwrite(png, np.where(mask, 0, 255).astype(np.uint8))

    found, expected = verify(board, dictionary, mask)

    bits = dictionary.markerSize + 2
    print(f"[i] Sheet        : {sw:.1f} x {sh:.1f} mm  ({sw/25.4:.1f} x {sh/25.4:.1f} in)")
    print(f"[i] Pattern      : {SQUARES_X*SQUARE_MM:.0f} x {SQUARES_Y*SQUARE_MM:.0f} mm")
    print(f"[i] Margins      : {mx:.1f} mm sides, {my:.1f} mm top/bottom")
    print(f"[i] Square       : {SQUARE_MM:g} mm   marker {marker_mm:g} mm "
          f"({bits} bits, {marker_mm/bits:.2f} mm per bit)")
    print(f"[i] Corners      : {(SQUARES_X-1)*(SQUARES_Y-1)}   "
          f"markers {len(board.getIds())}")
    print(f"[i] Vector rects : {len(rects)}")
    print(f"[{'✓' if found == expected else '!'}] Self-check   : "
          f"detected {found}/{expected} ChArUco corners")
    print(f"[✓] Wrote '{out_pdf}'  and preview '{png}'")
    print()
    print("    Calibration settings for this board:")
    print(f"      CB_COLS = {SQUARES_X}   CB_ROWS = {SQUARES_Y}   "
          f"(squares, not corners)")
    print(f"      SQUARE_SIZE_M = {SQUARE_MM/1000:.4f}   "
          f"MARKER_SIZE_M = {marker_mm/1000:.4f}")
    print(f"      DICT = cv2.aruco.{DICT_NAME}")
    print("    Measure a printed square with calipers and use the real value.")


if __name__ == "__main__":
    main()
