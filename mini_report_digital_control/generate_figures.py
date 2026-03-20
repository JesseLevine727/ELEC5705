from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
FIGS = ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

WIDTH = 1800
HEIGHT = 900


def font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_BODY = font(30)
FONT_SMALL = font(24)
FONT_TITLE = font(42, bold=True)
FONT_SUBTITLE = font(34, bold=True)


def canvas(width=WIDTH, height=HEIGHT):
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    return img, draw


def centered_text(draw, x, y, text, fill="black", font_obj=FONT_BODY):
    box = draw.multiline_textbbox((0, 0), text, font=font_obj, align="center", spacing=6)
    w = box[2] - box[0]
    h = box[3] - box[1]
    draw.multiline_text((x - w / 2, y - h / 2), text, fill=fill, font=font_obj, align="center", spacing=6)


def box(draw, xy, text, fill, outline="#1f2937", width=4, text_font=FONT_BODY):
    draw.rounded_rectangle(xy, radius=22, fill=fill, outline=outline, width=width)
    x1, y1, x2, y2 = xy
    centered_text(draw, (x1 + x2) / 2, (y1 + y2) / 2, text, font_obj=text_font)


def arrow(draw, start, end, label=None, label_offset=(0, -32)):
    draw.line([start, end], fill="#0f172a", width=6)
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    mag = max((dx * dx + dy * dy) ** 0.5, 1)
    ux = dx / mag
    uy = dy / mag
    px = -uy
    py = ux
    tip = (x2, y2)
    left = (x2 - 28 * ux + 14 * px, y2 - 28 * uy + 14 * py)
    right = (x2 - 28 * ux - 14 * px, y2 - 28 * uy - 14 * py)
    draw.polygon([tip, left, right], fill="#0f172a")
    if label:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        centered_text(draw, mx, my, label, font_obj=FONT_SMALL)


def save(img, name):
    img.save(FIGS / name)


def project_scope():
    img, draw = canvas(1800, 700)
    centered_text(draw, 900, 75, "Project Boundary Relative to the Paper", font_obj=FONT_TITLE)
    box(draw, (80, 190, 520, 560), "Full paper:\n13-bit SAR ADC chip\nanalog + digital +\nmeasurements", "#fde68a")
    box(draw, (680, 190, 1120, 560), "Digital/control side:\nSense&Force\nextra-cycle control\nsign extraction\nLPF + registers", "#bfdbfe")
    box(draw, (1280, 190, 1720, 560), "FPGA/HLS project:\ncontroller + ADC/error model\nno transistor-level analog", "#c7f9cc")
    arrow(draw, (520, 375), (680, 375), "focus")
    arrow(draw, (1120, 375), (1280, 375), "maps well")
    save(img, "project_scope.png")


def digital_control_loop():
    img, draw = canvas(2200, 760)
    centered_text(draw, 1100, 70, "Digital/Control Calibration Loop", font_obj=FONT_TITLE)
    colors = ["#fff7d6", "#dbeafe", "#e9d5ff", "#fee2e2", "#dcfce7", "#fde68a"]
    labels = [
        "15-cycle\nredundant SAR\nconversion",
        "Sense&Force\npattern detect",
        "16th cycle:\ncode swap or\nmode switch",
        "Compare\nD15 vs D16\nextract sign",
        "LPF:\n6-bit up/down\ncounter",
        "Calibration\nregisters\nupdate",
    ]
    left = 70
    width = 290
    gap = 55
    top = 220
    for idx, (color, label) in enumerate(zip(colors, labels)):
        x1 = left + idx * (width + gap)
        box(draw, (x1, top, x1 + width, top + 250), label, color)
        if idx < len(labels) - 1:
            arrow(draw, (x1 + width, top + 125), (x1 + width + gap, top + 125))
    box(
        draw,
        (780, 560, 1420, 710),
        "Analog correction elements in the paper\n(DAC trim caps and comparator trim caps)",
        "#f0fdf4",
        text_font=FONT_SMALL,
    )
    arrow(draw, (2100, 470), (1420, 620), "control words", label_offset=(0, -20))
    save(img, "digital_control_loop.png")


def dac_sequence():
    img, draw = canvas(2200, 820)
    centered_text(draw, 1100, 70, "DAC Mismatch Calibration Event", font_obj=FONT_TITLE)
    labels = [
        "Normal conversion\nends at raw code A\nor B",
        "Trigger match:\nMSB ... MSB-4\npattern seen",
        "Force alternate\nredundant code\nA ↔ B",
        "Run 16th\ncomparison and\nobserve D16",
        "Use D15,\nD16 sign\nto step\ncal word",
    ]
    colors = ["#fff7d6", "#dbeafe", "#e9d5ff", "#fee2e2", "#dcfce7"]
    boxes = [
        (70, 220, 420, 470),
        (500, 220, 850, 470),
        (930, 220, 1280, 470),
        (1360, 220, 1710, 470),
        (1790, 220, 2080, 470),
    ]
    for b, label, color in zip(boxes, labels, colors):
        box(draw, b, label, color)
    for i in range(len(boxes) - 1):
        arrow(draw, (boxes[i][2], 345), (boxes[i + 1][0], 345))
    box(draw, (180, 560, 960, 720), "If A and B are truly equivalent, D15 and D16 agree.", "#f8fafc", text_font=FONT_SMALL)
    box(
        draw,
        (1160, 560, 2020, 720),
        "If mismatch makes V_A ≠ V_B, then D15 and D16 reveal\nwhich way the capacitor weight must move.",
        "#f8fafc",
        text_font=FONT_SMALL,
    )
    save(img, "dac_sequence.png")


def lpf_counter():
    img, draw = canvas(1900, 760)
    centered_text(draw, 950, 70, "Noise-Filtering LPF Used Before Register Updates", font_obj=FONT_TITLE)
    box(draw, (80, 280, 430, 500), "Calibration sign:\nINC_I or DEC_I", "#dbeafe")
    box(draw, (610, 220, 1070, 560), "6-bit bidirectional counter\naccumulates repeated\nsame-direction evidence", "#dcfce7")
    box(draw, (1290, 300, 1510, 470), "INC_O", "#fde68a")
    box(draw, (1600, 300, 1820, 470), "DEC_O", "#fde68a")
    arrow(draw, (430, 390), (610, 390))
    arrow(draw, (1070, 340), (1290, 340), "hit 63", label_offset=(0, -26))
    arrow(draw, (1070, 430), (1600, 430), "hit 0", label_offset=(0, -26))
    box(
        draw,
        (280, 610, 1620, 720),
        "Single noisy sign flips do not immediately change the calibration register.\nOnly persistent evidence creates an update pulse.",
        "#f8fafc",
        text_font=FONT_SMALL,
    )
    save(img, "lpf_counter.png")


def convergence():
    img, draw = canvas(1700, 900)
    plot_left, plot_top, plot_right, plot_bottom = 180, 160, 1540, 720
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="#111827", width=4)
    centered_text(draw, 850, 70, "Illustrative convergence trend based on the paper's reported ~400k-cycle settling", font_obj=FONT_SUBTITLE)
    centered_text(draw, 850, 815, "Calibration time (k cycles)", font_obj=FONT_BODY)
    centered_text(draw, 70, 440, "SNDR (dB)", font_obj=FONT_BODY)

    for i, x in enumerate(range(0, 401, 50)):
        px = plot_left + (plot_right - plot_left) * i / 8
        draw.line((px, plot_bottom, px, plot_bottom + 18), fill="#111827", width=3)
        centered_text(draw, px, plot_bottom + 45, str(x), font_obj=FONT_SMALL)
        if i < 8:
            draw.line((px, plot_top, px, plot_bottom), fill="#d1d5db", width=1)
    y_ticks = [54, 56, 58, 60, 62, 64]
    for yv in y_ticks:
        py = plot_bottom - (yv - 54) / 10 * (plot_bottom - plot_top)
        draw.line((plot_left - 18, py, plot_left, py), fill="#111827", width=3)
        centered_text(draw, plot_left - 55, py, str(yv), font_obj=FONT_SMALL)
        draw.line((plot_left, py, plot_right, py), fill="#d1d5db", width=1)

    xs = [0, 50, 100, 150, 200, 250, 300, 350, 400]
    ys = [55.2, 57.1, 59.0, 60.7, 62.0, 63.0, 63.6, 64.0, 64.1]
    points = []
    for xv, yv in zip(xs, ys):
        px = plot_left + (xv / 400) * (plot_right - plot_left)
        py = plot_bottom - ((yv - 54) / 10) * (plot_bottom - plot_top)
        points.append((px, py))
    draw.line(points, fill="#0f766e", width=6)
    for px, py in points:
        draw.ellipse((px - 9, py - 9, px + 9, py + 9), fill="#0f766e", outline="#0f766e")
    save(img, "convergence_trend.png")


def main():
    project_scope()
    digital_control_loop()
    dac_sequence()
    lpf_counter()
    convergence()


if __name__ == "__main__":
    main()
