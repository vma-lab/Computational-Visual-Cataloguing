import os
import json
import numpy as np
from PIL import Image, ImageDraw
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.mixture import GaussianMixture


# === CONFIG ===
image_folder = "img"
json_folder = "jsons"

output_folder = "ColoredName_CropedPatch_Output"
output_json_folder = "ColoredJson_Output"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_json_folder, exist_ok=True)

# Local background context around polygon bbox
LOCAL_BG_PADDING = 40


# ============================================================
# Manuscript-aware adaptive colour naming
# ============================================================

def relative_luminance_y(rgb):
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def rgb_to_lab_lch(rgb):
    r, g, b = rgb

    arr = np.uint8([[[r, g, b]]])

    lab = cv2.cvtColor(
        arr,
        cv2.COLOR_RGB2LAB
    )[0, 0].astype(float)

    L = lab[0] * 100.0 / 255.0
    a = lab[1] - 128.0
    b_lab = lab[2] - 128.0

    C = np.sqrt(a * a + b_lab * b_lab)

    h = np.degrees(
        np.arctan2(b_lab, a)
    )

    if h < 0:
        h += 360

    return L, C, h


def is_red_ink(rgb):
    """
    Strong red manuscript ink should be labelled red, not brown.
    """
    r, g, b = rgb
    L, C, h = rgb_to_lab_lch(rgb)

    strong_red_rgb = (
        r >= 150
        and r >= g + 55
        and r >= b + 55
        and abs(g - b) <= 22
    )

    red_hue = (
        h <= 35
        or h >= 345
    )

    red_orange_hue_with_strong_r = (
        20 <= h <= 55
        and r >= g + 55
        and r >= b + 55
        and abs(g - b) <= 25
    )

    return (
        C >= 20
        and (
            (strong_red_rgb and red_hue)
            or red_orange_hue_with_strong_r
            or strong_red_rgb
        )
    )


def is_dark_purple_ink(rgb):
    """
    Dark purple should remain purple, not black.
    """
    r, g, b = rgb
    L, C, h = rgb_to_lab_lch(rgb)

    return (
        255 <= h <= 345
        and C >= 8
        and b >= g + 10
        and r >= g + 8
        and max(rgb) >= 65
    )


def is_dark_blue_ink(rgb):
    """
    Dark / muted blue manuscript ink should be labelled blue, not black.
    """
    r, g, b = rgb
    L, C, h = rgb_to_lab_lch(rgb)

    max_c = max(rgb)
    min_c = min(rgb)
    chroma_rgb = max_c - min_c

    rgb_blue_dominant = (
        b >= g + 18
        and b >= r + 28
        and g >= r - 10
        and max_c >= 85
        and chroma_rgb >= 35
    )

    very_dark_blue = (
        b >= g + 25
        and b >= r + 35
        and max_c >= 88
        and chroma_rgb >= 38
    )

    lch_blue = (
        190 <= h <= 285
        and C >= 10
        and b > r
        and b > g
        and max_c >= 85
        and chroma_rgb >= 28
    )

    muted_blue_gray = (
        b >= g + 3
        and g >= r + 6
        and b >= r + 10
        and 70 <= max_c <= 115
        and 10 <= chroma_rgb <= 35
        and 190 <= h <= 260
    )

    dark_cyan_blue = (
        b >= g
        and g >= r + 8
        and b >= r + 12
        and 70 <= max_c <= 120
        and chroma_rgb >= 12
    )

    return (
        rgb_blue_dominant
        or very_dark_blue
        or lch_blue
        or muted_blue_gray
        or dark_cyan_blue
    )


def is_blue_black_ink(rgb):
    """
    Very dark blue that human annotators perceive as blue / blue-black.
    """
    r, g, b = rgb

    return (
        b >= 90
        and b >= g + 25
        and b >= r + 35
        and g >= r - 12
    )


def is_dark_olive_black_ink(rgb):
    """
    Dark olive/brownish neutral ink should be labelled black, not yellow.
    """
    r, g, b = rgb
    L, C, h = rgb_to_lab_lch(rgb)
    Y = relative_luminance_y(rgb)

    return (
        r >= g >= b
        and 35 <= (r - b) <= 50
        and 25 <= (g - b) <= 38
        and max(rgb) <= 95
        and Y <= 82
        and L <= 38
    )


def is_user_blue_ochre_ink(rgb):
    """
    User-corrected blue group.

    Expected blue:
        123-115-83
        122-115-85
        123-117-89
        124-117-85
        120-114-83
    """
    r, g, b = rgb

    return (
        118 <= r <= 126
        and 112 <= g <= 119
        and 80 <= b <= 91
        and 30 <= (r - b) <= 45
        and 25 <= (g - b) <= 38
    )


def is_pale_bluish_gray(rgb):
    """
    Pale blue-gray / cool gray-blue should be labelled blue.

    Note:
    This function is kept, but the near-neutral blue-gray rule is made
    slightly stricter so 148-151-150 can be handled by gray if needed.
    """
    r, g, b = rgb
    L, C, h = rgb_to_lab_lch(rgb)

    strong_pale_blue_gray = (
        max(g, b) > r
        and (max(g, b) - r) >= 10
        and b >= g - 4
    )

    subtle_cool_blue_gray = (
        135 <= r <= 155
        and 138 <= g <= 155
        and 140 <= b <= 158
        and b >= r + 4
        and g >= r + 2
        and b >= g - 2
        and C <= 10
    )

    near_neutral_blue_gray = (
        135 <= r <= 155
        and 135 <= g <= 155
        and 135 <= b <= 158
        and b >= r + 5
        and b >= g
        and C <= 9
    )

    return (
        strong_pale_blue_gray
        or subtle_cool_blue_gray
        or near_neutral_blue_gray
    )


def is_pale_warm_gray_ink(rgb):
    """
    Pale / medium warm yellow-gray should be labelled gray, not yellow.

    This is intentionally broad because many manuscript colours are between
    yellow and gray, but human annotation prefers gray.
    """
    r, g, b = rgb
    L, C, h = rgb_to_lab_lch(rgb)

    # Do not steal user-defined blue group.
    if is_user_blue_ochre_ink(rgb):
        return False

    explicit_gray_cases = (
        # Cool / neutral gray-green / gray-blue near-neutral examples
        (
            140 <= r <= 156
            and 140 <= g <= 156
            and 140 <= b <= 155
            and abs(r - g) <= 8
            and abs(g - b) <= 12
        )

        # Earlier medium warm-gray cases
        or (
            145 <= r <= 150
            and 140 <= g <= 146
            and 105 <= b <= 113
        )
        or (
            124 <= r <= 132
            and 118 <= g <= 128
            and 84 <= b <= 96
        )
        or (
            114 <= r <= 118
            and 110 <= g <= 116
            and 80 <= b <= 90
        )

        # New gray groups from current yellow false positives
        or (
            108 <= r <= 118
            and 103 <= g <= 110
            and 86 <= b <= 92
        )
        or (
            118 <= r <= 142
            and 114 <= g <= 139
            and 95 <= b <= 120
            and 15 <= (r - b) <= 35
            and 10 <= (g - b) <= 30
        )
        or (
            95 <= r <= 110
            and 90 <= g <= 102
            and 75 <= b <= 86
            and 15 <= (r - b) <= 25
            and 10 <= (g - b) <= 22
        )
        or (
            145 <= r <= 156
            and 145 <= g <= 156
            and 140 <= b <= 150
            and abs(r - g) <= 6
            and 0 <= (g - b) <= 12
        )

        # Previous warm gray / yellowish gray cases
        or (
            160 <= r <= 176
            and 154 <= g <= 168
            and 118 <= b <= 132
            and 34 <= (r - b) <= 48
            and 28 <= (g - b) <= 42
        )
        or (
            150 <= r <= 160
            and 144 <= g <= 152
            and 108 <= b <= 116
            and 38 <= (r - b) <= 48
            and 32 <= (g - b) <= 42
        )
        or (
            135 <= r <= 150
            and 128 <= g <= 142
            and 92 <= b <= 106
            and 35 <= (r - b) <= 48
            and 30 <= (g - b) <= 42
        )
    )

    if explicit_gray_cases:
        return True

    light_warm_gray = (
        r >= g >= b
        and 18 <= (r - b) <= 50
        and 12 <= (g - b) <= 44
        and L >= 45
        and C <= 32
        and 65 <= h <= 115
    )

    medium_warm_gray = (
        r >= g >= b
        and 15 <= (r - b) <= 45
        and 10 <= (g - b) <= 38
        and 95 <= max(rgb) <= 180
        and C <= 32
        and 65 <= h <= 115
    )

    return light_warm_gray or medium_warm_gray


def is_yellow_olive_ink(rgb):
    """
    Yellow / olive-yellow manuscript ink.

    Important:
    This rule is now stricter, because many yellow-like RGB values
    are labelled gray by human perception in this dataset.
    """
    r, g, b = rgb
    L, C, h = rgb_to_lab_lch(rgb)

    if is_dark_olive_black_ink(rgb):
        return False

    if is_user_blue_ochre_ink(rgb):
        return False

    if is_pale_bluish_gray(rgb):
        return False

    if is_pale_warm_gray_ink(rgb):
        return False

    yellow_olive_strong = (
        r >= g >= b
        and (r - b) >= 45
        and (g - b) >= 28
        and C >= 18
        and 65 <= h <= 115
        and max(rgb) >= 115
        and L < 70
    )

    yellow_brown_strong = (
        r >= g >= b
        and (r - b) >= 55
        and (g - b) >= 35
        and C >= 20
        and 55 <= h <= 110
        and max(rgb) >= 120
    )

    return yellow_olive_strong or yellow_brown_strong


def is_warm_gray(rgb):
    r, g, b = rgb

    return (
        r >= g >= b
        and (r - b) <= 45
        and abs(r - g) <= 20
    )


def is_dark_neutral_black_ink(rgb):
    """
    Dark neutral / warm low-saturation manuscript ink should be black,
    not gray/yellow.
    """
    r, g, b = rgb

    L, C, h = rgb_to_lab_lch(rgb)
    Y = relative_luminance_y(rgb)

    max_c = max(rgb)
    min_c = min(rgb)
    chroma_rgb = max_c - min_c

    dark_neutral = (
        max_c <= 85
        and Y <= 78
        and chroma_rgb <= 28
    )

    return (
        (dark_neutral or is_dark_olive_black_ink(rgb))
        and not is_red_ink(rgb)
        and not is_dark_purple_ink(rgb)
        and not is_dark_blue_ink(rgb)
        and not is_blue_black_ink(rgb)
        and not is_user_blue_ochre_ink(rgb)
        and not is_pale_bluish_gray(rgb)
        and not is_yellow_olive_ink(rgb)
        and not is_pale_warm_gray_ink(rgb)
    )


def is_warm_dark_black_ink(rgb):
    """
    Warm dark brown/olive/neutral ink should be black.
    """
    r, g, b = rgb
    Y = relative_luminance_y(rgb)
    max_c = max(rgb)

    return (
        max_c <= 70
        and Y <= 68
        and r >= g >= b
        and not is_red_ink(rgb)
        and not is_dark_purple_ink(rgb)
        and not is_dark_blue_ink(rgb)
        and not is_blue_black_ink(rgb)
        and not is_user_blue_ochre_ink(rgb)
        and not is_pale_bluish_gray(rgb)
        and not is_yellow_olive_ink(rgb)
        and not is_pale_warm_gray_ink(rgb)
    )


def contextual_dark_ink_is_black(text_rgb, bg_rgb):
    """
    Context-aware black rule:
    If text is much darker than local paper/background,
    human annotators usually perceive it as black ink.

    Important:
    visible red/blue/purple/yellow-gray groups should not be swallowed by black.
    """
    text_y = relative_luminance_y(text_rgb)
    bg_y = relative_luminance_y(bg_rgb)

    contrast = bg_y - text_y

    text_L, text_C, text_h = rgb_to_lab_lch(text_rgb)

    max_c = max(text_rgb)
    min_c = min(text_rgb)
    chroma_rgb = max_c - min_c

    if is_red_ink(text_rgb):
        return False

    if is_dark_purple_ink(text_rgb):
        return False

    if is_dark_blue_ink(text_rgb) or is_blue_black_ink(text_rgb):
        return False

    if is_user_blue_ochre_ink(text_rgb):
        return False

    if is_pale_bluish_gray(text_rgb):
        return False

    if is_dark_olive_black_ink(text_rgb):
        return True

    if is_pale_warm_gray_ink(text_rgb):
        return False

    if is_yellow_olive_ink(text_rgb):
        return False

    if contrast >= 55 and text_y <= 105 and text_C < 35:
        return True

    if contrast >= 35 and text_y <= 85 and chroma_rgb <= 35:
        return True

    if is_dark_neutral_black_ink(text_rgb):
        return True

    if is_warm_dark_black_ink(text_rgb):
        return True

    return False


def human_colour_name_contextual(text_rgb, bg_rgb=None):
    """
    Human/manuscript-aware colour label.
    """
    text_rgb = tuple(int(x) for x in text_rgb)

    if bg_rgb is not None:
        bg_rgb = tuple(int(x) for x in bg_rgb)

    r, g, b = text_rgb

    L, C, h = rgb_to_lab_lch(text_rgb)

    # 1. White / very light.
    if L > 88 and C < 10:
        return "white"

    # 2. Preserve visible chromatic ink before black/brown/yellow.
    if is_red_ink(text_rgb):
        return "red"

    if is_dark_purple_ink(text_rgb):
        return "purple"

    if is_dark_blue_ink(text_rgb) or is_blue_black_ink(text_rgb):
        return "blue"

    # User-corrected blue group.
    if is_user_blue_ochre_ink(text_rgb):
        return "blue"

    # Pale cool blue-gray.
    if is_pale_bluish_gray(text_rgb):
        return "blue"

    # Special dark olive/brownish black ink.
    if is_dark_olive_black_ink(text_rgb):
        return "black"

    # Gray must be checked before yellow for warm gray cases.
    if is_pale_warm_gray_ink(text_rgb):
        return "gray"

    if is_yellow_olive_ink(text_rgb):
        return "yellow"

    # Context-aware black ink.
    if bg_rgb is not None:
        if contextual_dark_ink_is_black(text_rgb, bg_rgb):
            return "black"

    # Absolute dark ink fallback.
    if is_dark_neutral_black_ink(text_rgb):
        return "black"

    if is_warm_dark_black_ink(text_rgb):
        return "black"

    # Neutral gray.
    if C < 14:
        return "gray"

    if is_warm_gray(text_rgb) and C < 30:
        return "gray"

    # Hue-based coarse label.
    if h < 20 or h >= 345:
        return "red"

    if 20 <= h < 70:
        if r >= g + 45 and r >= b + 45:
            return "red"

        if L < 58:
            return "brown"

        return "orange"

    if 70 <= h < 115:
        return "yellow"

    if 115 <= h < 170:
        return "green"

    if 170 <= h < 255:
        return "blue"

    if 255 <= h < 335:
        return "purple"

    if 335 <= h < 355 and L > 55:
        return "pink"

    return "red"


# ============================================================
# Better binarization
# ============================================================

def sauvola_binarization(
    img_pil,
    win_list=(15, 31, 61),
    k_strict=0.08,
    k_loose=0.28,
    bernsen_contrast_min=12,
    post_open=3,
    post_close=3
):
    rgb = np.array(img_pil.convert("RGB"))

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    ).astype(np.float32)

    gray = gray / 255.0

    # Illumination correction + light Retinex.
    bg = cv2.GaussianBlur(gray, (0, 0), 51)
    bg = np.clip(bg, 1e-6, 1.0)

    norm = np.clip(
        (gray / bg) * 0.5 + 0.25,
        0.0,
        1.0
    )

    blur_r = cv2.GaussianBlur(norm, (0, 0), 5)

    retinex = np.log(norm + 1e-6) - np.log(blur_r + 1e-6)

    rmin, rmax = np.percentile(retinex, (1, 99))

    if rmax > rmin:
        retinex = (retinex - rmin) / (rmax - rmin)

    retinex = np.clip(retinex, 0, 1).astype(np.float32)

    # CLAHE.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enh = clahe.apply(
        (retinex * 255).astype(np.uint8)
    ).astype(np.float32) / 255.0

    def local_stats(img, w):
        k = (w, w)

        m = cv2.boxFilter(
            img,
            ddepth=-1,
            ksize=k,
            normalize=True
        )

        m2 = cv2.boxFilter(
            img * img,
            ddepth=-1,
            ksize=k,
            normalize=True
        )

        s = cv2.sqrt(
            np.maximum(m2 - m * m, 0)
        )

        st = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (w, w)
        )

        vmin = cv2.erode(img, st)
        vmax = cv2.dilate(img, st)

        return m, s, vmin, vmax

    def sauvola(img, w, k, R=0.5):
        m, s, _, _ = local_stats(img, w)
        T = m * (1.0 + k * ((s / R) - 1.0))
        return (img < T).astype(np.uint8)

    def wolf_jolion(img, w, k):
        m, s, vmin, _ = local_stats(img, w)
        R = np.max(s) + 1e-6
        T = m + k * ((s / R) - 1.0) * (m - vmin)
        return (img < T).astype(np.uint8)

    def bernsen(img, w, cmin):
        _, _, vmin, vmax = local_stats(img, w)
        C = vmax - vmin
        T = (vmax + vmin) * 0.5

        return (
            (img < T)
            & (C >= (cmin / 255.0))
        ).astype(np.uint8)

    H, W = enh.shape

    votes_strict = np.zeros((H, W), np.int16)
    votes_loose = np.zeros((H, W), np.int16)

    for w in win_list:
        b_s_sau = sauvola(enh, w, k_strict)
        b_s_wjf = wolf_jolion(enh, w, k=-0.2)
        b_s_ber = bernsen(enh, w, bernsen_contrast_min)

        votes_strict += b_s_sau + b_s_wjf + b_s_ber

        b_l_sau = sauvola(enh, w, k_loose)
        b_l_wjf = wolf_jolion(enh, w, k=+0.2)
        b_l_ber = bernsen(
            enh,
            w,
            max(8, bernsen_contrast_min - 4)
        )

        votes_loose += b_l_sau + b_l_wjf + b_l_ber

    strong = (votes_strict >= 4).astype(np.uint8)
    weak = (votes_loose >= 3).astype(np.uint8)

    k3 = np.ones((3, 3), np.uint8)

    strong_dil = cv2.dilate(
        strong,
        k3,
        1
    )

    fused = (
        (strong == 1)
        | ((weak == 1) & (strong_dil == 1))
    ).astype(np.uint8)

    if post_open > 0:
        fused = cv2.morphologyEx(
            fused,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (post_open, post_open)
            )
        )

    if post_close > 0:
        fused = cv2.morphologyEx(
            fused,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (post_close, post_close)
            )
        )

    dt = cv2.distanceTransform(
        fused,
        cv2.DIST_L2,
        3
    )

    sw = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0

    min_area = max(
        16,
        int(2.0 * sw * sw)
    )

    max_ar = 25

    num, lab, stats, _ = cv2.connectedComponentsWithStats(
        fused,
        connectivity=8
    )

    keep = np.zeros(num, np.uint8)
    keep[0] = 0

    for i in range(1, num):
        x, y, w, h, area = stats[i]

        ar = max(w / h, h / w) if h > 0 and w > 0 else 1

        if area >= min_area and ar <= max_ar:
            keep[i] = 1

    cleaned = keep[lab]

    bin_img = (1 - cleaned) * 255

    return bin_img.astype(np.uint8)


# ============================================================
# Local polygon mask / context extraction
# ============================================================

def get_local_crop_bounds_from_polygon(polygon, width, height, padding=40):
    xs = polygon[0::2]
    ys = polygon[1::2]

    x1 = int(max(0, np.floor(min(xs) - padding)))
    y1 = int(max(0, np.floor(min(ys) - padding)))
    x2 = int(min(width, np.ceil(max(xs) + padding)))
    y2 = int(min(height, np.ceil(max(ys) + padding)))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def extract_fg_bg_pixels_local(img_np, binary_img, polygon, padding=40):
    """
    Extract foreground text pixels and local background pixels
    around the polygon.

    Returns:
        fg_pixels: text pixels inside polygon and foreground mask
        bg_pixels: local background pixels outside polygon within bbox+padding
        full_mask_np: full image polygon mask, used for crop patch output
    """
    H, W = img_np.shape[:2]

    bounds = get_local_crop_bounds_from_polygon(
        polygon,
        W,
        H,
        padding=padding
    )

    if bounds is None:
        empty_mask = np.zeros((H, W), dtype=np.uint8)
        return (
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0, 3), dtype=np.uint8),
            empty_mask
        )

    x1, y1, x2, y2 = bounds

    local_img = img_np[y1:y2, x1:x2]
    local_binary = binary_img[y1:y2, x1:x2]

    local_polygon_xy = [
        (
            polygon[i] - x1,
            polygon[i + 1] - y1
        )
        for i in range(0, len(polygon), 2)
    ]

    local_mask_pil = Image.new(
        "L",
        (x2 - x1, y2 - y1),
        0
    )

    ImageDraw.Draw(local_mask_pil).polygon(
        local_polygon_xy,
        fill=1
    )

    local_mask = np.array(local_mask_pil).astype(bool)

    # Foreground/text inside polygon
    local_foreground = (local_binary == 0)
    fg_mask = local_mask & local_foreground
    fg_pixels = local_img[fg_mask]

    # Local background:
    # pixels inside bbox+padding, outside polygon.
    bg_mask = ~local_mask

    # Remove very dark pixels from local background,
    # so nearby text strokes do not contaminate bg median too much.
    local_gray = cv2.cvtColor(
        local_img,
        cv2.COLOR_RGB2GRAY
    )

    bg_mask = bg_mask & (local_gray > 80)

    bg_pixels = local_img[bg_mask]

    # Fallback if background is too small after filtering
    if len(bg_pixels) < 10:
        bg_pixels = local_img[~local_mask]

    # Full-size mask for saving crop patch
    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = local_mask.astype(np.uint8)

    return fg_pixels, bg_pixels, full_mask


# ============================================================
# GMM-based text colour extraction + contextual naming
# ============================================================

def get_text_color_via_gmm(fg_pixels, bg_pixels):
    if len(fg_pixels) < 10 or len(bg_pixels) < 10:
        return "unknown", np.array([0, 0, 0]), np.array([0, 0, 0])

    bg_median = np.median(
        bg_pixels,
        axis=0
    ).astype(int)

    try:
        gmm = GaussianMixture(
            n_components=2,
            covariance_type="tied",
            random_state=42
        )

        gmm.fit(fg_pixels)

        labels = gmm.predict(fg_pixels)
        centers = gmm.means_

        dists = np.linalg.norm(
            centers - bg_median,
            axis=1
        )

        text_cluster = np.argmax(dists)
        text_pixels = fg_pixels[labels == text_cluster]

    except Exception:
        text_pixels = fg_pixels

    if len(text_pixels) == 0:
        text_pixels = fg_pixels

    text_median = np.median(
        text_pixels,
        axis=0
    ).astype(int)

    color_name = human_colour_name_contextual(
        text_median,
        bg_median
    )

    return color_name, text_median, bg_median


# ============================================================
# Matching images and JSONs
# ============================================================

def get_image_files():
    valid_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

    if not os.path.exists(image_folder):
        return []

    return [
        f
        for f in os.listdir(image_folder)
        if f.lower().endswith(valid_exts)
    ]


def get_json_basenames():
    if not os.path.exists(json_folder):
        return set()

    return {
        os.path.splitext(f)[0]
        for f in os.listdir(json_folder)
        if f.lower().endswith(".json")
    }


def get_images_with_matching_json():
    """
    Only process images in image_folder if same base name exists in json_folder.

    Example:
        test_jpg/ABC.jpg + jsons/ABC.json -> process
        test_jpg/XYZ.jpg + no jsons/XYZ.json -> skip
    """
    image_files = get_image_files()
    json_basenames = get_json_basenames()

    selected_images = []
    skipped_images = []

    for image_name in image_files:
        image_base = os.path.splitext(image_name)[0]

        if image_base in json_basenames:
            selected_images.append(image_name)
        else:
            skipped_images.append(image_name)

    return selected_images, skipped_images


# ============================================================
# Process one image
# ============================================================

def process_image(image_name):
    image_base = os.path.splitext(image_name)[0]

    image_path = os.path.join(
        image_folder,
        image_name
    )

    json_path = os.path.join(
        json_folder,
        image_base + ".json"
    )

    if not os.path.exists(image_path):
        print(f"Warning: image not found: {image_path}")
        return

    if not os.path.exists(json_path):
        print(f"Warning: matching JSON not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img = Image.open(image_path).convert("RGB")
    binary_img = sauvola_binarization(img)
    img_np = np.array(img)

    for ann in data.get("annotations", []):
        polygon = ann.get(
            "segmentation",
            [[]]
        )[0]

        if len(polygon) < 6:
            continue

        fg_pixels, bg_pixels, mask_np = extract_fg_bg_pixels_local(
            img_np=img_np,
            binary_img=binary_img,
            polygon=polygon,
            padding=LOCAL_BG_PADDING
        )

        if len(fg_pixels) > 0:
            color_name, text_median, bg_median = get_text_color_via_gmm(
                fg_pixels,
                bg_pixels
            )
        else:
            color_name = "unknown"
            text_median = np.array([0, 0, 0])
            bg_median = np.array([0, 0, 0])

        color_code = "-".join(
            map(str, text_median)
        )

        bg_color_code = "-".join(
            map(str, bg_median)
        )

        safe_color_name = (
            color_name
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

        ann["color_name"] = color_name
        ann["color_code"] = color_code
        ann["bg_color_code"] = bg_color_code

        # Save cropped patch
        masked_img = np.ones_like(img_np) * 255
        masked_img[mask_np == 1] = img_np[mask_np == 1]

        patch = Image.fromarray(masked_img)

        bbox = Image.fromarray(
            (mask_np * 255).astype(np.uint8)
        ).getbbox()

        if bbox:
            cropped = patch.crop(bbox)

            patch_name = (
                f"{image_base}___"
                f"{ann['id']}_"
                f"{safe_color_name}_"
                f"{color_code}.png"
            )

            cropped.save(
                os.path.join(
                    output_folder,
                    patch_name
                )
            )

    with open(
        os.path.join(output_json_folder, image_base + ".json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if not os.path.exists(image_folder):
        print(f"Input image folder not found: {image_folder}")
        raise SystemExit

    if not os.path.exists(json_folder):
        print(f"Input JSON folder not found: {json_folder}")
        raise SystemExit

    selected_images, skipped_images = get_images_with_matching_json()

    print("Processing only images that have matching JSON files.")
    print(f"Image folder: {image_folder}")
    print(f"JSON folder: {json_folder}")
    print(f"Images found: {len(get_image_files())}")
    print(f"JSON files found: {len(get_json_basenames())}")
    print(f"Selected images to process: {len(selected_images)}")
    print(f"Skipped images without matching JSON: {len(skipped_images)}")
    print("Colour naming mode: manuscript-aware LOCAL contextual perception")
    print(f"Local background padding: {LOCAL_BG_PADDING}px")
    print("Dark blue / muted blue / user-blue-ochre / pale blue-gray / red / black-olive / warm-gray preservation: ON")
    print("Yellow rule: stricter, to reduce yellow-vs-gray false positives")

    if len(selected_images) == 0:
        print("No matching image-JSON pairs found.")
        raise SystemExit

    Parallel(
        n_jobs=max(2, os.cpu_count() - 2),
        backend="threading"
    )(
        delayed(process_image)(img_name)
        for img_name in tqdm(
            selected_images,
            desc="Processing",
            unit="image"
        )
    )

    print(f"Saved cropped patches to: {output_folder}")
    print(f"Saved updated JSON files to: {output_json_folder}")
    print("Processing complete.")
