import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import pygame
import numpy as np
import sys
import os
import random
import time
import urllib.request

# --- Download hand landmark model if not present ---
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model (~26MB)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    print("Model downloaded.")

# --- MediaPipe setup ---
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)
detector = vision.HandLandmarker.create_from_options(options)

# --- Webcam ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- Layout ---
GAME_W  = 800
GAME_H  = 600
DBG_W   = 640
DBG_H   = 360
PANEL_W = 660
TOTAL_W = GAME_W + PANEL_W
HEIGHT  = GAME_H

# --- Pygame ---
pygame.init()
screen     = pygame.display.set_mode((TOTAL_W, HEIGHT))
pygame.display.set_caption("Whack-a-Mole  |  Debug View ->")
clock      = pygame.time.Clock()
font_huge  = pygame.font.SysFont(None, 100)
font_large = pygame.font.SysFont(None, 72)
font_med   = pygame.font.SysFont(None, 42)
font_small = pygame.font.SysFont(None, 32)
font_tiny  = pygame.font.SysFont(None, 24)

# --- Colours ---
WHITE   = (255, 255, 255)
BLACK   = (0,   0,   0)
GREY    = (180, 180, 180)
DGREY   = (60,  60,  60)
GREEN   = (0,   210, 90)
YELLOW  = (255, 220, 0)
RED     = (220, 60,  60)
BROWN   = (101, 67,  33)
L_BROWN = (150, 100, 55)
DIRT    = (82,  54,  25)
SKIN    = (255, 200, 130)
NOSE    = (220, 150, 90)
BG_TOP  = (95,  175, 95)   # grass green
BG_BOT  = (70,  130, 60)

# Debug annotation colours (RGB — drawn on RGB numpy frame)
DBG_LEFT_RGB  = (100, 255, 100)
DBG_RIGHT_RGB = (255, 220,   0)
DBG_PIVOT_RGB = (255,  60,  60)

# --- Grid layout ---
COLS        = 4
ROWS        = 3
MOLE_RADIUS = 48          # hit radius (also used for drawing)
HOLE_RADIUS = 52

# Compute hole centres, centred in the game area with some top margin
TOP_MARGIN  = 110         # space for score / timer header
SIDE_MARGIN = 80
cell_w = (GAME_W - 2 * SIDE_MARGIN) // COLS
cell_h = (GAME_H - TOP_MARGIN - 40) // ROWS
HOLES = [
    (SIDE_MARGIN + cell_w * c + cell_w // 2,
     TOP_MARGIN  + cell_h * r + cell_h // 2)
    for r in range(ROWS)
    for c in range(COLS)
]                          # 12 holes total

# --- Game config ---
GAME_DURATION   = 60       # seconds
INITIAL_MOLE_MS = 1800     # how long a mole stays up (ms)
MIN_MOLE_MS     = 600      # floor — never faster than this
SPEED_RAMP      = 15       # every N whacks, speed increases
MAX_ACTIVE      = 3        # max moles visible at once
POINTS_HIT      = 10
POINTS_MISS     = -5       # penalty for whacking empty hole

# --- Smoothing ---
SMOOTH_N = 5
hand_histories = {'Left': [], 'Right': []}

def smooth(history, new_val, n=SMOOTH_N):
    history.append(new_val)
    if len(history) > n:
        history.pop(0)
    return (int(sum(x for x, _ in history) / len(history)),
            int(sum(y for _, y in history) / len(history)))

# --- MediaPipe skeleton connections ---
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# -----------------------------------------------------------------------
# Detection
# -----------------------------------------------------------------------

def get_hand_positions(frame):
    """
    Returns:
      positions  — dict  label -> (px, py)  using landmark 9 (palm centre)
      result     — raw MediaPipe result (for debug drawing)
      rgb_frame  — flipped RGB numpy array
    """
    flipped  = cv2.flip(frame, 1)
    rgb      = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)

    positions = {}
    if result.hand_landmarks and result.handedness:
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label = handedness[0].display_name
            x = int(landmarks[9].x * GAME_W)
            y = int(landmarks[9].y * GAME_H)
            positions[label] = (x, y)

    return positions, result, rgb

# -----------------------------------------------------------------------
# Debug surface
# -----------------------------------------------------------------------

def build_debug_surface(rgb_frame, result, hand_positions):
    debug = rgb_frame.copy()
    fh, fw = debug.shape[:2]

    if result.hand_landmarks and result.handedness:
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label    = handedness[0].display_name
            bone_col = DBG_LEFT_RGB if label == 'Left' else DBG_RIGHT_RGB
            pts = [(int(lm.x * fw), int(lm.y * fh)) for lm in landmarks]

            for a, b in HAND_CONNECTIONS:
                cv2.line(debug, pts[a], pts[b], bone_col, 2, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(debug, pt, 4, bone_col, -1, cv2.LINE_AA)

            # Landmark 9 — control point
            cv2.circle(debug, pts[9], 11, DBG_PIVOT_RGB,  -1, cv2.LINE_AA)
            cv2.circle(debug, pts[9], 11, (255,255,255),   2, cv2.LINE_AA)

            wx, wy = pts[0]
            cv2.putText(debug,
                        f"{label}  ({landmarks[9].x:.2f}, {landmarks[9].y:.2f})",
                        (wx - 30, wy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bone_col, 2, cv2.LINE_AA)

    resized = cv2.resize(debug, (DBG_W, DBG_H), interpolation=cv2.INTER_AREA)
    return pygame.surfarray.make_surface(resized.swapaxes(0, 1))

# -----------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------

def draw_divider():
    pygame.draw.line(screen, DGREY, (GAME_W, 0), (GAME_W, HEIGHT), 2)

def draw_debug_panel(debug_surf, hand_positions):
    px    = GAME_W + 10
    cam_y = (HEIGHT - DBG_H) // 2 - 20

    lbl = font_tiny.render("Camera Feed  -  MediaPipe Detection", True, GREY)
    screen.blit(lbl, (px, cam_y - 22))

    screen.blit(debug_surf, (px, cam_y))
    pygame.draw.rect(screen, DGREY, (px - 1, cam_y - 1, DBG_W + 2, DBG_H + 2), 1)

    legend_y = cam_y + DBG_H + 12
    for colour, text in [
        (DBG_PIVOT_RGB, "Red dot  =  Landmark 9  (palm centre / hit point)"),
        (DBG_LEFT_RGB,  "Green    =  Left hand"),
        (DBG_RIGHT_RGB, "Yellow   =  Right hand"),
    ]:
        screen.blit(font_tiny.render(text, True, colour), (px, legend_y))
        legend_y += 20

    legend_y += 8
    for label, colour in [('Left', GREEN), ('Right', YELLOW)]:
        if label in hand_positions:
            hx, hy = hand_positions[label]
            txt    = f"{label}:  px ({hx}, {hy})"
        else:
            txt    = f"{label}:  -- not detected --"
            colour = DGREY
        screen.blit(font_tiny.render(txt, True, colour), (px, legend_y))
        legend_y += 20

# -----------------------------------------------------------------------
# Game drawing
# -----------------------------------------------------------------------

def draw_background():
    # Simple two-tone grass background
    screen.fill(BG_TOP, (0, TOP_MARGIN, GAME_W, GAME_H - TOP_MARGIN))
    screen.fill((30, 20, 10), (0, 0, GAME_W, TOP_MARGIN))   # dark header bar

def draw_hole(cx, cy):
    """Draw a dirt hole."""
    pygame.draw.ellipse(screen, DIRT,
                        (cx - HOLE_RADIUS, cy - HOLE_RADIUS // 2,
                         HOLE_RADIUS * 2,  HOLE_RADIUS))
    pygame.draw.ellipse(screen, BLACK,
                        (cx - HOLE_RADIUS + 6, cy - HOLE_RADIUS // 2 + 4,
                         (HOLE_RADIUS - 6) * 2, HOLE_RADIUS - 8))

def draw_mole(cx, cy, progress, whacked=False):
    """
    Draw a mole rising from hole cy.
    progress 0.0 = fully hidden, 1.0 = fully visible.
    whacked = show stars / dazed face briefly.
    """
    # How far above the hole centre the mole body appears
    rise   = int(MOLE_RADIUS * 1.6 * progress)
    body_y = cy - rise + HOLE_RADIUS // 4

    # Cover lower part with dirt ellipse (so mole looks like it's inside hole)
    # Body
    body_col = BROWN if not whacked else (200, 80, 80)
    pygame.draw.circle(screen, body_col, (cx, body_y), MOLE_RADIUS)
    pygame.draw.circle(screen, L_BROWN, (cx, body_y), MOLE_RADIUS, 3)

    # Face
    if not whacked:
        # Eyes
        pygame.draw.circle(screen, BLACK,  (cx - 14, body_y - 10), 6)
        pygame.draw.circle(screen, BLACK,  (cx + 14, body_y - 10), 6)
        pygame.draw.circle(screen, WHITE,  (cx - 12, body_y - 12), 2)
        pygame.draw.circle(screen, WHITE,  (cx + 16, body_y - 12), 2)
        # Nose
        pygame.draw.ellipse(screen, NOSE,
                            (cx - 8, body_y - 4, 16, 10))
        # Smile
        pygame.draw.arc(screen, BLACK,
                        (cx - 10, body_y, 20, 12), 3.14, 0, 3)
    else:
        # X eyes when whacked
        for dx in [-14, 14]:
            ex, ey = cx + dx, body_y - 10
            pygame.draw.line(screen, BLACK, (ex-5,ey-5), (ex+5,ey+5), 3)
            pygame.draw.line(screen, BLACK, (ex+5,ey-5), (ex-5,ey+5), 3)
        # Wavy mouth
        pygame.draw.arc(screen, BLACK,
                        (cx - 10, body_y + 2, 20, 12), 0, 3.14, 3)

    # Redraw dirt in front of the lower half so mole looks submerged
    pygame.draw.ellipse(screen, DIRT,
                        (cx - HOLE_RADIUS, cy - HOLE_RADIUS // 2,
                         HOLE_RADIUS * 2,  HOLE_RADIUS))

def draw_cursor(pos, colour, label):
    """Draw a crosshair + hand label at the given game position."""
    if pos is None:
        return
    x, y = pos
    size = 22
    pygame.draw.line(screen, colour, (x - size, y), (x + size, y), 3)
    pygame.draw.line(screen, colour, (x, y - size), (x, y + size), 3)
    pygame.draw.circle(screen, colour, (x, y), 8, 3)
    lbl = font_tiny.render(label, True, colour)
    screen.blit(lbl, (x + 12, y - 18))

def draw_hit_feedback(feedbacks):
    """
    feedbacks: list of (x, y, text, colour, born_time)
    Floats the text upward and fades it out over ~0.6 s.
    """
    now   = time.time()
    alive = []
    for (x, y, txt, col, born) in feedbacks:
        age = now - born
        if age > 0.6:
            continue
        alpha  = max(0, 1.0 - age / 0.6)
        offset = int(age * 80)
        surf   = font_med.render(txt, True, col)
        surf.set_alpha(int(alpha * 255))
        screen.blit(surf, (x - surf.get_width() // 2, y - offset))
        alive.append((x, y, txt, col, born))
    return alive

def draw_header(score, time_left, whack_count):
    # Background already dark from draw_background
    # Score
    sc_surf = font_large.render(str(score), True, YELLOW)
    screen.blit(sc_surf, (GAME_W // 2 - sc_surf.get_width() // 2, 10))

    label_surf = font_tiny.render("SCORE", True, GREY)
    screen.blit(label_surf, (GAME_W // 2 - label_surf.get_width() // 2, 74))

    # Timer bar
    bar_w   = 340
    bar_h   = 18
    bar_x   = GAME_W // 2 - bar_w // 2
    bar_y   = 90
    ratio   = max(0, time_left / GAME_DURATION)
    bar_col = GREEN if ratio > 0.4 else (YELLOW if ratio > 0.2 else RED)
    pygame.draw.rect(screen, DGREY,   (bar_x, bar_y, bar_w, bar_h), border_radius=9)
    pygame.draw.rect(screen, bar_col, (bar_x, bar_y, int(bar_w * ratio), bar_h), border_radius=9)
    pygame.draw.rect(screen, GREY,    (bar_x, bar_y, bar_w, bar_h), 2, border_radius=9)

    t_surf = font_tiny.render(f"{int(time_left)}s", True, GREY)
    screen.blit(t_surf, (bar_x + bar_w + 8, bar_y))

    # Whack count (speed indicator)
    spd_txt = font_tiny.render(f"Whacks: {whack_count}", True, GREY)
    screen.blit(spd_txt, (18, 12))

def draw_start_screen(debug_surf, hand_positions):
    screen.fill(BLACK)
    t1 = font_large.render("WHACK-A-MOLE", True, WHITE)
    t2 = font_med.render("Move your hand over a mole to whack it!", True, GREY)
    t3 = font_small.render("Show at least one hand to start", True, GREEN)
    t4 = font_tiny.render(f"First to {GAME_DURATION}s  |  +{POINTS_HIT} pts per mole  |  {POINTS_MISS} for misses", True, DGREY)
    screen.blit(t1, (GAME_W // 2 - t1.get_width() // 2, 160))
    screen.blit(t2, (GAME_W // 2 - t2.get_width() // 2, 270))
    screen.blit(t3, (GAME_W // 2 - t3.get_width() // 2, 330))
    screen.blit(t4, (GAME_W // 2 - t4.get_width() // 2, 390))
    draw_divider()
    draw_debug_panel(debug_surf, hand_positions)

def draw_end_screen(score, debug_surf, hand_positions):
    screen.fill(BLACK)
    t1 = font_large.render("TIME'S UP!", True, RED)
    t2 = font_huge.render(str(score), True, YELLOW)
    t3 = font_small.render("FINAL SCORE", True, GREY)
    t4 = font_small.render("SPACE to play again   |   ESC to quit", True, DGREY)
    screen.blit(t1, (GAME_W // 2 - t1.get_width() // 2, 130))
    screen.blit(t2, (GAME_W // 2 - t2.get_width() // 2, 220))
    screen.blit(t3, (GAME_W // 2 - t3.get_width() // 2, 330))
    screen.blit(t4, (GAME_W // 2 - t4.get_width() // 2, 400))
    draw_divider()
    draw_debug_panel(debug_surf, hand_positions)

# -----------------------------------------------------------------------
# Mole class
# -----------------------------------------------------------------------

class Mole:
    # States
    RISING   = "rising"
    UP       = "up"
    SINKING  = "sinking"
    HIDDEN   = "hidden"
    WHACKED  = "whacked"

    RISE_MS  = 200    # animation duration up/down
    SINK_MS  = 200
    WHACK_MS = 300    # brief dazed pause before sinking

    def __init__(self, hole_index, mole_ms):
        self.hole    = hole_index
        self.cx, self.cy = HOLES[hole_index]
        self.mole_ms = mole_ms
        self.state   = self.RISING
        self.born    = pygame.time.get_ticks()
        self.progress = 0.0    # 0 = hidden, 1 = fully up

    def update(self):
        now    = pygame.time.get_ticks()
        age    = now - self.born

        if self.state == self.RISING:
            self.progress = min(1.0, age / self.RISE_MS)
            if self.progress >= 1.0:
                self.state = self.UP
                self.born  = now

        elif self.state == self.UP:
            if age >= self.mole_ms:
                self.state = self.SINKING
                self.born  = now

        elif self.state == self.WHACKED:
            self.progress = 1.0
            if age >= self.WHACK_MS:
                self.state = self.SINKING
                self.born  = now

        elif self.state == self.SINKING:
            self.progress = max(0.0, 1.0 - age / self.SINK_MS)
            if self.progress <= 0.0:
                self.state = self.HIDDEN

    def is_alive(self):
        return self.state != self.HIDDEN

    def whack(self):
        if self.state in (self.RISING, self.UP):
            self.state = self.WHACKED
            self.born  = pygame.time.get_ticks()
            return True
        return False

    def draw(self):
        if self.state == self.HIDDEN:
            return
        draw_mole(self.cx, self.cy, self.progress,
                  whacked=(self.state == self.WHACKED))

# -----------------------------------------------------------------------
# Game state
# -----------------------------------------------------------------------
STATE_START, STATE_PLAYING, STATE_END = "start", "playing", "end"

def new_game():
    return {
        "state":        STATE_START,
        "score":        0,
        "whack_count":  0,
        "moles":        [],
        "active_holes": set(),
        "next_mole_t":  0,
        "start_time":   0.0,
        "feedbacks":    [],
        "smoothed":     {'Left': None, 'Right': None},
        "prev_pos":     {'Left': None, 'Right': None},
    }

G = new_game()

blank_debug  = pygame.Surface((DBG_W, DBG_H))
blank_debug.fill((30, 30, 30))
last_debug   = blank_debug
last_hands   = {}

def current_mole_ms(whack_count):
    """Speed increases every SPEED_RAMP whacks, down to MIN_MOLE_MS."""
    reduction = (whack_count // SPEED_RAMP) * 150
    return max(MIN_MOLE_MS, INITIAL_MOLE_MS - reduction)

# -----------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------
while True:
    dt  = clock.tick(60)
    now = pygame.time.get_ticks()

    # --- Events ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                cap.release()
                pygame.quit()
                sys.exit()
            if event.key == pygame.K_SPACE and G["state"] == STATE_END:
                G = new_game()

    # --- Camera + detection ---
    ret, frame = cap.read()
    if not ret:
        pygame.display.flip()
        continue

    raw_hands, mp_result, rgb_frame = get_hand_positions(frame)
    last_debug = build_debug_surface(rgb_frame, mp_result, raw_hands)
    last_hands = raw_hands

    # Smooth hand positions
    for label in ('Left', 'Right'):
        if label in raw_hands:
            G["smoothed"][label] = smooth(
                hand_histories[label], raw_hands[label])
        else:
            hand_histories[label].clear()
            G["smoothed"][label] = None

    # --- State: START ---
    if G["state"] == STATE_START:
        draw_start_screen(last_debug, last_hands)
        pygame.display.flip()
        if raw_hands:   # any hand detected → begin
            G["state"]      = STATE_PLAYING
            G["start_time"] = time.time()
            G["next_mole_t"] = now + 300
        continue

    # --- State: END ---
    if G["state"] == STATE_END:
        draw_end_screen(G["score"], last_debug, last_hands)
        pygame.display.flip()
        continue

    # --- State: PLAYING ---

    time_left = GAME_DURATION - (time.time() - G["start_time"])
    if time_left <= 0:
        G["state"] = STATE_END
        continue

    # Spawn new moles
    if (now >= G["next_mole_t"] and
            len([m for m in G["moles"] if m.is_alive()]) < MAX_ACTIVE):
        free = [i for i in range(len(HOLES)) if i not in G["active_holes"]]
        if free:
            idx  = random.choice(free)
            ms   = current_mole_ms(G["whack_count"])
            mole = Mole(idx, ms)
            G["moles"].append(mole)
            G["active_holes"].add(idx)
            # Next spawn: between 0.5× and 1× of current mole_ms
            G["next_mole_t"] = now + random.randint(ms // 2, ms)

    # Update moles; remove hidden ones
    alive = []
    for m in G["moles"]:
        m.update()
        if m.is_alive():
            alive.append(m)
        else:
            G["active_holes"].discard(m.hole)
    G["moles"] = alive

    # Hit detection — palm (landmark 9) within MOLE_RADIUS of mole centre
    # Track whether this frame is a "new" hit (hand moved onto mole)
    for label, colour in [('Left', GREEN), ('Right', YELLOW)]:
        pos     = G["smoothed"][label]
        prev    = G["prev_pos"][label]

        if pos is None:
            G["prev_pos"][label] = None
            continue

        hx, hy  = pos
        hit_any = False

        for m in G["moles"]:
            dist = ((hx - m.cx) ** 2 + (hy - m.cy) ** 2) ** 0.5
            if dist <= MOLE_RADIUS:
                # Only register a new hit if hand just entered this radius
                # (i.e. was NOT already inside it last frame)
                if prev is not None:
                    px, py = prev
                    prev_dist = ((px - m.cx) ** 2 + (py - m.cy) ** 2) ** 0.5
                else:
                    prev_dist = MOLE_RADIUS + 1   # treat as just-entered

                if prev_dist > MOLE_RADIUS:
                    if m.whack():
                        G["score"]       += POINTS_HIT
                        G["whack_count"] += 1
                        G["feedbacks"].append(
                            (m.cx, m.cy - MOLE_RADIUS - 10,
                             f"+{POINTS_HIT}", GREEN, time.time()))
                        hit_any = True

        # Miss penalty: hand entered a hole area with no mole
        if not hit_any and prev is not None:
            px, py = prev
            for (hcx, hcy) in HOLES:
                dist      = ((hx  - hcx) ** 2 + (hy  - hcy) ** 2) ** 0.5
                prev_dist = ((px  - hcx) ** 2 + (py  - hcy) ** 2) ** 0.5
                if dist <= HOLE_RADIUS and prev_dist > HOLE_RADIUS:
                    # Check no mole is there
                    mole_there = any(
                        m.hole == HOLES.index((hcx, hcy)) and
                        m.state in (Mole.RISING, Mole.UP)
                        for m in G["moles"])
                    if not mole_there:
                        G["score"] += POINTS_MISS
                        G["feedbacks"].append(
                            (hcx, hcy - HOLE_RADIUS - 10,
                             str(POINTS_MISS), RED, time.time()))

        G["prev_pos"][label] = pos

    # --- Draw ---
    draw_background()
    draw_header(G["score"], time_left, G["whack_count"])

    # Holes (back layer)
    for (hx, hy) in HOLES:
        draw_hole(hx, hy)

    # Moles
    for m in G["moles"]:
        m.draw()

    # Cursors
    for label, colour in [('Left', GREEN), ('Right', YELLOW)]:
        pos = G["smoothed"][label]
        if pos:
            draw_cursor(pos, colour, label)

    # Floating hit/miss feedback
    G["feedbacks"] = draw_hit_feedback(G["feedbacks"])

    # Debug panel
    draw_divider()
    draw_debug_panel(last_debug, last_hands)

    pygame.display.flip()
