import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import pygame
import numpy as np
import sys
import os
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

# --- MediaPipe setup (new Tasks API) ---
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)
detector = vision.HandLandmarker.create_from_options(options)

# --- Webcam setup ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- Layout ---
# Left panel: game  |  Right panel: debug camera feed
GAME_W,  GAME_H  = 800, 600
DBG_W,   DBG_H   = 640, 360          # 16:9 scaled-down feed
PANEL_W          = 660               # right panel width (adds a little padding)
TOTAL_W          = GAME_W + PANEL_W
HEIGHT           = GAME_H

# --- Pygame setup ---
pygame.init()
screen     = pygame.display.set_mode((TOTAL_W, HEIGHT))
pygame.display.set_caption("Gesture Pong  |  Debug View ->")
clock      = pygame.time.Clock()
font_large = pygame.font.SysFont(None, 80)
font_med   = pygame.font.SysFont(None, 42)
font_small = pygame.font.SysFont(None, 30)
font_tiny  = pygame.font.SysFont(None, 24)

# --- Colours ---
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
GREY   = (180, 180, 180)
DGREY  = (60,  60,  60)
GREEN  = (0,   255, 100)
YELLOW = (255, 220, 0)

# Annotation colours in RGB (used for both OpenCV drawing on RGB array and Pygame)
DBG_LEFT_RGB  = (100, 255, 100)
DBG_RIGHT_RGB = (255, 220,   0)
DBG_PIVOT_RGB = (255,  60,  60)

# --- Game constants ---
PADDLE_W      = 15
PADDLE_H      = 100
BALL_RADIUS   = 10
WINNING_SCORE = 5
INITIAL_SPEED = 5
SMOOTH_N      = 5

left_history  = []
right_history = []

def smooth(history, new_val, n=SMOOTH_N):
    history.append(new_val)
    if len(history) > n:
        history.pop(0)
    return int(sum(history) / len(history))

def reset_ball(dx_direction=1):
    return GAME_W // 2, HEIGHT // 2, INITIAL_SPEED * dx_direction, 4

# --- Hand skeleton connections (MediaPipe standard 21-point hand) ---
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# -----------------------------------------------------------------------
# Core detection
# -----------------------------------------------------------------------

def get_paddle_positions(frame):
    """
    Run MediaPipe on the frame.
    Returns (positions dict, mp result, flipped RGB numpy array).
    The RGB array is used directly for both annotation and Pygame blitting.
    """
    flipped  = cv2.flip(frame, 1)
    rgb      = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)

    positions = {}
    if result.hand_landmarks and result.handedness:
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label = handedness[0].display_name  # 'Left' or 'Right'
            y     = landmarks[9].y              # landmark 9 = palm centre
            positions[label] = int(y * HEIGHT)

    return positions, result, rgb

# -----------------------------------------------------------------------
# Debug surface builder  (pure numpy + OpenCV on RGB array → Pygame surface)
# -----------------------------------------------------------------------

def build_debug_surface(rgb_frame, result):
    """
    Annotate a copy of rgb_frame with hand skeleton overlays,
    resize to DBG_W x DBG_H, and return a pygame.Surface.
    Note: cv2 drawing functions treat channel order literally —
    since our array is already RGB we pass RGB tuples directly.
    """
    debug = rgb_frame.copy()
    h, w  = debug.shape[:2]

    if result.hand_landmarks and result.handedness:
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label    = handedness[0].display_name
            bone_col = DBG_LEFT_RGB if label == 'Left' else DBG_RIGHT_RGB

            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            # Skeleton bones
            for a, b in HAND_CONNECTIONS:
                cv2.line(debug, pts[a], pts[b], bone_col, 2, cv2.LINE_AA)

            # All 21 landmark dots
            for pt in pts:
                cv2.circle(debug, pt, 4, bone_col, -1, cv2.LINE_AA)

            # Landmark 9 highlighted as the control point
            cv2.circle(debug, pts[9], 11, DBG_PIVOT_RGB,    -1, cv2.LINE_AA)
            cv2.circle(debug, pts[9], 11, (255, 255, 255),   2, cv2.LINE_AA)

            # Text label next to wrist
            wx, wy  = pts[0]
            y_norm  = landmarks[9].y
            cv2.putText(debug,
                        f"{label}  y={y_norm:.2f}",
                        (wx - 30, wy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, bone_col, 2, cv2.LINE_AA)

    # Scale down to debug panel size
    resized = cv2.resize(debug, (DBG_W, DBG_H), interpolation=cv2.INTER_AREA)

    # pygame.surfarray.make_surface expects shape (W, H, 3)
    return pygame.surfarray.make_surface(resized.swapaxes(0, 1))

# -----------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------

def draw_divider():
    pygame.draw.line(screen, DGREY, (GAME_W, 0), (GAME_W, HEIGHT), 2)

def draw_dashed_centre():
    dash_h, gap = 20, 10
    x, y = GAME_W // 2, 0
    while y < HEIGHT:
        pygame.draw.rect(screen, DGREY, (x - 1, y, 2, dash_h))
        y += dash_h + gap

def draw_debug_panel(debug_surf, paddle_pos):
    """Render the camera feed + legend + live y-value readouts."""
    px = GAME_W + 10                          # left edge of panel content
    cam_y = (HEIGHT - DBG_H) // 2 - 20       # vertically centre the feed

    # Feed title
    lbl = font_tiny.render("Camera Feed  -  MediaPipe Detection", True, GREY)
    screen.blit(lbl, (px, cam_y - 22))

    # Camera feed surface
    screen.blit(debug_surf, (px, cam_y))
    pygame.draw.rect(screen, DGREY, (px - 1, cam_y - 1, DBG_W + 2, DBG_H + 2), 1)

    # Legend below feed
    legend_y = cam_y + DBG_H + 12
    legend_items = [
        (DBG_PIVOT_RGB, "Red dot  =  Landmark 9  (control point)"),
        (DBG_LEFT_RGB,  "Green    =  Left hand skeleton"),
        (DBG_RIGHT_RGB, "Yellow   =  Right hand skeleton"),
    ]
    for colour, text in legend_items:
        screen.blit(font_tiny.render(text, True, colour), (px, legend_y))
        legend_y += 20

    # Live y-value readout per hand
    legend_y += 8
    for label, colour in [('Left', GREEN), ('Right', YELLOW)]:
        if label in paddle_pos:
            y_norm = paddle_pos[label] / HEIGHT
            txt    = f"{label}:  y = {y_norm:.3f}  ->  paddle px {paddle_pos[label]}"
        else:
            txt    = f"{label}:  -- not detected --"
            colour = DGREY
        screen.blit(font_tiny.render(txt, True, colour), (px, legend_y))
        legend_y += 20

def draw_start_screen(debug_surf, paddle_pos):
    screen.fill(BLACK)
    title = font_large.render("GESTURE PONG", True, WHITE)
    sub   = font_med.render("Show both hands to start", True, GREY)
    p1    = font_small.render("Left hand   ->   Left paddle",  True, GREEN)
    p2    = font_small.render("Right hand  ->   Right paddle", True, YELLOW)
    screen.blit(title, (GAME_W//2 - title.get_width()//2, 150))
    screen.blit(sub,   (GAME_W//2 - sub.get_width()//2,   270))
    screen.blit(p1,    (GAME_W//2 - p1.get_width()//2,    330))
    screen.blit(p2,    (GAME_W//2 - p2.get_width()//2,    370))
    draw_divider()
    draw_debug_panel(debug_surf, paddle_pos)

def draw_win_screen(winner_name, debug_surf, paddle_pos):
    screen.fill(BLACK)
    msg = font_large.render(f"{winner_name} Wins!", True, WHITE)
    sub = font_small.render("SPACE to play again   |   ESC to quit", True, GREY)
    screen.blit(msg, (GAME_W//2 - msg.get_width()//2, 220))
    screen.blit(sub, (GAME_W//2 - sub.get_width()//2, 330))
    draw_divider()
    draw_debug_panel(debug_surf, paddle_pos)

# -----------------------------------------------------------------------
# Game state
# -----------------------------------------------------------------------
STATE_START, STATE_PLAYING, STATE_WIN = "start", "playing", "win"

state   = STATE_START
left_y  = HEIGHT // 2
right_y = HEIGHT // 2
ball_x, ball_y, ball_dx, ball_dy = reset_ball()
score   = [0, 0]
winner  = ""

# Blank placeholder until the first real frame arrives
blank_debug = pygame.Surface((DBG_W, DBG_H))
blank_debug.fill((30, 30, 30))
last_debug_surf = blank_debug
last_paddle_pos = {}

# -----------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------
while True:
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
            if event.key == pygame.K_SPACE and state == STATE_WIN:
                score   = [0, 0]
                ball_x, ball_y, ball_dx, ball_dy = reset_ball()
                left_history.clear()
                right_history.clear()
                state   = STATE_START

    ret, frame = cap.read()
    if not ret:
        continue

    paddle_pos, mp_result, rgb_frame = get_paddle_positions(frame)
    debug_surf = build_debug_surface(rgb_frame, mp_result)

    last_debug_surf = debug_surf
    last_paddle_pos = paddle_pos

    # Smooth paddle positions
    if 'Left'  in paddle_pos:
        left_y  = smooth(left_history,  paddle_pos['Left'])
    if 'Right' in paddle_pos:
        right_y = smooth(right_history, paddle_pos['Right'])

    # --- Start screen ---
    if state == STATE_START:
        draw_start_screen(debug_surf, paddle_pos)
        pygame.display.flip()
        clock.tick(60)
        if 'Left' in paddle_pos and 'Right' in paddle_pos:
            state = STATE_PLAYING
        continue

    # --- Win screen ---
    if state == STATE_WIN:
        draw_win_screen(winner, debug_surf, paddle_pos)
        pygame.display.flip()
        clock.tick(60)
        continue

    # --- Playing ---

    ball_x += ball_dx
    ball_y += ball_dy

    if ball_y - BALL_RADIUS <= 0:
        ball_dy = abs(ball_dy)
    if ball_y + BALL_RADIUS >= HEIGHT:
        ball_dy = -abs(ball_dy)

    # Left paddle collision
    if (ball_x - BALL_RADIUS <= 10 + PADDLE_W and
            left_y - PADDLE_H//2 < ball_y < left_y + PADDLE_H//2):
        ball_dx  =  abs(ball_dx) + 0.3
        ball_dy += 0.2 if ball_dy > 0 else -0.2

    # Right paddle collision
    if (ball_x + BALL_RADIUS >= GAME_W - 10 - PADDLE_W and
            right_y - PADDLE_H//2 < ball_y < right_y + PADDLE_H//2):
        ball_dx  = -(abs(ball_dx) + 0.3)
        ball_dy += 0.2 if ball_dy > 0 else -0.2

    # Scoring
    if ball_x < 0:
        score[1] += 1
        ball_x, ball_y, ball_dx, ball_dy = reset_ball(dx_direction=1)
    if ball_x > GAME_W:
        score[0] += 1
        ball_x, ball_y, ball_dx, ball_dy = reset_ball(dx_direction=-1)

    # Win check
    if score[0] >= WINNING_SCORE:
        winner = "Left";  state = STATE_WIN
    if score[1] >= WINNING_SCORE:
        winner = "Right"; state = STATE_WIN

    # --- Draw game panel ---
    screen.fill(BLACK)
    draw_dashed_centre()

    pygame.draw.rect(screen, GREEN,
                     (10, left_y - PADDLE_H//2, PADDLE_W, PADDLE_H))
    pygame.draw.rect(screen, YELLOW,
                     (GAME_W - 10 - PADDLE_W, right_y - PADDLE_H//2, PADDLE_W, PADDLE_H))

    pygame.draw.circle(screen, WHITE, (int(ball_x), int(ball_y)), BALL_RADIUS)

    ls = font_large.render(str(score[0]), True, GREEN)
    rs = font_large.render(str(score[1]), True, YELLOW)
    screen.blit(ls, (GAME_W//4   - ls.get_width()//2, 20))
    screen.blit(rs, (3*GAME_W//4 - rs.get_width()//2, 20))

    for label, colour, anchor in [('Left', GREEN, 10), ('Right', YELLOW, None)]:
        detected = label in paddle_pos
        txt  = "Hand detected" if detected else "No hand"
        col  = colour if detected else DGREY
        surf = font_small.render(txt, True, col)
        x    = anchor if anchor is not None else GAME_W - surf.get_width() - 10
        screen.blit(surf, (x, HEIGHT - 30))

    # --- Draw debug panel ---
    draw_divider()
    draw_debug_panel(debug_surf, paddle_pos)

    pygame.display.flip()
    clock.tick(60)
