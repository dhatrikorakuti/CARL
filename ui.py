"""
CARL - Cognitive Autonomous Reasoning & Learning
Minimal Cinematic AI Interface - Amber Neural Reactor
Matching reference: holographic reactor center, mic/mute buttons, glass input, status.
Backend API (JarvisUI) preserved exactly for main.py.
"""
from __future__ import annotations
import json, math, os, platform, random, sys, threading, time
from pathlib import Path
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import (QBrush, QColor, QFont, QKeySequence, QPainter, QPainterPath, QPen, QRadialGradient, QShortcut)
from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QLineEdit, QMainWindow, QPushButton, QSizePolicy, QVBoxLayout, QWidget)
_OS = platform.system()
def _base_dir():
    if getattr(sys, "frozen", False): return Path(sys.executable).parent
    return Path(__file__).resolve().parent
BASE_DIR = _base_dir()
CONFIG_DIR = BASE_DIR / "config"
API_FILE = CONFIG_DIR / "api_keys.json"


class ReactorCanvas(QWidget):
    """Full-screen amber holographic reactor with particles and rings."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setMinimumSize(600, 400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.state = "LISTENING"
        self._tick = 0
        self._glow = 0.5
        self._tgt_glow = 0.5
        self._breath = 0.0
        # 8 rings at different angles/speeds
        self._rings = [{"a": i*45.0, "spd": 0.2+i*0.08*(-1 if i%2 else 1), "r": 0.3+i*0.06, "w": 1.5+i*0.3} for i in range(8)]
        # Particles
        self._particles = []
        for _ in range(600):
            self._particles.append([random.uniform(0, 2*math.pi), random.uniform(20, 200),
                random.uniform(0.002, 0.015)*random.choice([1,-1]), random.uniform(0.3, 1.8), random.uniform(0.1, 0.5)])
        # Status/text
        self._status = "LISTENING..."
        self._connections = []  # Neural circuit connections
        self._conn_timer = 0
        self._text = ""
        self._text_role = "carl"
        self._text_alpha = 0.0
        self._text_tgt = 0.0
        self._text_timer = 0
        self._tmr = QTimer(self)
        self._tmr.timeout.connect(self._step)
        self._tmr.start(16)

    def set_state(self, s):
        self.state = s
        g = {"IDLE":0.3,"LISTENING":0.5,"UNDERSTANDING":0.65,"THINKING":0.8,"PLANNING":0.75,
             "PROCESSING":0.7,"CLEANING":0.7,"TRAINING":0.8,"EVALUATING":0.75,
             "SPEAKING":0.95,"COMPLETED":0.4,"ERROR":0.3,"MUTED":0.15,"SLEEPING":0.1}
        self._tgt_glow = g.get(s, 0.4)
        spd = {"THINKING":2.5,"TRAINING":2.0,"PROCESSING":1.8,"SPEAKING":1.5,"LISTENING":1.0}.get(s, 0.8)
        for i, ring in enumerate(self._rings):
            ring["spd"] = (0.2+i*0.08*(-1 if i%2 else 1)) * spd
        st = {"LISTENING":"LISTENING...","THINKING":"THINKING...","PROCESSING":"PROCESSING...",
              "CLEANING":"CLEANING...","TRAINING":"TRAINING MODELS...","EVALUATING":"EVALUATING...",
              "SPEAKING":"SPEAKING...","COMPLETED":"COMPLETED","ERROR":"ERROR","MUTED":"MUTED",
              "UNDERSTANDING":"UNDERSTANDING...","PLANNING":"PLANNING...","GENERATING":"GENERATING..."}
        self._status = st.get(s, "")

    def show_text(self, text, role="carl"):
        self._text = text[:200]; self._text_role = role
        self._text_tgt = 1.0; self._text_timer = 300

    def _step(self):
        self._tick += 1
        self._glow += (self._tgt_glow - self._glow) * 0.04
        self._breath = math.sin(self._tick * 0.012) * 0.5 + 0.5
        for ring in self._rings:
            ring["a"] = (ring["a"] + ring["spd"]) % 360
        for p in self._particles:
            p[0] += p[2] * (0.8 + self._glow)
        # Neural connections â€” create/destroy
        self._conn_timer += 0.016
        if self._conn_timer > 0.4/(1+self._glow):
            self._conn_timer = 0
            if len(self._connections) < 30:
                i, j = random.randint(0,len(self._particles)-1), random.randint(0,len(self._particles)-1)
                if i != j: self._connections.append([i, j, 0.0, random.uniform(2,5)])
        self._connections = [[*c[:2], c[2]+0.016*(1+self._glow), c[3]] for c in self._connections if c[2] < c[3]]
        self._text_alpha += (self._text_tgt - self._text_alpha) * 0.05
        if self._text_timer > 0:
            self._text_timer -= 1
            if self._text_timer == 0: self._text_tgt = 0.0
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        cx, cy = W/2, H*0.44
        t = self._tick

        # â”€â”€ Pure black background â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        p.fillRect(self.rect(), QColor(0, 0, 0))
        # Subtle volumetric fog from core
        fog = QRadialGradient(cx, cy, min(W,H)*0.4)
        fog.setColorAt(0.0, QColor(255, 120, 0, int(self._glow*15)))
        fog.setColorAt(0.5, QColor(100, 40, 0, int(self._glow*5)))
        fog.setColorAt(1.0, QColor(0,0,0,0))
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(fog)); p.drawRect(self.rect())

        # â”€â”€ Tiny background dust particles â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        for pt in self._particles[:80]:
            x = cx + math.cos(pt[0]*3+t*0.001) * pt[1] * 1.5
            y = cy + math.sin(pt[0]*2+t*0.0008) * pt[1] * 0.8
            a = int(pt[4] * self._glow * 60)
            p.setBrush(QBrush(QColor(255, 160, 40, max(0, min(60, a)))))
            p.drawEllipse(QPointF(x, y), 0.8, 0.8)

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # AI REACTOR â€” Vertical stacked mechanical layers
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

        # Layer config: (y_offset, width, height, segments, rotation, color_intensity)
        # Asymmetric â€” different sizes, thicknesses, complexity
        layers = [
            (-90, 200, 6, 16, self._rings[0]["a"], 0.5),    # top stabilizer (thin, wide)
            (-65, 160, 10, 12, self._rings[1]["a"]*-1, 0.7),  # upper processor
            (-40, 220, 4, 20, self._rings[2]["a"]*0.7, 0.6),  # wide circuit ring
            (-18, 140, 14, 8, self._rings[3]["a"]*-0.5, 0.9), # thick computing module
            (0,   100, 20, 6, self._rings[0]["a"]*0.3, 1.0),  # CORE (brightest)
            (22,  170, 12, 10, self._rings[4]["a"]*-0.8, 0.85), # lower processing
            (48,  240, 5, 24, self._rings[5]["a"]*0.6, 0.55), # wide energy disc
            (70,  130, 8, 14, self._rings[6]["a"]*-1.2, 0.65), # cooling assembly
            (95,  180, 4, 18, self._rings[7]["a"]*0.4, 0.4),  # base projector
        ]

        for y_off, width, height, segs, rot, intensity in layers:
            ly = cy + y_off
            half_w = width / 2
            a = int(self._glow * 180 * intensity)
            seg_width = (width - segs*3) / segs  # segments with gaps

            # Draw segmented mechanical plate
            for s in range(segs):
                # Rotation offset per segment
                seg_x_offset = math.sin(rot*0.017 + s*0.5) * 4
                sx = cx - half_w + s * (seg_width + 3) + seg_x_offset

                # Color varies per segment (amber â†’ gold â†’ copper)
                r_col = 255
                g_col = int(140 + intensity * 40 + math.sin(s+t*0.02)*20)
                col = QColor(r_col, min(220, g_col), 0, a)
                p.setPen(QPen(col, 1.2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                # Rounded rect segment (mechanical plate piece)
                p.drawRect(QRectF(sx, ly - height/2, seg_width, height))

                # Tiny internal circuit lines
                if height > 8 and s % 2 == 0:
                    inner_a = int(a * 0.4)
                    p.setPen(QPen(QColor(255, 200, 60, inner_a), 0.5))
                    p.drawLine(QPointF(sx+2, ly), QPointF(sx+seg_width-2, ly))

            # Connecting brackets on edges (mechanical arms)
            bracket_a = int(a * 0.6)
            p.setPen(QPen(QColor(255, 180, 50, bracket_a), 1.5))
            # Left bracket
            p.drawLine(QPointF(cx-half_w-8, ly-height/2), QPointF(cx-half_w-8, ly+height/2))
            p.drawLine(QPointF(cx-half_w-8, ly), QPointF(cx-half_w-3, ly))
            # Right bracket
            p.drawLine(QPointF(cx+half_w+8, ly-height/2), QPointF(cx+half_w+8, ly+height/2))
            p.drawLine(QPointF(cx+half_w+8, ly), QPointF(cx+half_w+3, ly))

        # â”€â”€ Energy conduits (vertical lines connecting layers) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        conduit_positions = [-60, -30, 30, 60]
        for cx_off in conduit_positions:
            ca = int(self._glow * 80)
            p.setPen(QPen(QColor(255, 140, 0, ca), 0.8))
            x_pos = cx + cx_off
            p.drawLine(QPointF(x_pos, cy-85), QPointF(x_pos, cy+90))
            # Energy spark traveling through conduit
            spark_y = cy - 85 + ((t * 1.5 + cx_off*2) % 175)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(255, 220, 100, int(self._glow*200))))
            p.drawRect(QRectF(x_pos-1.5, spark_y, 3, 6))

        # â”€â”€ Core energy glow (center) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        core_r = 25 * (1 + self._breath * 0.15 * self._glow)
        cg = QRadialGradient(cx, cy, core_r*3)
        cg.setColorAt(0.0, QColor(255, 255, 240, int(self._glow*230)))
        cg.setColorAt(0.2, QColor(255, 180, 50, int(self._glow*140)))
        cg.setColorAt(0.5, QColor(255, 100, 0, int(self._glow*50)))
        cg.setColorAt(1.0, QColor(0,0,0,0))
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(cg))
        p.drawEllipse(QPointF(cx, cy), core_r*3, core_r*2)
        # White-hot center
        p.setBrush(QBrush(QColor(255, 255, 255, int(self._glow*240))))
        p.drawEllipse(QPointF(cx, cy), core_r*0.25, core_r*0.25)

        # â”€â”€ Tiny blue-white processor lights (deep inside, for contrast) â”€â”€
        for i in range(6):
            lx = cx + math.cos(t*0.01+i*1.05) * 20
            ly_l = cy + math.sin(t*0.008+i*1.2) * 12
            p.setBrush(QBrush(QColor(180, 220, 255, int(self._glow*120))))
            p.drawEllipse(QPointF(lx, ly_l), 1.5, 1.5)

        # â”€â”€ Floating data particles (near the reactor) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        for pt in self._particles[80:200]:
            dist = pt[1] * 0.4
            x = cx + math.cos(pt[0]+t*0.003) * dist
            y = cy + math.sin(pt[0]*1.5+t*0.002) * dist * 0.5
            a = int(pt[4] * self._glow * 160)
            p.setBrush(QBrush(QColor(255, 179, 0, max(0, min(140, a)))))
            p.drawRect(QRectF(x-1, y-1, 2, 2))  # Square data blocks

        # â”€â”€ Audio waveform (bottom, reacts to state) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self.state in ("LISTENING", "SPEAKING", "UNDERSTANDING"):
            bar_count = 40; bar_w = 3
            total_w = bar_count * (bar_w + 2)
            start_x = cx - total_w/2
            wave_y = H - 68
            for i in range(bar_count):
                bx = start_x + i*(bar_w+2)
                if self.state == "SPEAKING":
                    h = abs(math.sin(i*0.3 + t*0.08)) * 16 * self._glow
                else:
                    h = abs(math.sin(i*0.5 + t*0.12)) * 10 * self._glow + 2
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(QColor(255, 140, 0, int(self._glow*180))))
                p.drawRect(QRectF(bx, wave_y-h, bar_w, h*2))

        # â”€â”€ Status dots â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self._status:
            dot_y = H - 38; active = (t//20)%5
            for i in range(5):
                dx = cx - 20 + i*10
                p.setBrush(QBrush(QColor(255,140,0, int(self._glow*(200 if i==active else 50)))))
                p.drawEllipse(QPointF(dx, dot_y), 2.5, 2.5)

        # â”€â”€ CARL label â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        p.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        p.setPen(QPen(QColor(255, 140, 0, 210)))
        p.drawText(24, 32, "C A R L")
        p.setFont(QFont("Segoe UI", 8)); p.setPen(QPen(QColor(255, 140, 0, 90)))
        p.drawText(24, 48, "Cognitive Autonomous Reasoning & Learning")
        # Clock
        p.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
        p.setPen(QPen(QColor(255, 179, 71, 190)))
        p.drawText(QRectF(W-170, 16, 150, 22), Qt.AlignmentFlag.AlignRight, time.strftime("%H:%M:%S"))
        p.setFont(QFont("Consolas", 9)); p.setPen(QPen(QColor(255, 140, 0, 100)))
        p.drawText(QRectF(W-170, 38, 150, 16), Qt.AlignmentFlag.AlignRight, time.strftime("%d %b %Y").upper())
        # Status text
        if self._status:
            p.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            p.setPen(QPen(QColor(255, 140, 0, 200)))
            p.drawText(QRectF(0, H-52, W, 18), Qt.AlignmentFlag.AlignCenter, self._status)
        # Floating text
        if self._text_alpha > 0.02 and self._text:
            ta = int(self._text_alpha*200)
            p.setFont(QFont("Segoe UI", 11))
            col = QColor(255,200,87,ta) if self._text_role=="carl" else QColor(255,246,213,ta)
            p.setPen(QPen(col))
            p.drawText(QRectF(W*0.15, H-140, W*0.7, 60),
                Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop|Qt.TextFlag.TextWordWrap, self._text)
        p.end()



class MainWindow(QMainWindow):
    _log_sig = pyqtSignal(str)
    _state_sig = pyqtSignal(str)
    _content_sig = pyqtSignal(str, str)
    _reconfig_sig = pyqtSignal()
    _camera_sig = pyqtSignal(bytes)
    _cam_stream_sig = pyqtSignal(bool)
    _cam_frame_sig = pyqtSignal(bytes)

    def __init__(self, face_path):
        super().__init__()
        self.setWindowTitle("CARL")
        self.setMinimumSize(900, 650)
        self.resize(1200, 800)
        self.setStyleSheet("background: #050505;")
        screen = QApplication.primaryScreen().availableGeometry()
        self.move((screen.width()-1200)//2, (screen.height()-800)//2)
        self.on_text_command = None
        self.on_remote_clicked = None
        self.on_interrupt = None
        self._muted = False
        self._cam_stop = threading.Event()
        self._force_quit = False
        self._current_file = None
        self._task_active = False
        self.setAcceptDrops(True)
        # Layout
        central = QWidget(); central.setStyleSheet("background:#050505;")
        self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        # Reactor canvas (80% of screen)
        self.hud = ReactorCanvas()
        root.addWidget(self.hud, stretch=1)
        # Bottom bar: mic + input + mute
        bar = QWidget(); bar.setFixedHeight(70); bar.setStyleSheet("background: rgba(5,5,5,0.98);")
        blay = QHBoxLayout(bar); blay.setContentsMargins(24,8,24,14); blay.setSpacing(16)
        # Mic button (left)
        self._mic_btn = QPushButton("MIC")
        self._mic_btn.setFixedSize(50,50); self._mic_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._mic_btn.setStyleSheet("QPushButton{background:rgba(255,140,0,0.08);border:2px solid rgba(255,140,0,0.3);border-radius:25px;color:#FF8C00;font-size:9px;font-weight:bold;}QPushButton:hover{background:rgba(255,140,0,0.15);border-color:rgba(255,140,0,0.5);}")
        self._mic_btn.clicked.connect(self._do_interrupt)
        blay.addWidget(self._mic_btn)
        # Input
        self._input = QLineEdit()
        self._input.setPlaceholderText("Ask CARL anything...")
        self._input.setFont(QFont("Segoe UI", 11))
        self._input.setFixedHeight(42)
        self._input.setStyleSheet("QLineEdit{background:rgba(15,12,5,0.85);color:#FFC857;border:1px solid rgba(255,140,0,0.15);border-radius:21px;padding:0 20px;}QLineEdit:focus{border:1px solid rgba(255,140,0,0.4);}")
        self._input.returnPressed.connect(self._send)
        blay.addWidget(self._input, stretch=1)
        # Mute button (right)
        self._mute_btn = QPushButton("MUTE")
        self._mute_btn.setFixedSize(50,50); self._mute_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._mute_btn.setStyleSheet("QPushButton{background:rgba(255,140,0,0.08);border:2px solid rgba(255,140,0,0.3);border-radius:25px;color:#FF8C00;font-size:8px;font-weight:bold;}QPushButton:hover{background:rgba(255,140,0,0.15);}")
        self._mute_btn.clicked.connect(self._toggle_mute)
        blay.addWidget(self._mute_btn)
        root.addWidget(bar)
        # Signals
        self._log_sig.connect(self._on_log)
        self._state_sig.connect(self._apply_state)
        self._content_sig.connect(self._on_content)
        self._reconfig_sig.connect(self._show_setup)
        self._camera_sig.connect(lambda b: None)
        self._cam_stream_sig.connect(lambda b: None)
        self._cam_frame_sig.connect(lambda b: None)
        QShortcut(QKeySequence("F4"), self).activated.connect(self._toggle_mute)
        QShortcut(QKeySequence("F11"), self).activated.connect(self._toggle_fs)
        QShortcut(QKeySequence("Escape"), self).activated.connect(self._do_interrupt)
        self._overlay = None
        self._ready = self._check_config()
        if not self._ready: self._show_setup()

    def _on_log(self, text):
        tl = text.lower()
        if tl.startswith("you:"): self.hud.show_text(text[4:].strip(), "user")
        elif tl.startswith("carl:") or tl.startswith("jarvis:"): self.hud.show_text(text.split(":",1)[1].strip(), "carl")
    def _apply_state(self, state):
        if state == "LISTENING" and self._task_active: return
        self._task_active = state not in ("LISTENING","IDLE","READY","MUTED","SLEEPING","SPEAKING","COMPLETED")
        self.hud.set_state(state)
    def _on_content(self, title, text): self.hud.show_text(f"{title}: {text[:100]}", "carl")
    def _send(self):
        txt = self._input.text().strip()
        if not txt: return
        self._input.clear(); self.hud.show_text(txt, "user")
        if self.on_text_command: threading.Thread(target=self.on_text_command, args=(txt,), daemon=True).start()
    def _toggle_mute(self):
        self._muted = not self._muted
        self._mute_btn.setText("UNMUTE" if self._muted else "MUTE")
        self.hud.set_state("MUTED" if self._muted else "LISTENING")
    def _toggle_fs(self): self.showNormal() if self.isFullScreen() else self.showFullScreen()
    def _do_interrupt(self):
        if self.on_interrupt: self.on_interrupt()
    def notify_phone_connected(self): self.hud.show_text("Remote connected.", "carl")
    def _check_config(self):
        if not API_FILE.exists(): return False
        try:
            d = json.loads(API_FILE.read_text(encoding="utf-8"))
            return bool(d.get("gemini_api_key")) and bool(d.get("os_system"))
        except: return False
    def _show_setup(self):
        if self._check_config(): self._ready = True
    def start_camera_stream(self): pass
    def stop_camera_stream(self): self._cam_stop.set()
    def closeEvent(self, event):
        if self._force_quit: event.accept(); return
        event.ignore(); self.hide()
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if not urls: return
        path = urls[0].toLocalFile()
        if not path: return
        from pathlib import Path as P
        pp = P(path)
        if not pp.exists(): return
        self._current_file = path; name = pp.name
        size = pp.stat().st_size if pp.is_file() else 0
        sz = f"{size/1024:.1f}KB" if size<1048576 else f"{size/1048576:.1f}MB"
        self.hud.show_text(f"Received: {name} ({sz})", "carl")
        if self.on_text_command:
            msg = (f"[FILE_UPLOADED] path={path} | name={name} | type={pp.suffix.lstrip('.')} | size={sz} | "
                   f"A file has been uploaded: '{name}' ({sz}). Acknowledge it, then ask what to do.")
            threading.Thread(target=self.on_text_command, args=(msg,), daemon=True).start()
    class _DropZone:
        def __init__(self, w): self._w = w
        def current_file(self): return self._w._current_file
    @property
    def _drop_zone(self): return self._DropZone(self)


class _RootShim:
    def __init__(self, app): self._app = app
    def mainloop(self): self._app.exec()
    def protocol(self, *_): pass

class JarvisUI:
    def __init__(self, face_path: str, size=None):
        self._app = QApplication.instance() or QApplication(sys.argv)
        self._app.setStyle("Fusion")
        self._app.setQuitOnLastWindowClosed(False)
        self._win = MainWindow(face_path)
        self._win.show()
        self.root = _RootShim(self._app)
    @property
    def muted(self): return self._win._muted
    @muted.setter
    def muted(self, v):
        if v != self._win._muted: self._win._toggle_mute()
    @property
    def current_file(self): return self._win._drop_zone.current_file()
    @property
    def on_text_command(self): return self._win.on_text_command
    @on_text_command.setter
    def on_text_command(self, cb): self._win.on_text_command = cb
    @property
    def on_remote_clicked(self): return self._win.on_remote_clicked
    @on_remote_clicked.setter
    def on_remote_clicked(self, cb): self._win.on_remote_clicked = cb
    @property
    def on_interrupt(self): return self._win.on_interrupt
    @on_interrupt.setter
    def on_interrupt(self, cb): self._win.on_interrupt = cb
    def notify_phone_connected(self): self._win.notify_phone_connected()
    def set_state(self, state): self._win._state_sig.emit(state)
    def write_log(self, text):
        self._win._log_sig.emit(text)
        if not self._win.isVisible(): self._win.show(); self._win.raise_()
    def wait_for_api_key(self):
        while not self._win._ready: time.sleep(0.1)
    def show_content(self, title, text): self._win._content_sig.emit(title[:48], text[:4000])
    def prompt_reconfig(self): self._win._ready = False; self._win._reconfig_sig.emit()
    def show_camera_frame(self, b): self._win._camera_sig.emit(b)
    def start_camera_stream(self): self._win.start_camera_stream()
    def stop_camera_stream(self): self._win.stop_camera_stream()
    def start_speaking(self): self.set_state("SPEAKING")
    def stop_speaking(self):
        if not self.muted: self.set_state("LISTENING")
    def force_quit(self): self._win._force_quit = True; self._app.quit()
    def show_window(self): self._win.show(); self._win.raise_(); self._win.activateWindow()
