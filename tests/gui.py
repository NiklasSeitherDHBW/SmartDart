import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow,
                             QPushButton, QVBoxLayout, QWidget)

BOARD_IMG = "dartboard.svg"


class DartMaster(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dart Master - overlay mode")
        self.resize(1100, 650)

        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)

        self.board_base = QPixmap(str(Path(BOARD_IMG))).scaled(550, 550,
                                                               Qt.AspectRatioMode.KeepAspectRatio,
                                                               Qt.TransformationMode.SmoothTransformation)
        self.hits = []
        self.build_ui()

    # ----------  UI ----------------------------------------------------------
    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # left panel (same as before, trimmed)
        left = QVBoxLayout()
        root.addLayout(left, 1)
        logo = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        left.addWidget(logo)
        left.addStretch()

        # ---- right: board + controls ---------------------------------------
        right = QVBoxLayout()
        root.addLayout(right, 2)
        # top bar
        bar = QHBoxLayout()
        self.btn_start = QPushButton("Start", clicked=self.start_cam)
        self.btn_stop = QPushButton("Stop",  clicked=self.stop_cam)
        self.btn_stop.setEnabled(False)
        bar.addWidget(self.btn_start)
        bar.addWidget(self.btn_stop)
        bar.addStretch()
        right.addLayout(bar)

        # board label
        self.lbl_board = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.lbl_board.setPixmap(self.board_base)
        right.addWidget(self.lbl_board, 1)

        # footer
        bottom = QHBoxLayout()
        bottom.addStretch()
        self.btn_reset = QPushButton("Reset hits", clicked=self.reset_hits)
        bottom.addWidget(self.btn_reset)
        right.addLayout(bottom)

        self.setStyleSheet("""
            QMainWindow {background:#3d3d3d;}
            QLabel      {color:white;}
            QPushButton {padding:4px 14px;}
        """)

    def start_cam(self):
        # Use DirectShow on Windows for full-speed property control
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # Try to request 60 fps  (must set *before* asking FPS on many cams)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        print("Camera reports", self.cap.get(cv2.CAP_PROP_FPS), "fps")
        if not self.cap.isOpened():
            print("camera not found")
            return

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.timer.start(16)

    def stop_cam(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def process_frame(self):
        # keep reading frames, but never display them
        ok, frame = self.cap.read()
        cv2.imshow("Cam", frame)
        if not ok:
            return

        # ---- YOUR VISION CODE HERE ---------------------------------------
        # Detect a new dart.  Suppose you find its tip at (u,v) in the frame
        # coordinate system.  Convert that to board space and add a hit:
        #
        #  board_x, board_y = self.camera_to_board(u, v)
        #  if board_x is not None:
        #      self.add_hit(board_x, board_y)
        #
        # For demo purposes we’ll add a random fake hit every ~60 frames.
        # ------------------------------------------------------------------
        if np.random.rand() < 0.02:                 # demo: 1 random hit / ~50 f
            rand_angle = np.random.rand() * 2*np.pi
            rand_r = np.random.rand() * 0.95    # within double ring
            cx = self.board_base.width() / 2
            cy = self.board_base.height() / 2
            bx = cx + rand_r * cx * math.cos(rand_angle)
            by = cy - rand_r * cy * math.sin(rand_angle)
            self.add_hit(bx, by)

    def add_hit(self, x, y):
        """Store and repaint a new hit in board-pixel coords."""
        self.hits.append(QPointF(x, y))
        self.repaint_board()

    def reset_hits(self):
        self.hits.clear()
        self.repaint_board()

    def repaint_board(self):
        pix = QPixmap(self.board_base)         # copy
        if self.hits:
            p = QPainter(pix)
            pen = QPen(QColor("red"), 6, Qt.PenStyle.SolidLine,
                       Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            for pt in self.hits:
                p.drawPoint(pt)
            p.end()
        self.lbl_board.setPixmap(pix)

    # ----------  geometry ----------------------------------------------------
    def camera_to_board(self, u, v):
        """
        Map camera pixel (u,v) to board image pixel (x,y).
        IMPLEMENT with your calibration / homography.
        Return (None, None) if the point is outside the board FOV.
        """
        # Placeholder: identity mapping – replace with real maths.
        # After you estimate a 3×3 homography H (camera→board) via cv2.findHomography,
        # do:
        #   pt = H @ [u, v, 1];  pt /= pt[2];  return pt[0], pt[1]
        return None, None

    def closeEvent(self, e):
        if self.cap:
            self.cap.release()
        e.accept()


if __name__ == "__main__":
    if not Path(BOARD_IMG).exists():
        sys.exit(f"‼  {BOARD_IMG} not found – place a transparent board image "
                 "next to this script.")
    app = QApplication(sys.argv)
    win = DartMaster()
    win.show()
    sys.exit(app.exec())
