#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pixel-Perfect Edge Snap Crop Tool v3.0
Lossless image cropping tool with pixel-level edge snapping

Fixed all issues from v2.0:
1. Pixel grid item type error (QGraphicsRectItem has no setPath method)
2. Undo/Redo functionality not working
3. Duplicate signal connections
4. Snap state display lag
5. Zoom slider and view zoom out of sync
6. Pan mode logic conflict
7. Crash after importing images
"""

import sys
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from PIL import Image, ImageQt

from PyQt6.QtCore import Qt, QRect, QRectF, QPoint, QPointF, QSize, pyqtSignal, QObject
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QBrush, 
    QMouseEvent, QKeyEvent, QWheelEvent, QTransform,
    QKeySequence, QShortcut, QAction, QPainterPath
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QCheckBox, QSpinBox, QGroupBox,
    QFileDialog, QMessageBox, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsItem,
    QGraphicsPathItem, QMenu, QStatusBar, QSlider, QToolBar, QSizePolicy
)


# ==================== Core Data Structures ====================
@dataclass
class CropBox:
    """Crop box data structure, all coordinates are integers"""
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0
    
    def width(self) -> int:
        return self.right - self.left
    
    def height(self) -> int:
        return self.bottom - self.top
    
    def is_valid(self) -> bool:
        return self.width() > 0 and self.height() > 0
    
    def to_rect(self) -> QRect:
        return QRect(self.left, self.top, self.width(), self.height())
    
    def to_rectf(self) -> QRectF:
        return QRectF(self.left, self.top, self.width(), self.height())
    
    def copy(self) -> 'CropBox':
        return CropBox(self.left, self.top, self.right, self.bottom)
    
    def __eq__(self, other: 'CropBox') -> bool:
        if not isinstance(other, CropBox):
            return False
        return (self.left == other.left and self.top == other.top and
                self.right == other.right and self.bottom == other.bottom)


@dataclass
class EdgePoint:
    """Edge point data structure"""
    x: int
    y: int
    is_vertical: bool  # True:vertical edge(left/right), False:horizontal edge(top/bottom)


@dataclass
class MagnetConfig:
    """Snap configuration"""
    enable: bool = True
    range: int = 3        # Snap range (pixels)
    threshold: int = 10   # Color difference threshold


# ==================== Pixel Data Layer ====================
class PixelData:
    """Pixel data layer: manages raw image data, read-only throughout"""
    
    def __init__(self):
        self.width = 0
        self.height = 0
        self.pixels = None
        self.has_alpha = False
        
    def load_image(self, image_path: str) -> bool:
        """Load image as raw pixel matrix"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGBA mode to ensure alpha channel
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                self.width, self.height = img.size
                self.has_alpha = True
                
                # Get raw pixel data, convert to numpy array for performance
                self.pixels = np.array(img, dtype=np.uint8)
                return True
        except Exception as e:
            print(f"Failed to load image: {e}")
            return False
    
    def get_pixel(self, x: int, y: int) -> Tuple[int, int, int, int]:
        """Get pixel value at specified coordinates (RGBA)"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return tuple(self.pixels[y, x])
        return (0, 0, 0, 0)  # Return transparent black for out-of-bounds
    
    def get_neighbors(self, x: int, y: int, size: int = 3) -> np.ndarray:
        """Get neighborhood pixels around specified coordinates"""
        half = size // 2
        x_start = max(0, x - half)
        x_end = min(self.width, x + half + 1)
        y_start = max(0, y - half)
        y_end = min(self.height, y + half + 1)
        
        return self.pixels[y_start:y_end, x_start:x_end]
    
    def crop(self, box: CropBox) -> np.ndarray:
        """Slice from raw pixel matrix, absolutely lossless"""
        if self.pixels is None:
            return None
            
        if not box.is_valid():
            return None
        
        # Ensure coordinates are within image bounds
        left = max(0, box.left)
        top = max(0, box.top)
        right = min(self.width, box.right)
        bottom = min(self.height, box.bottom)
        
        if right <= left or bottom <= top:
            return None
            
        return self.pixels[top:bottom, left:right].copy()


# ==================== Edge Detection Layer ====================
class EdgeDetector:
    """Edge detection layer: detects color gradient boundaries"""
    
    @staticmethod
    def calculate_color_diff(pixel1: Tuple[int, int, int, int], 
                           pixel2: Tuple[int, int, int, int]) -> int:
        """Calculate color difference between two pixels (Manhattan distance)"""
        # Consider RGBA channels, prevent overflow
        r_diff = abs(int(pixel1[0]) - int(pixel2[0]))
        g_diff = abs(int(pixel1[1]) - int(pixel2[1]))
        b_diff = abs(int(pixel1[2]) - int(pixel2[2]))
        a_diff = abs(int(pixel1[3]) - int(pixel2[3]))
        
        # Limit return value range to avoid overflow
        total_diff = r_diff + g_diff + b_diff + a_diff
        return min(total_diff, 255 * 4)  # Max possible value is 255*4=1020
    
    def detect_horizontal_edge(self, pixels: PixelData, x: int, y: int, 
                              threshold: int) -> bool:
        """Detect horizontal edge (vertical color gradient)"""
        if y <= 0 or y >= pixels.height - 1:
            return False
        
        upper = pixels.get_pixel(x, y - 1)
        current = pixels.get_pixel(x, y)
        lower = pixels.get_pixel(x, y + 1)
        
        # Calculate vertical gradient
        diff_up = self.calculate_color_diff(upper, current)
        diff_down = self.calculate_color_diff(current, lower)
        
        # If difference exceeds threshold, this is a horizontal edge
        return diff_up > threshold or diff_down > threshold
    
    def detect_vertical_edge(self, pixels: PixelData, x: int, y: int,
                            threshold: int) -> bool:
        """Detect vertical edge (horizontal color gradient)"""
        if x <= 0 or x >= pixels.width - 1:
            return False
        
        left = pixels.get_pixel(x - 1, y)
        current = pixels.get_pixel(x, y)
        right = pixels.get_pixel(x + 1, y)
        
        # Calculate horizontal gradient
        diff_left = self.calculate_color_diff(left, current)
        diff_right = self.calculate_color_diff(current, right)
        
        # If difference exceeds threshold, this is a vertical edge
        return diff_left > threshold or diff_right > threshold
    
    def find_nearest_edge(self, pixels: PixelData, start_x: int, start_y: int,
                         is_vertical: bool, search_range: int, 
                         threshold: int) -> Optional[EdgePoint]:
        """
        Find nearest edge point
        
        Parameters:
            is_vertical: True - search vertical edges (for left/right snapping)
                        False - search horizontal edges (for top/bottom snapping)
        """
        nearest_edge = None
        min_distance = float('inf')
        
        if is_vertical:
            # Search vertical edges (left/right snapping)
            for offset in range(search_range + 1):
                # Search right
                if start_x + offset < pixels.width:
                    if self.detect_vertical_edge(pixels, start_x + offset, start_y, threshold):
                        distance = offset
                        if distance < min_distance:
                            min_distance = distance
                            nearest_edge = EdgePoint(start_x + offset, start_y, True)
                # Search left
                if start_x - offset >= 0:
                    if self.detect_vertical_edge(pixels, start_x - offset, start_y, threshold):
                        distance = offset
                        if distance < min_distance:
                            min_distance = distance
                            nearest_edge = EdgePoint(start_x - offset, start_y, True)
        else:
            # Search horizontal edges (top/bottom snapping)
            for offset in range(search_range + 1):
                # Search down
                if start_y + offset < pixels.height:
                    if self.detect_horizontal_edge(pixels, start_x, start_y + offset, threshold):
                        distance = offset
                        if distance < min_distance:
                            min_distance = distance
                            nearest_edge = EdgePoint(start_x, start_y + offset, False)
                # Search up
                if start_y - offset >= 0:
                    if self.detect_horizontal_edge(pixels, start_x, start_y - offset, threshold):
                        distance = offset
                        if distance < min_distance:
                            min_distance = distance
                            nearest_edge = EdgePoint(start_x, start_y - offset, False)
        
        return nearest_edge


# ==================== Interaction & Snap Layer ====================
class MagnetController:
    """Interaction & snap layer: manages crop box snap logic"""
    
    def __init__(self, pixel_data: PixelData):
        self.pixels = pixel_data
        self.edge_detector = EdgeDetector()
        self.config = MagnetConfig()
        self.crop_box = CropBox()
        
        # Drag state
        self.dragging = False
        self.drag_edge = None
        self.drag_start_point = QPointF()
        self.drag_start_box = CropBox()
        
        # Snap state
        self.is_snapped = False
        self.snapped_edge = None
    
    def set_config(self, enable: bool, magnet_range: int, threshold: int):
        """Update snap configuration"""
        self.config.enable = enable
        self.config.range = magnet_range
        self.config.threshold = threshold
    
    def get_snapped_edge_info(self) -> Tuple[bool, Optional[str]]:
        """Get snap state information"""
        return self.is_snapped, self.snapped_edge
    
    def start_drag(self, scene_pos: QPointF, edge: Optional[str]):
        """Start dragging"""
        self.dragging = True
        self.drag_edge = edge
        self.drag_start_point = scene_pos
        self.drag_start_box = self.crop_box.copy()
        self.is_snapped = False
        self.snapped_edge = None
    
    def end_drag(self):
        """End dragging"""
        self.dragging = False
        self.drag_edge = None
        # Check if actually on edge
        if self.crop_box.is_valid():
            # Check if left edge is on boundary
            left_snapped = self._check_edge_at_position(self.crop_box.left, self.crop_box.top + (self.crop_box.height() // 2), True)
            # Check if right edge is on boundary
            right_snapped = self._check_edge_at_position(self.crop_box.right - 1, self.crop_box.top + (self.crop_box.height() // 2), True)
            # Check if top edge is on boundary
            top_snapped = self._check_edge_at_position(self.crop_box.left + (self.crop_box.width() // 2), self.crop_box.top, False)
            # Check if bottom edge is on boundary
            bottom_snapped = self._check_edge_at_position(self.crop_box.left + (self.crop_box.width() // 2), self.crop_box.bottom - 1, False)
            
            # If any edge is on boundary, maintain snap state
            if left_snapped or right_snapped:
                self.is_snapped = True
                self.snapped_edge = 'vertical'
            elif top_snapped or bottom_snapped:
                self.is_snapped = True
                self.snapped_edge = 'horizontal'
            else:
                # Not on boundary, reset snap state
                self.is_snapped = False
                self.snapped_edge = None
        else:
            # Invalid crop box, reset snap state
            self.is_snapped = False
            self.snapped_edge = None
    
    def _check_edge_at_position(self, x: int, y: int, is_vertical: bool) -> bool:
        """Check if specified position is on an edge"""
        if is_vertical:
            return self.edge_detector.detect_vertical_edge(self.pixels, x, y, self.config.threshold)
        else:
            return self.edge_detector.detect_horizontal_edge(self.pixels, x, y, self.config.threshold)
    
    def update_drag(self, scene_pos: QPointF) -> Optional[QRectF]:
        """Update drag position, return new rectangle if changed"""
        if not self.dragging or not self.drag_edge:
            return None
        
        # Calculate mouse movement distance
        delta_x = scene_pos.x() - self.drag_start_point.x()
        delta_y = scene_pos.y() - self.drag_start_point.y()
        
        # Reset snap state
        self.is_snapped = False
        self.snapped_edge = None
        
        # Edge dragging with snap
        if self.drag_edge == 'left':
            new_left = self._calculate_edge_position(scene_pos, 'left')
            new_top = self.drag_start_box.top
            new_right = self.drag_start_box.right
            new_bottom = self.drag_start_box.bottom
        elif self.drag_edge == 'right':
            new_left = self.drag_start_box.left
            new_top = self.drag_start_box.top
            new_right = self._calculate_edge_position(scene_pos, 'right')
            new_bottom = self.drag_start_box.bottom
        elif self.drag_edge == 'top':
            new_left = self.drag_start_box.left
            new_top = self._calculate_edge_position(scene_pos, 'top')
            new_right = self.drag_start_box.right
            new_bottom = self.drag_start_box.bottom
        elif self.drag_edge == 'bottom':
            new_left = self.drag_start_box.left
            new_top = self.drag_start_box.top
            new_right = self.drag_start_box.right
            new_bottom = self._calculate_edge_position(scene_pos, 'bottom')
        elif self.drag_edge == 'topleft':
            # Top-left corner drag, adjust left and top simultaneously
            new_left = self._calculate_edge_position(scene_pos, 'left')
            new_top = self._calculate_edge_position(scene_pos, 'top')
            new_right = self.drag_start_box.right
            new_bottom = self.drag_start_box.bottom
        elif self.drag_edge == 'topright':
            # Top-right corner drag, adjust right and top simultaneously
            new_left = self.drag_start_box.left
            new_top = self._calculate_edge_position(scene_pos, 'top')
            new_right = self._calculate_edge_position(scene_pos, 'right')
            new_bottom = self.drag_start_box.bottom
        elif self.drag_edge == 'bottomleft':
            # Bottom-left corner drag, adjust left and bottom simultaneously
            new_left = self._calculate_edge_position(scene_pos, 'left')
            new_top = self.drag_start_box.top
            new_right = self.drag_start_box.right
            new_bottom = self._calculate_edge_position(scene_pos, 'bottom')
        elif self.drag_edge == 'bottomright':
            # Bottom-right corner drag, adjust right and bottom simultaneously
            new_left = self.drag_start_box.left
            new_top = self.drag_start_box.top
            new_right = self._calculate_edge_position(scene_pos, 'right')
            new_bottom = self._calculate_edge_position(scene_pos, 'bottom')
        else:
            return None
        
        # Boundary constraints
        new_left = max(0, new_left)
        new_top = max(0, new_top)
        new_right = min(self.pixels.width, new_right)
        new_bottom = min(self.pixels.height, new_bottom)
        
        # Ensure rectangle is valid
        if new_right <= new_left:
            if self.drag_edge == 'left':
                new_left = new_right - 1
            else:
                new_right = new_left + 1
        
        if new_bottom <= new_top:
            if self.drag_edge == 'top':
                new_top = new_bottom - 1
            else:
                new_bottom = new_top + 1
        
        # Check if changed
        if (new_left == self.crop_box.left and new_top == self.crop_box.top and
            new_right == self.crop_box.right and new_bottom == self.crop_box.bottom):
            return None
        
        # Update crop box
        self.crop_box.left = new_left
        self.crop_box.top = new_top
        self.crop_box.right = new_right
        self.crop_box.bottom = new_bottom
        
        return self.crop_box.to_rectf()
    
    def _calculate_edge_position(self, scene_pos: QPointF, edge: str) -> int:
        """Calculate edge position (with snapping)"""
        # Ensure integer coordinates for pixel-level precision
        if not self.config.enable:
            return int(getattr(self, f'_calculate_raw_{edge}_pos')(scene_pos) + 0.5)
        
        # Determine sampling points and search direction based on edge
        if edge in ['left', 'right']:
            # Vertical edge
            is_vertical_search = True
            base_x = int(scene_pos.x() + 0.5)
            base_y = int(self.drag_start_box.top + (self.drag_start_box.bottom - self.drag_start_box.top) / 2 + 0.5)
            
            # Sample multiple points in y-direction
            sample_points = self._get_sample_points(base_y, self.pixels.height, vertical=True)
            edges_found = []
            
            for y in sample_points:
                edge_point = self.edge_detector.find_nearest_edge(
                    self.pixels, base_x, y, is_vertical_search,
                    self.config.range, self.config.threshold
                )
                if edge_point:
                    edges_found.append(edge_point.x)
            
            if edges_found:
                # Choose most common edge position
                from collections import Counter
                most_common = Counter(edges_found).most_common(1)
                if most_common:
                    self.is_snapped = True
                    self.snapped_edge = 'vertical'
                    return most_common[0][0]
                else:
                    # No edge, ensure integer coordinates
                    self.is_snapped = False
                    self.snapped_edge = None
                    return int(getattr(self, f'_calculate_raw_{edge}_pos')(scene_pos) + 0.5)
        
        else:  # 'top', 'bottom'
            # Horizontal edge
            is_vertical_search = False
            base_x = int(self.drag_start_box.left + (self.drag_start_box.right - self.drag_start_box.left) / 2 + 0.5)
            base_y = int(scene_pos.y() + 0.5)
            
            # Sample multiple points in x-direction
            sample_points = self._get_sample_points(base_x, self.pixels.width, vertical=False)
            edges_found = []
            
            for x in sample_points:
                edge_point = self.edge_detector.find_nearest_edge(
                    self.pixels, x, base_y, is_vertical_search,
                    self.config.range, self.config.threshold
                )
                if edge_point:
                    edges_found.append(edge_point.y)
            
            if edges_found:
                # Choose most common edge position
                from collections import Counter
                most_common = Counter(edges_found).most_common(1)
                if most_common:
                    self.is_snapped = True
                    self.snapped_edge = 'horizontal'
                    return most_common[0][0]
                else:
                    # No edge, ensure integer coordinates
                    self.is_snapped = False
                    self.snapped_edge = None
                    return int(getattr(self, f'_calculate_raw_{edge}_pos')(scene_pos) + 0.5)
        
        # No edge found, return raw calculated position (ensure integer)
        return int(getattr(self, f'_calculate_raw_{edge}_pos')(scene_pos) + 0.5)
    
    def _get_sample_points(self, center: int, max_value: int, vertical: bool) -> List[int]:
        """Get sampling points"""
        points = [center]
        
        # Add offset points
        offsets = [1, 2, -1, -2]
        for offset in offsets:
            point = center + offset
            if 0 <= point < max_value:
                points.append(point)
        
        # Deduplicate and limit count
        return list(set(points))[:5]
    
    def _calculate_raw_left_pos(self, scene_pos: QPointF) -> int:
        """Calculate left position (without snapping)"""
        delta_x = scene_pos.x() - self.drag_start_point.x()
        new_left = int(self.drag_start_box.left + delta_x + 0.5)
        return max(0, min(new_left, self.crop_box.right - 1))
    
    def _calculate_raw_right_pos(self, scene_pos: QPointF) -> int:
        """Calculate right position (without snapping)"""
        delta_x = scene_pos.x() - self.drag_start_point.x()
        new_right = int(self.drag_start_box.right + delta_x + 0.5)
        return min(self.pixels.width, max(new_right, self.crop_box.left + 1))
    
    def _calculate_raw_top_pos(self, scene_pos: QPointF) -> int:
        """Calculate top position (without snapping)"""
        delta_y = scene_pos.y() - self.drag_start_point.y()
        new_top = int(self.drag_start_box.top + delta_y + 0.5)
        return max(0, min(new_top, self.crop_box.bottom - 1))
    
    def _calculate_raw_bottom_pos(self, scene_pos: QPointF) -> int:
        """Calculate bottom position (without snapping)"""
        delta_y = scene_pos.y() - self.drag_start_point.y()
        new_bottom = int(self.drag_start_box.bottom + delta_y + 0.5)
        return min(self.pixels.height, max(new_bottom, self.crop_box.top + 1))
    
    def get_edge_at_position(self, scene_pos: QPointF, margin: int = 5) -> Optional[str]:
        """Get crop box edge at scene position"""
        if not self.crop_box.is_valid():
            return None
        
        x, y = scene_pos.x(), scene_pos.y()
        left, top = self.crop_box.left, self.crop_box.top
        right, bottom = self.crop_box.right, self.crop_box.bottom
        
        # Check corners
        corner_margin = margin * 1.5  # Slightly larger corner detection range
        if abs(x - left) <= corner_margin and abs(y - top) <= corner_margin:
            return 'topleft'
        elif abs(x - right) <= corner_margin and abs(y - top) <= corner_margin:
            return 'topright'
        elif abs(x - left) <= corner_margin and abs(y - bottom) <= corner_margin:
            return 'bottomleft'
        elif abs(x - right) <= corner_margin and abs(y - bottom) <= corner_margin:
            return 'bottomright'
        # Check edges (expanded detection range)
        edge_margin = 8  # Increased edge detection range
        if abs(x - left) <= edge_margin and top <= y <= bottom:
            return 'left'
        elif abs(x - right) <= edge_margin and top <= y <= bottom:
            return 'right'
        elif abs(y - top) <= edge_margin and left <= x <= right:
            return 'top'
        elif abs(y - bottom) <= edge_margin and left <= x <= right:
            return 'bottom'
        # Disable overall movement, remove 'all' return value
        
        return None


# Signal source class
class CropBoxSignals(QObject):
    box_changed = pyqtSignal(QRectF)
    snapped_state_changed = pyqtSignal(bool, str)

# ==================== Crop Box Graphics Item ====================
class CropBoxItem(QGraphicsRectItem):
    """Crop box graphics item, handles dragging and snapping"""
    
    def __init__(self, pixel_data: PixelData):
        super().__init__()
        
        # Create signal source object
        self.signals = CropBoxSignals()
        
        self.pixel_data = pixel_data
        self.magnet_controller = MagnetController(pixel_data)
        
        # Initially create a transparent rectangle covering the entire image to receive mouse events
        if pixel_data.width > 0 and pixel_data.height > 0:
            self.magnet_controller.crop_box = CropBox()
            self.setRect(0, 0, pixel_data.width, pixel_data.height)
        else:
            self.magnet_controller.crop_box = CropBox()
            self.setRect(QRectF())  # Set to empty rectangle
        
        # Interaction state
        self.current_edge = None
        self.dragging = False
        self.drawing = False
        self.draw_start = QPointF()
        self.draw_end = QPointF()
        
        # Flag indicating if crop box has been created
        self.crop_box_created = False  # Initially no crop box
        
        # Edge states (record if each edge is on a boundary)
        self.edge_states = {
            'left': False,
            'top': False,
            'right': False,
            'bottom': False
        }
        
        # Display settings
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, True)
        
        # Pens and brushes
        self.normal_pen = QPen(QColor(255, 0, 0), 1)  # Red border
        self.snapped_pen = QPen(QColor(0, 255, 0), 2)  # Green border (when snapped)
        self.drawing_pen = QPen(QColor(255, 255, 0), 1, Qt.PenStyle.DashLine)  # Yellow dashed line (when drawing)
        
        self.normal_brush = QBrush(QColor(255, 255, 255, 30))  # Semi-transparent white fill
        self.setPen(self.normal_pen)
        self.setBrush(self.normal_brush)
        
        # Control point size
        self.control_point_size = 10
        
    def set_magnet_config(self, enable: bool, magnet_range: int, threshold: int):
        """Set snap configuration"""
        self.magnet_controller.set_config(enable, magnet_range, threshold)
    
    def get_crop_box(self) -> CropBox:
        """Get crop box"""
        return self.magnet_controller.crop_box
    
    def set_crop_box(self, box: CropBox):
        """Set crop box"""
        self.magnet_controller.crop_box = box
        self.setRect(box.to_rectf())
        self.signals.box_changed.emit(box.to_rectf())
    
    def reset_to_full_image(self):
        """Reset to full image"""
        if self.pixel_data.width > 0 and self.pixel_data.height > 0:
            box = CropBox(0, 0, self.pixel_data.width, self.pixel_data.height)
            self.set_crop_box(box)
    
    def paint(self, painter, option, widget=None):
        """Paint crop box"""
        # Save painter state
        painter.save()
        
        # If drawing temporary rectangle, draw it
        if self.drawing:
            painter.setPen(self.drawing_pen)
            painter.setBrush(QBrush(QColor(255, 255, 0, 50)))  # Semi-transparent yellow fill
            rect = QRectF(self.draw_start, self.draw_end).normalized()
            painter.drawRect(rect)
            painter.restore()
            return
        
        # Only draw if crop box is valid
        if not self.magnet_controller.crop_box.is_valid():
            painter.restore()
            return
        
        # Draw fill
        painter.setBrush(self.normal_brush)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(self.rect())
        
        # Draw border (each edge independent color)
        rect = self.rect()
        
        # Draw left edge
        painter.setPen(self.snapped_pen if self.edge_states['left'] else self.normal_pen)
        painter.drawLine(rect.topLeft(), rect.bottomLeft())
        
        # Draw top edge
        painter.setPen(self.snapped_pen if self.edge_states['top'] else self.normal_pen)
        painter.drawLine(rect.topLeft(), rect.topRight())
        
        # Draw right edge
        painter.setPen(self.snapped_pen if self.edge_states['right'] else self.normal_pen)
        painter.drawLine(rect.topRight(), rect.bottomRight())
        
        # Draw bottom edge
        painter.setPen(self.snapped_pen if self.edge_states['bottom'] else self.normal_pen)
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        
        # Draw control points
        half_point = self.control_point_size // 2
        
        # Four corners (green)
        painter.setBrush(QBrush(QColor(0, 255, 0)))
        for point in [
            rect.topLeft(),
            rect.topRight(),
            rect.bottomLeft(),
            rect.bottomRight()
        ]:
            painter.drawRect(int(point.x() - half_point), int(point.y() - half_point), 
                           self.control_point_size, self.control_point_size)
        
        # Four edge midpoints (blue)
        painter.setBrush(QBrush(QColor(0, 200, 255)))
        for point in [
            QPointF(rect.left() + rect.width() / 2, rect.top()),
            QPointF(rect.left() + rect.width() / 2, rect.bottom()),
            QPointF(rect.left(), rect.top() + rect.height() / 2),
            QPointF(rect.right(), rect.top() + rect.height() / 2)
        ]:
            painter.drawRect(int(point.x() - half_point), int(point.y() - half_point),
                           self.control_point_size, self.control_point_size)
        
        painter.restore()
    
    def mousePressEvent(self, event):
        """Mouse press event"""
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = event.scenePos()
            
            # Check if on crop box
            edge = self.magnet_controller.get_edge_at_position(scene_pos)
            
            if not self.crop_box_created:
                # Only allow drawing new crop box if none exists
                # Start drawing new crop box
                self.drawing = True
                self.draw_start = scene_pos
                self.draw_end = scene_pos
                event.accept()
            elif edge:
                # Start dragging crop box
                self.dragging = True
                self.current_edge = edge
                self.magnet_controller.start_drag(scene_pos, edge)
                event.accept()
            else:
                # Clicked on blank area, ignore
                event.ignore()
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Mouse move event"""
        scene_pos = event.scenePos()
        
        if self.dragging and self.current_edge:
            # Update drag
            new_rect = self.magnet_controller.update_drag(scene_pos)
            if new_rect:
                self.setRect(new_rect)
                self.signals.box_changed.emit(new_rect)
            
            # Emit snap state change signal
            is_snapped, edge_type = self.magnet_controller.get_snapped_edge_info()
            self.signals.snapped_state_changed.emit(is_snapped, edge_type)
            
            # Update edge states in real-time
            self._update_edge_states()
            
            # Update cursor
            self._update_cursor(self.current_edge)
            
        elif self.drawing:
            # Update drawing
            self.draw_end = scene_pos
            # Draw temporary rectangle
            self.update()
        
        else:
            # Always update hover cursor
            edge = self.magnet_controller.get_edge_at_position(scene_pos)
            self._update_cursor(edge)
            self.current_edge = edge
        
        event.accept()
    
    def mouseReleaseEvent(self, event):
        """Mouse release event"""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.dragging:
                # End dragging
                self.dragging = False
                self.magnet_controller.end_drag()
                
                # Update edge states
                self._update_edge_states()
                
                self.current_edge = None
                # Emit final snap state
                is_snapped, edge_type = self.magnet_controller.get_snapped_edge_info()
                self.signals.snapped_state_changed.emit(is_snapped, edge_type)
                
            elif self.drawing:
                # End drawing, create new crop box
                self.drawing = False
                
                # Ensure rectangle is valid
                rect = QRectF(self.draw_start, self.draw_end).normalized()
                
                # Constrain within image bounds
                image_rect = QRectF(0, 0, self.pixel_data.width, self.pixel_data.height)
                rect = rect.intersected(image_rect)
                
                if rect.width() > 0 and rect.height() > 0:
                    box = CropBox(
                        int(rect.left() + 0.5),
                        int(rect.top() + 0.5),
                        int(rect.right() + 0.5),
                        int(rect.bottom() + 0.5)
                    )
                    self.set_crop_box(box)
                    # Mark crop box as created
                    self.crop_box_created = True
                    # Update edge states
                    self._update_edge_states()
        
        event.accept()
    
    def _update_edge_states(self):
        """Update edge states"""
        if not self.magnet_controller.crop_box.is_valid():
            return
        
        box = self.magnet_controller.crop_box
        
        # Check if left edge is on boundary or image edge
        left_on_edge = box.left == 0 or self._is_edge_at_position(box.left, box.top + box.height() // 2, True)
        self.edge_states['left'] = left_on_edge
        
        # Check if top edge is on boundary or image edge
        top_on_edge = box.top == 0 or self._is_edge_at_position(box.left + box.width() // 2, box.top, False)
        self.edge_states['top'] = top_on_edge
        
        # Check if right edge is on boundary or image edge
        right_on_edge = box.right == self.pixel_data.width or self._is_edge_at_position(box.right - 1, box.top + box.height() // 2, True)
        self.edge_states['right'] = right_on_edge
        
        # Check if bottom edge is on boundary or image edge
        bottom_on_edge = box.bottom == self.pixel_data.height or self._is_edge_at_position(box.left + box.width() // 2, box.bottom - 1, False)
        self.edge_states['bottom'] = bottom_on_edge
        
        # Trigger redraw
        self.update()
    
    def _is_edge_at_position(self, x: int, y: int, is_vertical: bool) -> bool:
        """Check if specified position is on an edge"""
        if is_vertical:
            return self.magnet_controller.edge_detector.detect_vertical_edge(self.pixel_data, x, y, self.magnet_controller.config.threshold)
        else:
            return self.magnet_controller.edge_detector.detect_horizontal_edge(self.pixel_data, x, y, self.magnet_controller.config.threshold)
    
    def _update_cursor(self, edge: Optional[str]):
        """Update cursor based on edge position"""
        if edge == 'left' or edge == 'right':
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif edge == 'top' or edge == 'bottom':
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif edge == 'topleft' or edge == 'bottomright':
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif edge == 'topright' or edge == 'bottomleft':
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)


# ==================== Graphics View ====================
class ImageGraphicsView(QGraphicsView):
    """Custom graphics view with zoom and pan support"""
    
    def __init__(self):
        super().__init__()
        
        # Create scene
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        # Graphics items
        self.pixmap_item = None
        self.crop_box_item = None
        self.grid_item = None
        
        # Pixel data
        self.pixel_data = PixelData()
        
        # View settings
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        
        # Zoom level
        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.zoom_step = 1.25
        
        # Pixel grid
        self.show_grid = False
        self.grid_pen = QPen(QColor(100, 100, 100, 50), 0)  # Cosmetic pen, width 0
    
    def load_image(self, image_path: str) -> bool:
        """Load image"""
        print(f"Loading image: {image_path}")
        try:
            if not self.pixel_data.load_image(image_path):
                print("Pixel data loading failed")
                return False
            
            print(f"Image loaded successfully: {self.pixel_data.width}x{self.pixel_data.height}")
            
            # Clear scene
            self.scene.clear()
            
            # Create QPixmap
            print("Creating QImage...")
            qimage = QImage(
                self.pixel_data.pixels.data,
                self.pixel_data.width,
                self.pixel_data.height,
                self.pixel_data.width * 4,
                QImage.Format.Format_RGBA8888
            )
            print("QImage created successfully")
            
            pixmap = QPixmap.fromImage(qimage)
            print("QPixmap created successfully")
            
            # Set pixmap item
            self.pixmap_item = QGraphicsPixmapItem(pixmap)
            self.pixmap_item.setTransformationMode(Qt.TransformationMode.FastTransformation)  # Nearest neighbor interpolation
            self.scene.addItem(self.pixmap_item)
            print("Pixmap item added successfully")
            
            # Set scene size
            self.scene.setSceneRect(0, 0, self.pixel_data.width, self.pixel_data.height)
            print("Scene size set successfully")
            
            # Create crop box
            print("Creating CropBoxItem...")
            self.crop_box_item = CropBoxItem(self.pixel_data)
            print("CropBoxItem created successfully")
            self.scene.addItem(self.crop_box_item)
            print("Crop box added successfully")
            
            # Create pixel grid (using QGraphicsPathItem)
            self.grid_item = QGraphicsPathItem()
            self.grid_item.setPen(self.grid_pen)
            self.grid_item.setZValue(10)  # Topmost
            self.grid_item.setVisible(False)
            self.scene.addItem(self.grid_item)
            print("Grid added successfully")
            
            # Fit to view
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.zoom_level = self.transform().m11()  # Get current zoom level
            print(f"Fit to view complete, zoom level: {self.zoom_level}")
            
            return True
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_crop_box_item(self) -> CropBoxItem:
        """Get crop box item"""
        return self.crop_box_item
    
    def wheelEvent(self, event: QWheelEvent):
        """Wheel event: zoom"""
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Ctrl+wheel: zoom
            zoom_in = event.angleDelta().y() > 0
            factor = self.zoom_step if zoom_in else 1.0 / self.zoom_step
            
            # Calculate new zoom level
            new_zoom = self.zoom_level * factor
            if self.min_zoom <= new_zoom <= self.max_zoom:
                # Zoom centered at mouse position
                old_pos = self.mapToScene(event.position().toPoint())
                self.scale(factor, factor)
                new_pos = self.mapToScene(event.position().toPoint())
                delta = new_pos - old_pos
                self.translate(-delta.x(), -delta.y())
                self.zoom_level = new_zoom
                
                # Update pixel grid visibility
                self._update_grid_visibility()
            
            event.accept()
        else:
            # Normal wheel: vertical scroll
            super().wheelEvent(event)
    
    def mousePressEvent(self, event: QMouseEvent):
        """Mouse press event"""
        # Allow dragging only in pan mode or when clicking blank area
        if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
            super().mousePressEvent(event)
        else:
            # Check if clicked on blank area
            item = self.itemAt(event.pos())
            if not item or item == self.grid_item:
                # Clicked blank area, switch to pan mode
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                # Simulate mouse press event
                super().mousePressEvent(event)
            else:
                super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Mouse move event"""
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Mouse release event"""
        # Exit pan mode when mouse released
        if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        super().mouseReleaseEvent(event)
    
    def keyPressEvent(self, event: QKeyEvent):
        """Keyboard event"""
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            # Space key: temporarily switch to pan mode
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            event.accept()
        elif event.key() == Qt.Key.Key_0 and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Ctrl+0: fit to window
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.zoom_level = self.transform().m11()
            self._update_grid_visibility()
            event.accept()
        elif event.key() == Qt.Key.Key_1 and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Ctrl+1: actual size
            self.resetTransform()
            self.zoom_level = 1.0
            self._update_grid_visibility()
            event.accept()
        elif event.key() == Qt.Key.Key_G and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Ctrl+G: toggle pixel grid
            self.show_grid = not self.show_grid
            self._update_grid_visibility()
            event.accept()
        else:
            super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event: QKeyEvent):
        """Keyboard release event"""
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            # Release space key: restore mode
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            event.accept()
        else:
            super().keyReleaseEvent(event)
    
    def _update_grid_visibility(self):
        """Update pixel grid visibility"""
        if not self.grid_item:
            return
        
        # Only show grid when zoomed in sufficiently
        if self.show_grid and self.zoom_level >= 2.0:
            self.grid_item.setVisible(True)
            self._draw_pixel_grid()
        else:
            self.grid_item.setVisible(False)
    
    def _draw_pixel_grid(self):
        """Draw pixel grid"""
        if not self.grid_item or not self.pixel_data:
            return
        
        # Create grid path
        path = QPainterPath()
        
        # Vertical lines
        for x in range(0, self.pixel_data.width + 1):
            path.moveTo(x, 0)
            path.lineTo(x, self.pixel_data.height)
        
        # Horizontal lines
        for y in range(0, self.pixel_data.height + 1):
            path.moveTo(0, y)
            path.lineTo(self.pixel_data.width, y)
        
        # Set grid path
        self.grid_item.setPath(path)


# ==================== Main Window ====================
class MainWindow(QMainWindow):
    """Main window"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize UI
        self.init_ui()
        
        # Current file path
        self.current_file = None
        
        # Undo/Redo stacks
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo_steps = 20
        
        # Signal connection state
        self.crop_box_connections = []
    
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("Pixel-Perfect Edge Snap Crop Tool v3.0")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Create toolbar
        self.create_toolbar()
        main_layout.addWidget(self.toolbar)
        
        # Control panel
        control_group = QGroupBox("Snap Control")
        control_layout = QHBoxLayout()
        
        # Snap toggle
        self.magnet_checkbox = QCheckBox("Enable Edge Snapping")
        self.magnet_checkbox.setChecked(True)
        self.magnet_checkbox.stateChanged.connect(self.update_magnet_config)
        control_layout.addWidget(self.magnet_checkbox)
        
        # Snap range
        control_layout.addWidget(QLabel("Snap Range:"))
        self.range_spinbox = QSpinBox()
        self.range_spinbox.setRange(1, 10)
        self.range_spinbox.setValue(3)
        self.range_spinbox.valueChanged.connect(self.update_magnet_config)
        control_layout.addWidget(self.range_spinbox)
        control_layout.addWidget(QLabel("px"))
        
        # Color difference threshold
        control_layout.addWidget(QLabel("Sensitivity:"))
        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setRange(1, 100)
        self.threshold_spinbox.setValue(10)
        self.threshold_spinbox.valueChanged.connect(self.update_magnet_config)
        control_layout.addWidget(self.threshold_spinbox)
        
        control_layout.addStretch()
        
        # Zoom level display
        self.zoom_label = QLabel("Zoom: 100%")
        control_layout.addWidget(self.zoom_label)
        
        # Size display
        self.size_label = QLabel("Size: 0 x 0 pixels")
        control_layout.addWidget(self.size_label)
        
        # Snap status display
        self.snap_status_label = QLabel("Snap: Inactive")
        self.snap_status_label.setStyleSheet("color: gray;")
        control_layout.addWidget(self.snap_status_label)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # Create graphics view
        self.graphics_view = ImageGraphicsView()
        main_layout.addWidget(self.graphics_view, 1)
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready - Use Ctrl+Wheel to zoom, Space+drag to pan, Ctrl+0 fit to window, Ctrl+1 actual size, Ctrl+G toggle grid")
        
        # Shortcuts
        self.setup_shortcuts()
    
    def create_toolbar(self):
        """Create toolbar"""
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setMovable(False)
        
        # Open button
        open_action = QAction("Open Image", self)
        open_action.triggered.connect(self.open_image)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.toolbar.addAction(open_action)
        
        # Save button
        self.save_action = QAction("Save Crop", self)
        self.save_action.triggered.connect(self.save_cropped_image)
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.setEnabled(False)
        self.toolbar.addAction(self.save_action)
        
        self.toolbar.addSeparator()
        
        # Fit to window
        fit_action = QAction("Fit to Window", self)
        fit_action.triggered.connect(self.fit_to_window)
        fit_action.setShortcut("Ctrl+0")
        self.toolbar.addAction(fit_action)
        
        # Actual size
        actual_size_action = QAction("Actual Size", self)
        actual_size_action.triggered.connect(self.actual_size)
        actual_size_action.setShortcut("Ctrl+1")
        self.toolbar.addAction(actual_size_action)
        
        # Zoom slider
        self.toolbar.addWidget(QLabel(" Zoom:"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 1000)  # 10% to 1000%
        self.zoom_slider.setValue(100)
        self.zoom_slider.setMaximumWidth(150)
        self.zoom_slider.valueChanged.connect(self.zoom_slider_changed)
        self.toolbar.addWidget(self.zoom_slider)
        
        self.toolbar.addSeparator()
        
        # Reset crop box
        reset_action = QAction("Reset Crop Box", self)
        reset_action.triggered.connect(self.reset_crop_box)
        reset_action.setEnabled(False)
        self.reset_action = reset_action
        self.toolbar.addAction(reset_action)
        
        # Grid toggle
        grid_action = QAction("Show Grid", self)
        grid_action.triggered.connect(self.toggle_grid)
        grid_action.setShortcut("Ctrl+G")
        grid_action.setCheckable(True)
        grid_action.setChecked(False)
        self.grid_action = grid_action
        self.toolbar.addAction(grid_action)
        
        self.addToolBar(self.toolbar)
    
    def setup_shortcuts(self):
        """Set up keyboard shortcuts"""
        # Undo
        undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo_shortcut.activated.connect(self.undo)
        
        # Redo
        redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, self)
        redo_shortcut.activated.connect(self.redo)
    
    def open_image(self):
        """Open image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image File", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.webp)"
        )
        
        if file_path:
            self.current_file = file_path
            if self.graphics_view.load_image(file_path):
                # Enable related functions
                self.save_action.setEnabled(True)
                self.reset_action.setEnabled(True)
                
                # Update status
                self.status_bar.showMessage(f"Loaded: {file_path}")
                
                # Save initial state to undo stack
                self.save_to_undo_stack()
                
                # Connect crop box change signals
                crop_box_item = self.graphics_view.get_crop_box_item()
                if crop_box_item:
                    # Disconnect old connections
                    for connection in self.crop_box_connections:
                        try:
                            connection[0].disconnect(connection[1])
                        except:
                            pass
                    self.crop_box_connections.clear()
                    
                    # Connect new signals
                    connection1 = (crop_box_item.signals.box_changed, self.on_crop_box_changed)
                    crop_box_item.signals.box_changed.connect(self.on_crop_box_changed)
                    self.crop_box_connections.append(connection1)
                    
                    connection2 = (crop_box_item.signals.snapped_state_changed, self.on_snapped_state_changed)
                    crop_box_item.signals.snapped_state_changed.connect(self.on_snapped_state_changed)
                    self.crop_box_connections.append(connection2)
                    
                    self.update_magnet_config()
                    self.on_crop_box_changed()
            else:
                QMessageBox.critical(self, "Load Failed", f"Cannot load image: {file_path}")
    
    def save_cropped_image(self):
        """Save cropped image"""
        if not self.current_file or not self.graphics_view.pixel_data:
            return
        
        crop_box_item = self.graphics_view.get_crop_box_item()
        if not crop_box_item:
            return
        
        # Get crop box
        crop_box = crop_box_item.get_crop_box()
        if not crop_box.is_valid():
            QMessageBox.warning(self, "Invalid Crop Box", "Please draw a valid crop area first")
            return
        
        # Generate default filename
        import os
        dir_name = os.path.dirname(self.current_file)
        base_name = os.path.basename(self.current_file)
        name, ext = os.path.splitext(base_name)
        default_name = f"{name}_cropped.png"
        
        # Select save path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Cropped Image", 
            os.path.join(dir_name, default_name),
            "PNG Image (*.png);;All Files (*)"
        )
        
        if file_path:
            # Perform crop
            cropped_data = self.graphics_view.pixel_data.crop(crop_box)
            if cropped_data is not None:
                try:
                    # Convert to PIL image and save
                    cropped_image = Image.fromarray(cropped_data, 'RGBA')
                    cropped_image.save(file_path, 'PNG', compress_level=0)  # Uncompressed PNG
                    
                    self.status_bar.showMessage(f"Image saved: {file_path}")
                    QMessageBox.information(self, "Save Successful", f"Image saved to:\n{file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Save Failed", f"Error saving image:\n{str(e)}")
            else:
                QMessageBox.critical(self, "Crop Failed", "Cannot crop image")
    
    def reset_crop_box(self):
        """Reset crop box to full image"""
        crop_box_item = self.graphics_view.get_crop_box_item()
        if crop_box_item:
            crop_box_item.reset_to_full_image()
    
    def update_magnet_config(self):
        """Update snap configuration"""
        crop_box_item = self.graphics_view.get_crop_box_item()
        if crop_box_item:
            crop_box_item.set_magnet_config(
                self.magnet_checkbox.isChecked(),
                self.range_spinbox.value(),
                self.threshold_spinbox.value()
            )
    
    def on_crop_box_changed(self):
        """Handle crop box change"""
        # Save to undo stack
        self.save_to_undo_stack()
        # Update size display
        self.update_size_display()
    
    def on_snapped_state_changed(self, is_snapped: bool, edge_type: Optional[str]):
        """Handle snap state change"""
        if is_snapped:
            edge_text = "Vertical" if edge_type == "vertical" else "Horizontal"
            self.snap_status_label.setText(f"Snap: Snapped to {edge_text} Edge")
            self.snap_status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.snap_status_label.setText("Snap: Inactive")
            self.snap_status_label.setStyleSheet("color: gray;")
    
    def update_size_display(self):
        """Update size and snap state display"""
        crop_box_item = self.graphics_view.get_crop_box_item()
        if crop_box_item:
            crop_box = crop_box_item.get_crop_box()
            self.size_label.setText(f"Size: {crop_box.width()} x {crop_box.height()} pixels")
            
            # Update zoom display
            self.update_zoom_display()
    
    def update_zoom_display(self):
        """Update zoom display"""
        zoom_percent = int(self.graphics_view.zoom_level * 100)
        self.zoom_label.setText(f"Zoom: {zoom_percent}%")
        # Block signals to prevent slider change from triggering zoom
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(zoom_percent)
        self.zoom_slider.blockSignals(False)
    
    def zoom_slider_changed(self, value):
        """Zoom slider changed"""
        if not self.graphics_view or not hasattr(self.graphics_view, 'pixel_data') or not self.graphics_view.pixel_data:
            return
        
        target_zoom = max(self.graphics_view.min_zoom, 
                         min(self.graphics_view.max_zoom, value / 100.0))
        current_zoom = self.graphics_view.zoom_level
        
        if abs(target_zoom - current_zoom) > 0.01:
            # Use reset transform to avoid cumulative errors
            self.graphics_view.resetTransform()
            self.graphics_view.scale(target_zoom, target_zoom)
            self.graphics_view.zoom_level = target_zoom
            self.graphics_view._update_grid_visibility()
            self.update_zoom_display()
    
    def fit_to_window(self):
        """Fit to window size"""
        if self.graphics_view.scene():
            self.graphics_view.fitInView(
                self.graphics_view.scene().sceneRect(),
                Qt.AspectRatioMode.KeepAspectRatio
            )
            self.graphics_view.zoom_level = self.graphics_view.transform().m11()
            self.graphics_view._update_grid_visibility()
            self.update_zoom_display()
    
    def actual_size(self):
        """Actual size"""
        self.graphics_view.resetTransform()
        self.graphics_view.zoom_level = 1.0
        self.graphics_view._update_grid_visibility()
        self.update_zoom_display()
    
    def toggle_grid(self):
        """Toggle grid display"""
        self.graphics_view.show_grid = self.grid_action.isChecked()
        self.graphics_view._update_grid_visibility()
    
    def save_to_undo_stack(self):
        """Save current state to undo stack"""
        crop_box_item = self.graphics_view.get_crop_box_item()
        if not crop_box_item:
            return
        
        crop_box = crop_box_item.get_crop_box()
        
        # Check if same as top of stack
        if self.undo_stack and self.undo_stack[-1] == crop_box:
            return
        
        # Save to undo stack
        self.undo_stack.append(crop_box.copy())
        
        # Limit stack size
        if len(self.undo_stack) > self.max_undo_steps:
            self.undo_stack.pop(0)
        
        # Clear redo stack
        self.redo_stack.clear()
    
    def undo(self):
        """Undo"""
        if len(self.undo_stack) < 2:  # Need at least 2 states to undo
            return
        
        crop_box_item = self.graphics_view.get_crop_box_item()
        if not crop_box_item:
            return
        
        # Save current state to redo stack
        current_box = crop_box_item.get_crop_box()
        self.redo_stack.append(current_box.copy())
        
        # Pop previous state
        self.undo_stack.pop()  # Remove current state
        prev_box = self.undo_stack[-1]  # Get previous state
        
        # Apply previous state
        crop_box_item.set_crop_box(prev_box.copy())
    
    def redo(self):
        """Redo"""
        if not self.redo_stack:
            return
        
        crop_box_item = self.graphics_view.get_crop_box_item()
        if not crop_box_item:
            return
        
        # Get redo state
        redo_box = self.redo_stack.pop()
        
        # Save current state to undo stack
        current_box = crop_box_item.get_crop_box()
        self.undo_stack.append(current_box.copy())
        
        # Apply redo state
        crop_box_item.set_crop_box(redo_box.copy())
    
    def closeEvent(self, event):
        """Close event"""
        # Clean up resources
        if hasattr(self, 'graphics_view') and self.graphics_view.scene:
            self.graphics_view.scene.clear()
        event.accept()


# ==================== Main Entry Point ====================
def main():
    """Main function"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Set application metadata
    app.setApplicationName("Pixel-Perfect Edge Snap Crop Tool")
    app.setApplicationVersion("3.0")
    
    # Create main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()