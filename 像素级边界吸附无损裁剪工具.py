#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
像素级边界吸附无损裁剪工具 v3.0
修复了v2.0中的所有问题，包括：
1. 像素网格项类型错误 (QGraphicsRectItem 没有 setPath 方法)
2. 撤销/重做功能不可用
3. 信号重复连接
4. 吸附状态显示滞后
5. 缩放滑块与视图缩放不同步
6. 平移模式逻辑冲突
7. 导入图片后闪退问题
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


# ==================== 核心数据结构 ====================
@dataclass
class CropBox:
    """裁剪框数据结构，所有坐标为整数"""
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
    """边界点数据结构"""
    x: int
    y: int
    is_vertical: bool  # True:垂直边界(左右), False:水平边界(上下)


@dataclass
class MagnetConfig:
    """吸附配置"""
    enable: bool = True
    range: int = 3        # 吸附作用范围(像素)
    threshold: int = 10   # 颜色差阈值


# ==================== 像素数据层 ====================
class PixelData:
    """像素数据层：管理原始图像数据，全程只读"""
    
    def __init__(self):
        self.width = 0
        self.height = 0
        self.pixels = None
        self.has_alpha = False
        
    def load_image(self, image_path: str) -> bool:
        """加载图像为原始像素矩阵"""
        try:
            with Image.open(image_path) as img:
                # 转换为RGBA模式以确保alpha通道
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                self.width, self.height = img.size
                self.has_alpha = True
                
                # 获取原始像素数据，转换为numpy数组提高性能
                self.pixels = np.array(img, dtype=np.uint8)
                return True
        except Exception as e:
            print(f"加载图像失败: {e}")
            return False
    
    def get_pixel(self, x: int, y: int) -> Tuple[int, int, int, int]:
        """获取指定坐标的像素值(RGBA)"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return tuple(self.pixels[y, x])
        return (0, 0, 0, 0)  # 坐标越界返回透明黑色
    
    def get_neighbors(self, x: int, y: int, size: int = 3) -> np.ndarray:
        """获取指定坐标的邻域像素"""
        half = size // 2
        x_start = max(0, x - half)
        x_end = min(self.width, x + half + 1)
        y_start = max(0, y - half)
        y_end = min(self.height, y + half + 1)
        
        return self.pixels[y_start:y_end, x_start:x_end]
    
    def crop(self, box: CropBox) -> np.ndarray:
        """从原始像素矩阵切片裁剪，绝对无损"""
        if self.pixels is None:
            return None
            
        if not box.is_valid():
            return None
        
        # 确保坐标在图像范围内
        left = max(0, box.left)
        top = max(0, box.top)
        right = min(self.width, box.right)
        bottom = min(self.height, box.bottom)
        
        if right <= left or bottom <= top:
            return None
            
        return self.pixels[top:bottom, left:right].copy()


# ==================== 边界检测层 ====================
class EdgeDetector:
    """边界检测层：检测像素颜色突变边界"""
    
    @staticmethod
    def calculate_color_diff(pixel1: Tuple[int, int, int, int], 
                           pixel2: Tuple[int, int, int, int]) -> int:
        """计算两个像素的颜色差异（曼哈顿距离）"""
        # 考虑RGBA四个通道，确保计算不会溢出
        r_diff = abs(int(pixel1[0]) - int(pixel2[0]))
        g_diff = abs(int(pixel1[1]) - int(pixel2[1]))
        b_diff = abs(int(pixel1[2]) - int(pixel2[2]))
        a_diff = abs(int(pixel1[3]) - int(pixel2[3]))
        
        # 限制返回值范围，避免溢出
        total_diff = r_diff + g_diff + b_diff + a_diff
        return min(total_diff, 255 * 4)  # 最大可能值为255*4=1020
    
    def detect_horizontal_edge(self, pixels: PixelData, x: int, y: int, 
                              threshold: int) -> bool:
        """检测水平边界（垂直方向上的颜色突变）"""
        if y <= 0 or y >= pixels.height - 1:
            return False
        
        upper = pixels.get_pixel(x, y - 1)
        current = pixels.get_pixel(x, y)
        lower = pixels.get_pixel(x, y + 1)
        
        # 计算垂直梯度
        diff_up = self.calculate_color_diff(upper, current)
        diff_down = self.calculate_color_diff(current, lower)
        
        # 如果上下像素差异超过阈值，说明这是水平边界
        return diff_up > threshold or diff_down > threshold
    
    def detect_vertical_edge(self, pixels: PixelData, x: int, y: int,
                            threshold: int) -> bool:
        """检测垂直边界（水平方向上的颜色突变）"""
        if x <= 0 or x >= pixels.width - 1:
            return False
        
        left = pixels.get_pixel(x - 1, y)
        current = pixels.get_pixel(x, y)
        right = pixels.get_pixel(x + 1, y)
        
        # 计算水平梯度
        diff_left = self.calculate_color_diff(left, current)
        diff_right = self.calculate_color_diff(current, right)
        
        # 如果左右像素差异超过阈值，说明这是垂直边界
        return diff_left > threshold or diff_right > threshold
    
    def find_nearest_edge(self, pixels: PixelData, start_x: int, start_y: int,
                         is_vertical: bool, search_range: int, 
                         threshold: int) -> Optional[EdgePoint]:
        """
        查找最近的边界点
        
        参数:
            is_vertical: True-查找垂直边界(用于吸附左右边)
                        False-查找水平边界(用于吸附上下边)
        """
        nearest_edge = None
        min_distance = float('inf')
        
        if is_vertical:
            # 查找垂直边界（左右边吸附）
            for offset in range(search_range + 1):
                # 向右搜索
                if start_x + offset < pixels.width:
                    if self.detect_vertical_edge(pixels, start_x + offset, start_y, threshold):
                        distance = offset
                        if distance < min_distance:
                            min_distance = distance
                            nearest_edge = EdgePoint(start_x + offset, start_y, True)
                # 向左搜索
                if start_x - offset >= 0:
                    if self.detect_vertical_edge(pixels, start_x - offset, start_y, threshold):
                        distance = offset
                        if distance < min_distance:
                            min_distance = distance
                            nearest_edge = EdgePoint(start_x - offset, start_y, True)
        else:
            # 查找水平边界（上下边吸附）
            for offset in range(search_range + 1):
                # 向下搜索
                if start_y + offset < pixels.height:
                    if self.detect_horizontal_edge(pixels, start_x, start_y + offset, threshold):
                        distance = offset
                        if distance < min_distance:
                            min_distance = distance
                            nearest_edge = EdgePoint(start_x, start_y + offset, False)
                # 向上搜索
                if start_y - offset >= 0:
                    if self.detect_horizontal_edge(pixels, start_x, start_y - offset, threshold):
                        distance = offset
                        if distance < min_distance:
                            min_distance = distance
                            nearest_edge = EdgePoint(start_x, start_y - offset, False)
        
        return nearest_edge


# ==================== 交互吸附层 ====================
class MagnetController:
    """交互吸附层：管理裁剪框的吸附逻辑"""
    
    def __init__(self, pixel_data: PixelData):
        self.pixels = pixel_data
        self.edge_detector = EdgeDetector()
        self.config = MagnetConfig()
        self.crop_box = CropBox()
        
        # 拖拽状态
        self.dragging = False
        self.drag_edge = None
        self.drag_start_point = QPointF()
        self.drag_start_box = CropBox()
        
        # 吸附状态
        self.is_snapped = False
        self.snapped_edge = None
    
    def set_config(self, enable: bool, magnet_range: int, threshold: int):
        """更新吸附配置"""
        self.config.enable = enable
        self.config.range = magnet_range
        self.config.threshold = threshold
    
    def get_snapped_edge_info(self) -> Tuple[bool, Optional[str]]:
        """获取吸附状态信息"""
        return self.is_snapped, self.snapped_edge
    
    def start_drag(self, scene_pos: QPointF, edge: Optional[str]):
        """开始拖拽"""
        self.dragging = True
        self.drag_edge = edge
        self.drag_start_point = scene_pos
        self.drag_start_box = self.crop_box.copy()
        self.is_snapped = False
        self.snapped_edge = None
    
    def end_drag(self):
        """结束拖拽"""
        self.dragging = False
        self.drag_edge = None
        # 检查是否真的在边界上
        if self.crop_box.is_valid():
            # 检查左边是否在边界上
            left_snapped = self._check_edge_at_position(self.crop_box.left, self.crop_box.top + (self.crop_box.height() // 2), True)
            # 检查右边是否在边界上
            right_snapped = self._check_edge_at_position(self.crop_box.right - 1, self.crop_box.top + (self.crop_box.height() // 2), True)
            # 检查上边是否在边界上
            top_snapped = self._check_edge_at_position(self.crop_box.left + (self.crop_box.width() // 2), self.crop_box.top, False)
            # 检查下边是否在边界上
            bottom_snapped = self._check_edge_at_position(self.crop_box.left + (self.crop_box.width() // 2), self.crop_box.bottom - 1, False)
            
            # 如果任何一边在边界上，保持吸附状态
            if left_snapped or right_snapped:
                self.is_snapped = True
                self.snapped_edge = 'vertical'
            elif top_snapped or bottom_snapped:
                self.is_snapped = True
                self.snapped_edge = 'horizontal'
            else:
                # 不在边界上，重置吸附状态
                self.is_snapped = False
                self.snapped_edge = None
        else:
            # 裁剪框无效，重置吸附状态
            self.is_snapped = False
            self.snapped_edge = None
    
    def _check_edge_at_position(self, x: int, y: int, is_vertical: bool) -> bool:
        """检查指定位置是否在边界上"""
        if is_vertical:
            return self.edge_detector.detect_vertical_edge(self.pixels, x, y, self.config.threshold)
        else:
            return self.edge_detector.detect_horizontal_edge(self.pixels, x, y, self.config.threshold)
    
    def update_drag(self, scene_pos: QPointF) -> Optional[QRectF]:
        """更新拖拽位置，返回新的矩形（如果有变化）"""
        if not self.dragging or not self.drag_edge:
            return None
        
        # 计算鼠标移动距离
        delta_x = scene_pos.x() - self.drag_start_point.x()
        delta_y = scene_pos.y() - self.drag_start_point.y()
        
        # 重置吸附状态
        self.is_snapped = False
        self.snapped_edge = None
        
        # 边缘拖拽，带吸附
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
            # 左上角拖拽，同时调整左边和上边
            new_left = self._calculate_edge_position(scene_pos, 'left')
            new_top = self._calculate_edge_position(scene_pos, 'top')
            new_right = self.drag_start_box.right
            new_bottom = self.drag_start_box.bottom
        elif self.drag_edge == 'topright':
            # 右上角拖拽，同时调整右边和上边
            new_left = self.drag_start_box.left
            new_top = self._calculate_edge_position(scene_pos, 'top')
            new_right = self._calculate_edge_position(scene_pos, 'right')
            new_bottom = self.drag_start_box.bottom
        elif self.drag_edge == 'bottomleft':
            # 左下角拖拽，同时调整左边和下边
            new_left = self._calculate_edge_position(scene_pos, 'left')
            new_top = self.drag_start_box.top
            new_right = self.drag_start_box.right
            new_bottom = self._calculate_edge_position(scene_pos, 'bottom')
        elif self.drag_edge == 'bottomright':
            # 右下角拖拽，同时调整右边和下边
            new_left = self.drag_start_box.left
            new_top = self.drag_start_box.top
            new_right = self._calculate_edge_position(scene_pos, 'right')
            new_bottom = self._calculate_edge_position(scene_pos, 'bottom')
        else:
            return None
        
        # 边界约束
        new_left = max(0, new_left)
        new_top = max(0, new_top)
        new_right = min(self.pixels.width, new_right)
        new_bottom = min(self.pixels.height, new_bottom)
        
        # 确保矩形有效
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
        
        # 检查是否有变化
        if (new_left == self.crop_box.left and new_top == self.crop_box.top and
            new_right == self.crop_box.right and new_bottom == self.crop_box.bottom):
            return None
        
        # 更新裁剪框
        self.crop_box.left = new_left
        self.crop_box.top = new_top
        self.crop_box.right = new_right
        self.crop_box.bottom = new_bottom
        
        return self.crop_box.to_rectf()
    
    def _calculate_edge_position(self, scene_pos: QPointF, edge: str) -> int:
        """计算边缘位置（带吸附）"""
        # 确保返回整数坐标，实现像素级精度
        if not self.config.enable:
            return int(getattr(self, f'_calculate_raw_{edge}_pos')(scene_pos) + 0.5)
        
        # 根据边缘确定采样点和搜索方向
        if edge in ['left', 'right']:
            # 垂直边缘
            is_vertical_search = True
            base_x = int(scene_pos.x() + 0.5)
            base_y = int(self.drag_start_box.top + (self.drag_start_box.bottom - self.drag_start_box.top) / 2 + 0.5)
            
            # 在y方向采样多个点
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
                # 选择最常见的边缘位置
                from collections import Counter
                most_common = Counter(edges_found).most_common(1)
                if most_common:
                    self.is_snapped = True
                    self.snapped_edge = 'vertical'
                    return most_common[0][0]
                else:
                    # 无边缘时，确保坐标为整数
                    self.is_snapped = False
                    self.snapped_edge = None
                    return int(getattr(self, f'_calculate_raw_{edge}_pos')(scene_pos) + 0.5)
        
        else:  # 'top', 'bottom'
            # 水平边缘
            is_vertical_search = False
            base_x = int(self.drag_start_box.left + (self.drag_start_box.right - self.drag_start_box.left) / 2 + 0.5)
            base_y = int(scene_pos.y() + 0.5)
            
            # 在x方向采样多个点
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
                # 选择最常见的边缘位置
                from collections import Counter
                most_common = Counter(edges_found).most_common(1)
                if most_common:
                    self.is_snapped = True
                    self.snapped_edge = 'horizontal'
                    return most_common[0][0]
                else:
                    # 无边缘时，确保坐标为整数
                    self.is_snapped = False
                    self.snapped_edge = None
                    return int(getattr(self, f'_calculate_raw_{edge}_pos')(scene_pos) + 0.5)
        
        # 无吸附边界，返回原始计算位置（确保整数）
        return int(getattr(self, f'_calculate_raw_{edge}_pos')(scene_pos) + 0.5)
    
    def _get_sample_points(self, center: int, max_value: int, vertical: bool) -> List[int]:
        """获取采样点位置"""
        points = [center]
        
        # 添加偏移点
        offsets = [1, 2, -1, -2]
        for offset in offsets:
            point = center + offset
            if 0 <= point < max_value:
                points.append(point)
        
        # 去重并限制数量
        return list(set(points))[:5]
    
    def _calculate_raw_left_pos(self, scene_pos: QPointF) -> int:
        """计算左边位置（无吸附）"""
        delta_x = scene_pos.x() - self.drag_start_point.x()
        new_left = int(self.drag_start_box.left + delta_x + 0.5)
        return max(0, min(new_left, self.crop_box.right - 1))
    
    def _calculate_raw_right_pos(self, scene_pos: QPointF) -> int:
        """计算右边位置（无吸附）"""
        delta_x = scene_pos.x() - self.drag_start_point.x()
        new_right = int(self.drag_start_box.right + delta_x + 0.5)
        return min(self.pixels.width, max(new_right, self.crop_box.left + 1))
    
    def _calculate_raw_top_pos(self, scene_pos: QPointF) -> int:
        """计算顶边位置（无吸附）"""
        delta_y = scene_pos.y() - self.drag_start_point.y()
        new_top = int(self.drag_start_box.top + delta_y + 0.5)
        return max(0, min(new_top, self.crop_box.bottom - 1))
    
    def _calculate_raw_bottom_pos(self, scene_pos: QPointF) -> int:
        """计算底边位置（无吸附）"""
        delta_y = scene_pos.y() - self.drag_start_point.y()
        new_bottom = int(self.drag_start_box.bottom + delta_y + 0.5)
        return min(self.pixels.height, max(new_bottom, self.crop_box.top + 1))
    
    def get_edge_at_position(self, scene_pos: QPointF, margin: int = 5) -> Optional[str]:
        """获取场景位置对应的裁剪框边缘"""
        if not self.crop_box.is_valid():
            return None
        
        x, y = scene_pos.x(), scene_pos.y()
        left, top = self.crop_box.left, self.crop_box.top
        right, bottom = self.crop_box.right, self.crop_box.bottom
        
        # 检查是否在角上
        corner_margin = margin * 1.5  # 角的检测范围稍大
        if abs(x - left) <= corner_margin and abs(y - top) <= corner_margin:
            return 'topleft'
        elif abs(x - right) <= corner_margin and abs(y - top) <= corner_margin:
            return 'topright'
        elif abs(x - left) <= corner_margin and abs(y - bottom) <= corner_margin:
            return 'bottomleft'
        elif abs(x - right) <= corner_margin and abs(y - bottom) <= corner_margin:
            return 'bottomright'
        # 检查是否在边缘附近（扩大边缘检测范围）
        edge_margin = 8  # 增加边缘检测范围
        if abs(x - left) <= edge_margin and top <= y <= bottom:
            return 'left'
        elif abs(x - right) <= edge_margin and top <= y <= bottom:
            return 'right'
        elif abs(y - top) <= edge_margin and left <= x <= right:
            return 'top'
        elif abs(y - bottom) <= edge_margin and left <= x <= right:
            return 'bottom'
        # 禁止整体移动，移除 'all' 返回值
        
        return None


# 信号源类
class CropBoxSignals(QObject):
    box_changed = pyqtSignal(QRectF)
    snapped_state_changed = pyqtSignal(bool, str)

# ==================== 裁剪框图形项 ====================
class CropBoxItem(QGraphicsRectItem):
    """裁剪框图形项，处理拖拽和吸附"""
    
    def __init__(self, pixel_data: PixelData):
        super().__init__()
        
        # 创建信号源对象
        self.signals = CropBoxSignals()
        
        self.pixel_data = pixel_data
        self.magnet_controller = MagnetController(pixel_data)
        
        # 初始时创建一个覆盖整个图像的透明矩形，以便接收鼠标事件
        if pixel_data.width > 0 and pixel_data.height > 0:
            self.magnet_controller.crop_box = CropBox()
            self.setRect(0, 0, pixel_data.width, pixel_data.height)
        else:
            self.magnet_controller.crop_box = CropBox()
            self.setRect(QRectF())  # 设置为空矩形
        
        # 交互状态
        self.current_edge = None
        self.dragging = False
        self.drawing = False
        self.draw_start = QPointF()
        self.draw_end = QPointF()
        
        # 标记是否已经创建过裁剪框
        self.crop_box_created = False  # 初始时未创建裁剪框
        
        # 边框状态（记录每个边是否在边界上）
        self.edge_states = {
            'left': False,
            'top': False,
            'right': False,
            'bottom': False
        }
        
        # 显示设置
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, True)
        
        # 画笔和画刷
        self.normal_pen = QPen(QColor(255, 0, 0), 1)  # 红色边框
        self.snapped_pen = QPen(QColor(0, 255, 0), 2)  # 绿色边框（吸附时）
        self.drawing_pen = QPen(QColor(255, 255, 0), 1, Qt.PenStyle.DashLine)  # 黄色虚线（绘制时）
        
        self.normal_brush = QBrush(QColor(255, 255, 255, 30))  # 半透明白色填充
        self.setPen(self.normal_pen)
        self.setBrush(self.normal_brush)
        
        # 控制点大小
        self.control_point_size = 10
        
    def set_magnet_config(self, enable: bool, magnet_range: int, threshold: int):
        """设置吸附配置"""
        self.magnet_controller.set_config(enable, magnet_range, threshold)
    
    def get_crop_box(self) -> CropBox:
        """获取裁剪框"""
        return self.magnet_controller.crop_box
    
    def set_crop_box(self, box: CropBox):
        """设置裁剪框"""
        self.magnet_controller.crop_box = box
        self.setRect(box.to_rectf())
        self.signals.box_changed.emit(box.to_rectf())
    
    def reset_to_full_image(self):
        """重置为整个图像"""
        if self.pixel_data.width > 0 and self.pixel_data.height > 0:
            box = CropBox(0, 0, self.pixel_data.width, self.pixel_data.height)
            self.set_crop_box(box)
    
    def paint(self, painter, option, widget=None):
        """绘制裁剪框"""
        # 保存画笔状态
        painter.save()
        
        # 如果正在绘制临时矩形，绘制它
        if self.drawing:
            painter.setPen(self.drawing_pen)
            painter.setBrush(QBrush(QColor(255, 255, 0, 50)))  # 半透明黄色填充
            rect = QRectF(self.draw_start, self.draw_end).normalized()
            painter.drawRect(rect)
            painter.restore()
            return
        
        # 只有当裁剪框有效时才绘制
        if not self.magnet_controller.crop_box.is_valid():
            painter.restore()
            return
        
        # 绘制填充
        painter.setBrush(self.normal_brush)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(self.rect())
        
        # 绘制边框（每个边独立颜色）
        rect = self.rect()
        
        # 绘制左边
        painter.setPen(self.snapped_pen if self.edge_states['left'] else self.normal_pen)
        painter.drawLine(rect.topLeft(), rect.bottomLeft())
        
        # 绘制上边
        painter.setPen(self.snapped_pen if self.edge_states['top'] else self.normal_pen)
        painter.drawLine(rect.topLeft(), rect.topRight())
        
        # 绘制右边
        painter.setPen(self.snapped_pen if self.edge_states['right'] else self.normal_pen)
        painter.drawLine(rect.topRight(), rect.bottomRight())
        
        # 绘制下边
        painter.setPen(self.snapped_pen if self.edge_states['bottom'] else self.normal_pen)
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        
        # 绘制控制点
        half_point = self.control_point_size // 2
        
        # 四个角（绿色）
        painter.setBrush(QBrush(QColor(0, 255, 0)))
        for point in [
            rect.topLeft(),
            rect.topRight(),
            rect.bottomLeft(),
            rect.bottomRight()
        ]:
            painter.drawRect(int(point.x() - half_point), int(point.y() - half_point), 
                           self.control_point_size, self.control_point_size)
        
        # 四个边中点（蓝色）
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
        """鼠标按下事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = event.scenePos()
            
            # 检查是否在裁剪框上
            edge = self.magnet_controller.get_edge_at_position(scene_pos)
            
            if not self.crop_box_created:
                # 只有当裁剪框尚未创建时，才允许绘制新的裁剪框
                # 开始绘制新裁剪框
                self.drawing = True
                self.draw_start = scene_pos
                self.draw_end = scene_pos
                event.accept()
            elif edge:
                # 开始拖拽裁剪框
                self.dragging = True
                self.current_edge = edge
                self.magnet_controller.start_drag(scene_pos, edge)
                event.accept()
            else:
                # 点击空白区域，不做处理
                event.ignore()
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        scene_pos = event.scenePos()
        
        if self.dragging and self.current_edge:
            # 更新拖拽
            new_rect = self.magnet_controller.update_drag(scene_pos)
            if new_rect:
                self.setRect(new_rect)
                self.signals.box_changed.emit(new_rect)
            
            # 发送吸附状态变化信号
            is_snapped, edge_type = self.magnet_controller.get_snapped_edge_info()
            self.signals.snapped_state_changed.emit(is_snapped, edge_type)
            
            # 实时更新边框状态
            self._update_edge_states()
            
            # 更新光标
            self._update_cursor(self.current_edge)
            
        elif self.drawing:
            # 更新绘制
            self.draw_end = scene_pos
            # 绘制临时矩形
            self.update()
        
        else:
            # 始终更新悬停光标，不做条件判断
            edge = self.magnet_controller.get_edge_at_position(scene_pos)
            self._update_cursor(edge)
            self.current_edge = edge
        
        event.accept()
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.dragging:
                # 结束拖拽
                self.dragging = False
                self.magnet_controller.end_drag()
                
                # 更新边框状态
                self._update_edge_states()
                
                self.current_edge = None
                # 发送最终吸附状态
                is_snapped, edge_type = self.magnet_controller.get_snapped_edge_info()
                self.signals.snapped_state_changed.emit(is_snapped, edge_type)
                
            elif self.drawing:
                # 结束绘制，创建新裁剪框
                self.drawing = False
                
                # 确保矩形有效
                rect = QRectF(self.draw_start, self.draw_end).normalized()
                
                # 约束在图像范围内
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
                    # 标记裁剪框已创建
                    self.crop_box_created = True
                    # 更新边框状态
                    self._update_edge_states()
        
        event.accept()
    
    def _update_edge_states(self):
        """更新边框状态"""
        if not self.magnet_controller.crop_box.is_valid():
            return
        
        box = self.magnet_controller.crop_box
        
        # 检查左边是否在边界上或图片边缘
        left_on_edge = box.left == 0 or self._is_edge_at_position(box.left, box.top + box.height() // 2, True)
        self.edge_states['left'] = left_on_edge
        
        # 检查上边是否在边界上或图片边缘
        top_on_edge = box.top == 0 or self._is_edge_at_position(box.left + box.width() // 2, box.top, False)
        self.edge_states['top'] = top_on_edge
        
        # 检查右边是否在边界上或图片边缘
        right_on_edge = box.right == self.pixel_data.width or self._is_edge_at_position(box.right - 1, box.top + box.height() // 2, True)
        self.edge_states['right'] = right_on_edge
        
        # 检查下边是否在边界上或图片边缘
        bottom_on_edge = box.bottom == self.pixel_data.height or self._is_edge_at_position(box.left + box.width() // 2, box.bottom - 1, False)
        self.edge_states['bottom'] = bottom_on_edge
        
        # 触发重绘
        self.update()
    
    def _is_edge_at_position(self, x: int, y: int, is_vertical: bool) -> bool:
        """检查指定位置是否在边界上"""
        if is_vertical:
            return self.magnet_controller.edge_detector.detect_vertical_edge(self.pixel_data, x, y, self.magnet_controller.config.threshold)
        else:
            return self.magnet_controller.edge_detector.detect_horizontal_edge(self.pixel_data, x, y, self.magnet_controller.config.threshold)
    
    def _update_cursor(self, edge: Optional[str]):
        """根据边缘位置更新光标"""
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


# ==================== 图形视图 ====================
class ImageGraphicsView(QGraphicsView):
    """自定义图形视图，支持缩放和平移"""
    
    def __init__(self):
        super().__init__()
        
        # 创建场景
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        # 图形项
        self.pixmap_item = None
        self.crop_box_item = None
        self.grid_item = None
        
        # 像素数据
        self.pixel_data = PixelData()
        
        # 视图设置
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        
        # 缩放级别
        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.zoom_step = 1.25
        
        # 像素网格
        self.show_grid = False
        self.grid_pen = QPen(QColor(100, 100, 100, 50), 0)  # Cosmetic pen，宽度为0
    
    def load_image(self, image_path: str) -> bool:
        """加载图像"""
        print(f"开始加载图像: {image_path}")
        try:
            if not self.pixel_data.load_image(image_path):
                print("像素数据加载失败")
                return False
            
            print(f"图像加载成功: {self.pixel_data.width}x{self.pixel_data.height}")
            
            # 清空场景
            self.scene.clear()
            
            # 创建QPixmap
            print("创建QImage...")
            qimage = QImage(
                self.pixel_data.pixels.data,
                self.pixel_data.width,
                self.pixel_data.height,
                self.pixel_data.width * 4,
                QImage.Format.Format_RGBA8888
            )
            print("QImage创建成功")
            
            pixmap = QPixmap.fromImage(qimage)
            print("QPixmap创建成功")
            
            # 设置像素图项
            self.pixmap_item = QGraphicsPixmapItem(pixmap)
            self.pixmap_item.setTransformationMode(Qt.TransformationMode.FastTransformation)  # 最近邻插值
            self.scene.addItem(self.pixmap_item)
            print("像素图项添加成功")
            
            # 设置场景大小
            self.scene.setSceneRect(0, 0, self.pixel_data.width, self.pixel_data.height)
            print("场景大小设置成功")
            
            # 创建裁剪框
            print("创建CropBoxItem...")
            self.crop_box_item = CropBoxItem(self.pixel_data)
            print("CropBoxItem创建成功")
            self.scene.addItem(self.crop_box_item)
            print("裁剪框添加成功")
            
            # 创建像素网格（使用QGraphicsPathItem）
            self.grid_item = QGraphicsPathItem()
            self.grid_item.setPen(self.grid_pen)
            self.grid_item.setZValue(10)  # 最上层
            self.grid_item.setVisible(False)
            self.scene.addItem(self.grid_item)
            print("网格添加成功")
            
            # 自适应显示
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.zoom_level = self.transform().m11()  # 获取当前缩放级别
            print(f"自适应显示完成，缩放级别: {self.zoom_level}")
            
            return True
        except Exception as e:
            print(f"加载图像时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_crop_box_item(self) -> CropBoxItem:
        """获取裁剪框项"""
        return self.crop_box_item
    
    def wheelEvent(self, event: QWheelEvent):
        """滚轮事件：缩放"""
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Ctrl+滚轮：缩放
            zoom_in = event.angleDelta().y() > 0
            factor = self.zoom_step if zoom_in else 1.0 / self.zoom_step
            
            # 计算新缩放级别
            new_zoom = self.zoom_level * factor
            if self.min_zoom <= new_zoom <= self.max_zoom:
                # 以鼠标位置为中心缩放
                old_pos = self.mapToScene(event.position().toPoint())
                self.scale(factor, factor)
                new_pos = self.mapToScene(event.position().toPoint())
                delta = new_pos - old_pos
                self.translate(-delta.x(), -delta.y())
                self.zoom_level = new_zoom
                
                # 更新像素网格显示
                self._update_grid_visibility()
            
            event.accept()
        else:
            # 普通滚轮：垂直滚动
            super().wheelEvent(event)
    
    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下事件"""
        # 只有在平移模式或点击空白区域时才允许拖拽
        if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
            super().mousePressEvent(event)
        else:
            # 检查是否点击在空白区域
            item = self.itemAt(event.pos())
            if not item or item == self.grid_item:
                # 点击空白区域，切换到平移模式
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                # 模拟鼠标按下事件
                super().mousePressEvent(event)
            else:
                super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件"""
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放事件"""
        # 释放鼠标时退出平移模式
        if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        super().mouseReleaseEvent(event)
    
    def keyPressEvent(self, event: QKeyEvent):
        """键盘事件"""
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            # 空格键：临时切换到平移模式
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            event.accept()
        elif event.key() == Qt.Key.Key_0 and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Ctrl+0：适应窗口
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.zoom_level = self.transform().m11()
            self._update_grid_visibility()
            event.accept()
        elif event.key() == Qt.Key.Key_1 and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Ctrl+1：实际大小
            self.resetTransform()
            self.zoom_level = 1.0
            self._update_grid_visibility()
            event.accept()
        elif event.key() == Qt.Key.Key_G and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Ctrl+G：切换像素网格
            self.show_grid = not self.show_grid
            self._update_grid_visibility()
            event.accept()
        else:
            super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event: QKeyEvent):
        """键盘释放事件"""
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            # 释放空格键：恢复模式
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            event.accept()
        else:
            super().keyReleaseEvent(event)
    
    def _update_grid_visibility(self):
        """更新像素网格可见性"""
        if not self.grid_item:
            return
        
        # 只在放大到一定程度时显示网格
        if self.show_grid and self.zoom_level >= 2.0:
            self.grid_item.setVisible(True)
            self._draw_pixel_grid()
        else:
            self.grid_item.setVisible(False)
    
    def _draw_pixel_grid(self):
        """绘制像素网格"""
        if not self.grid_item or not self.pixel_data:
            return
        
        # 创建网格路径
        path = QPainterPath()
        
        # 垂直线
        for x in range(0, self.pixel_data.width + 1):
            path.moveTo(x, 0)
            path.lineTo(x, self.pixel_data.height)
        
        # 水平线
        for y in range(0, self.pixel_data.height + 1):
            path.moveTo(0, y)
            path.lineTo(self.pixel_data.width, y)
        
        # 设置网格路径
        self.grid_item.setPath(path)


# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        
        # 初始化UI
        self.init_ui()
        
        # 当前文件路径
        self.current_file = None
        
        # 撤销/重做栈
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo_steps = 20
        
        # 信号连接状态
        self.crop_box_connections = []
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("像素级边界吸附无损裁剪工具 v3.0")
        self.setGeometry(100, 100, 1200, 800)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # 创建工具栏
        self.create_toolbar()
        main_layout.addWidget(self.toolbar)
        
        # 控制面板
        control_group = QGroupBox("吸附控制")
        control_layout = QHBoxLayout()
        
        # 吸附开关
        self.magnet_checkbox = QCheckBox("启用边界吸附")
        self.magnet_checkbox.setChecked(True)
        self.magnet_checkbox.stateChanged.connect(self.update_magnet_config)
        control_layout.addWidget(self.magnet_checkbox)
        
        # 吸附范围
        control_layout.addWidget(QLabel("吸附范围:"))
        self.range_spinbox = QSpinBox()
        self.range_spinbox.setRange(1, 10)
        self.range_spinbox.setValue(3)
        self.range_spinbox.valueChanged.connect(self.update_magnet_config)
        control_layout.addWidget(self.range_spinbox)
        control_layout.addWidget(QLabel("px"))
        
        # 颜色差阈值
        control_layout.addWidget(QLabel("敏感度:"))
        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setRange(1, 100)
        self.threshold_spinbox.setValue(10)
        self.threshold_spinbox.valueChanged.connect(self.update_magnet_config)
        control_layout.addWidget(self.threshold_spinbox)
        
        control_layout.addStretch()
        
        # 缩放级别显示
        self.zoom_label = QLabel("缩放: 100%")
        control_layout.addWidget(self.zoom_label)
        
        # 尺寸显示
        self.size_label = QLabel("尺寸: 0 x 0 像素")
        control_layout.addWidget(self.size_label)
        
        # 吸附状态显示
        self.snap_status_label = QLabel("吸附: 未激活")
        self.snap_status_label.setStyleSheet("color: gray;")
        control_layout.addWidget(self.snap_status_label)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # 创建图形视图
        self.graphics_view = ImageGraphicsView()
        main_layout.addWidget(self.graphics_view, 1)
        
        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪 - 使用Ctrl+滚轮缩放，空格键拖拽平移，Ctrl+0适应窗口，Ctrl+1实际大小，Ctrl+G切换网格")
        
        # 快捷键
        self.setup_shortcuts()
    
    def create_toolbar(self):
        """创建工具栏"""
        self.toolbar = QToolBar("主工具栏")
        self.toolbar.setMovable(False)
        
        # 打开按钮
        open_action = QAction("打开图像", self)
        open_action.triggered.connect(self.open_image)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.toolbar.addAction(open_action)
        
        # 保存按钮
        self.save_action = QAction("保存裁剪", self)
        self.save_action.triggered.connect(self.save_cropped_image)
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.setEnabled(False)
        self.toolbar.addAction(self.save_action)
        
        self.toolbar.addSeparator()
        
        # 适应窗口
        fit_action = QAction("适应窗口", self)
        fit_action.triggered.connect(self.fit_to_window)
        fit_action.setShortcut("Ctrl+0")
        self.toolbar.addAction(fit_action)
        
        # 实际大小
        actual_size_action = QAction("实际大小", self)
        actual_size_action.triggered.connect(self.actual_size)
        actual_size_action.setShortcut("Ctrl+1")
        self.toolbar.addAction(actual_size_action)
        
        # 缩放滑块
        self.toolbar.addWidget(QLabel(" 缩放:"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 1000)  # 10% 到 1000%
        self.zoom_slider.setValue(100)
        self.zoom_slider.setMaximumWidth(150)
        self.zoom_slider.valueChanged.connect(self.zoom_slider_changed)
        self.toolbar.addWidget(self.zoom_slider)
        
        self.toolbar.addSeparator()
        
        # 重置裁剪框
        reset_action = QAction("重置裁剪框", self)
        reset_action.triggered.connect(self.reset_crop_box)
        reset_action.setEnabled(False)
        self.reset_action = reset_action
        self.toolbar.addAction(reset_action)
        
        # 网格开关
        grid_action = QAction("显示网格", self)
        grid_action.triggered.connect(self.toggle_grid)
        grid_action.setShortcut("Ctrl+G")
        grid_action.setCheckable(True)
        grid_action.setChecked(False)
        self.grid_action = grid_action
        self.toolbar.addAction(grid_action)
        
        self.addToolBar(self.toolbar)
    
    def setup_shortcuts(self):
        """设置快捷键"""
        # 撤销
        undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo_shortcut.activated.connect(self.undo)
        
        # 重做
        redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, self)
        redo_shortcut.activated.connect(self.redo)
    
    def open_image(self):
        """打开图像文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.webp)"
        )
        
        if file_path:
            self.current_file = file_path
            if self.graphics_view.load_image(file_path):
                # 启用相关功能
                self.save_action.setEnabled(True)
                self.reset_action.setEnabled(True)
                
                # 更新状态
                self.status_bar.showMessage(f"已加载: {file_path}")
                
                # 保存初始状态到撤销栈
                self.save_to_undo_stack()
                
                # 连接裁剪框变化信号
                crop_box_item = self.graphics_view.get_crop_box_item()
                if crop_box_item:
                    # 断开旧连接
                    for connection in self.crop_box_connections:
                        try:
                            connection[0].disconnect(connection[1])
                        except:
                            pass
                    self.crop_box_connections.clear()
                    
                    # 连接新信号
                    connection1 = (crop_box_item.signals.box_changed, self.on_crop_box_changed)
                    crop_box_item.signals.box_changed.connect(self.on_crop_box_changed)
                    self.crop_box_connections.append(connection1)
                    
                    connection2 = (crop_box_item.signals.snapped_state_changed, self.on_snapped_state_changed)
                    crop_box_item.signals.snapped_state_changed.connect(self.on_snapped_state_changed)
                    self.crop_box_connections.append(connection2)
                    
                    self.update_magnet_config()
                    self.on_crop_box_changed()
            else:
                QMessageBox.critical(self, "加载失败", f"无法加载图像: {file_path}")
    
    def save_cropped_image(self):
        """保存裁剪后的图像"""
        if not self.current_file or not self.graphics_view.pixel_data:
            return
        
        crop_box_item = self.graphics_view.get_crop_box_item()
        if not crop_box_item:
            return
        
        # 获取裁剪框
        crop_box = crop_box_item.get_crop_box()
        if not crop_box.is_valid():
            QMessageBox.warning(self, "无效裁剪框", "请先绘制有效的裁剪区域")
            return
        
        # 生成默认文件名
        import os
        dir_name = os.path.dirname(self.current_file)
        base_name = os.path.basename(self.current_file)
        name, ext = os.path.splitext(base_name)
        default_name = f"{name}_cropped.png"
        
        # 选择保存路径
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存裁剪图像", 
            os.path.join(dir_name, default_name),
            "PNG图像 (*.png);;所有文件 (*)"
        )
        
        if file_path:
            # 执行裁剪
            cropped_data = self.graphics_view.pixel_data.crop(crop_box)
            if cropped_data is not None:
                try:
                    # 转换为PIL图像并保存
                    cropped_image = Image.fromarray(cropped_data, 'RGBA')
                    cropped_image.save(file_path, 'PNG', compress_level=0)  # 无压缩PNG
                    
                    self.status_bar.showMessage(f"图像已保存: {file_path}")
                    QMessageBox.information(self, "保存成功", f"图像已保存到:\n{file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "保存失败", f"保存图像时发生错误:\n{str(e)}")
            else:
                QMessageBox.critical(self, "裁剪失败", "无法裁剪图像")
    
    def reset_crop_box(self):
        """重置裁剪框到整个图像"""
        crop_box_item = self.graphics_view.get_crop_box_item()
        if crop_box_item:
            crop_box_item.reset_to_full_image()
    
    def update_magnet_config(self):
        """更新吸附配置"""
        crop_box_item = self.graphics_view.get_crop_box_item()
        if crop_box_item:
            crop_box_item.set_magnet_config(
                self.magnet_checkbox.isChecked(),
                self.range_spinbox.value(),
                self.threshold_spinbox.value()
            )
    
    def on_crop_box_changed(self):
        """裁剪框变化时的处理"""
        # 保存到撤销栈
        self.save_to_undo_stack()
        # 更新尺寸显示
        self.update_size_display()
    
    def on_snapped_state_changed(self, is_snapped: bool, edge_type: Optional[str]):
        """吸附状态变化时的处理"""
        if is_snapped:
            edge_text = "垂直" if edge_type == "vertical" else "水平"
            self.snap_status_label.setText(f"吸附: 已吸附到{edge_text}边界")
            self.snap_status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.snap_status_label.setText("吸附: 未激活")
            self.snap_status_label.setStyleSheet("color: gray;")
    
    def update_size_display(self):
        """更新尺寸和吸附状态显示"""
        crop_box_item = self.graphics_view.get_crop_box_item()
        if crop_box_item:
            crop_box = crop_box_item.get_crop_box()
            self.size_label.setText(f"尺寸: {crop_box.width()} x {crop_box.height()} 像素")
            
            # 更新缩放显示
            self.update_zoom_display()
    
    def update_zoom_display(self):
        """更新缩放显示"""
        zoom_percent = int(self.graphics_view.zoom_level * 100)
        self.zoom_label.setText(f"缩放: {zoom_percent}%")
        # 防止滑块值变化触发缩放
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(zoom_percent)
        self.zoom_slider.blockSignals(False)
    
    def zoom_slider_changed(self, value):
        """缩放滑块变化"""
        if not self.graphics_view or not hasattr(self.graphics_view, 'pixel_data') or not self.graphics_view.pixel_data:
            return
        
        target_zoom = max(self.graphics_view.min_zoom, 
                         min(self.graphics_view.max_zoom, value / 100.0))
        current_zoom = self.graphics_view.zoom_level
        
        if abs(target_zoom - current_zoom) > 0.01:
            # 使用重置变换的方式，避免累积误差
            self.graphics_view.resetTransform()
            self.graphics_view.scale(target_zoom, target_zoom)
            self.graphics_view.zoom_level = target_zoom
            self.graphics_view._update_grid_visibility()
            self.update_zoom_display()
    
    def fit_to_window(self):
        """适应窗口大小"""
        if self.graphics_view.scene():
            self.graphics_view.fitInView(
                self.graphics_view.scene().sceneRect(),
                Qt.AspectRatioMode.KeepAspectRatio
            )
            self.graphics_view.zoom_level = self.graphics_view.transform().m11()
            self.graphics_view._update_grid_visibility()
            self.update_zoom_display()
    
    def actual_size(self):
        """实际大小"""
        self.graphics_view.resetTransform()
        self.graphics_view.zoom_level = 1.0
        self.graphics_view._update_grid_visibility()
        self.update_zoom_display()
    
    def toggle_grid(self):
        """切换网格显示"""
        self.graphics_view.show_grid = self.grid_action.isChecked()
        self.graphics_view._update_grid_visibility()
    
    def save_to_undo_stack(self):
        """保存当前状态到撤销栈"""
        crop_box_item = self.graphics_view.get_crop_box_item()
        if not crop_box_item:
            return
        
        crop_box = crop_box_item.get_crop_box()
        
        # 检查是否与栈顶状态相同
        if self.undo_stack and self.undo_stack[-1] == crop_box:
            return
        
        # 保存到撤销栈
        self.undo_stack.append(crop_box.copy())
        
        # 限制栈大小
        if len(self.undo_stack) > self.max_undo_steps:
            self.undo_stack.pop(0)
        
        # 清空重做栈
        self.redo_stack.clear()
    
    def undo(self):
        """撤销"""
        if len(self.undo_stack) < 2:  # 需要至少2个状态才能撤销
            return
        
        crop_box_item = self.graphics_view.get_crop_box_item()
        if not crop_box_item:
            return
        
        # 当前状态保存到重做栈
        current_box = crop_box_item.get_crop_box()
        self.redo_stack.append(current_box.copy())
        
        # 弹出上一个状态
        self.undo_stack.pop()  # 移除当前状态
        prev_box = self.undo_stack[-1]  # 获取上一个状态
        
        # 应用上一个状态
        crop_box_item.set_crop_box(prev_box.copy())
    
    def redo(self):
        """重做"""
        if not self.redo_stack:
            return
        
        crop_box_item = self.graphics_view.get_crop_box_item()
        if not crop_box_item:
            return
        
        # 获取重做状态
        redo_box = self.redo_stack.pop()
        
        # 当前状态保存到撤销栈
        current_box = crop_box_item.get_crop_box()
        self.undo_stack.append(current_box.copy())
        
        # 应用重做状态
        crop_box_item.set_crop_box(redo_box.copy())
    
    def closeEvent(self, event):
        """关闭事件"""
        # 清理资源
        if hasattr(self, 'graphics_view') and self.graphics_view.scene:
            self.graphics_view.scene.clear()
        event.accept()


# ==================== 主程序入口 ====================
def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    # 设置应用程序图标和元数据
    app.setApplicationName("像素级边界吸附无损裁剪工具")
    app.setApplicationVersion("3.0")
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
