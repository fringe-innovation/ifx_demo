import numpy as np
from scipy import signal


class OSCFAR:

    def __init__(self, win_size, guard_size, pfa):
        self.win_size = win_size
        self.guard_size = guard_size
        self.pfa = pfa

    def smooth(self, input_map):
        '''平滑滤波预处理'''
        kernel = np.ones((5, 5)) / 25
        return signal.convolve2d(input_map, kernel, mode='same')

    def get_threshold(self, ref_cells):
        '''基于参考单元计算阈值'''
        sorted_ref = np.sort(ref_cells.flatten())
        thresh = sorted_ref[int((1 - self.pfa) * len(sorted_ref))]
        return thresh

    def detect(self, input_map):

        # 1. 平滑滤波
        input_map = self.smooth(input_map)

        h, w = input_map.shape
        output_map = np.zeros_like(input_map)

        # 2. 创建掩码矩阵
        mask = np.ones((self.win_size, self.win_size))
        mask[self.guard_size:-self.guard_size,
        self.guard_size:-self.guard_size] = 0

        # 3. 遍历所有单元
        for r in range(self.win_size // 2, h - self.win_size // 2):
            for c in range(self.win_size // 2, w - self.win_size // 2):

                # 4. 提取检测单元
                cell = input_map[r, c]

                # 5. 提取参考单元
                ref_cells = input_map[r - self.win_size // 2:r + self.win_size // 2,
                            c - self.win_size // 2:c + self.win_size // 2] * mask

                # 6. 计算阈值
                thresh = self.get_threshold(ref_cells)

                # 7. 判断目标
                if cell > thresh:
                    output_map[r, c] = cell

        return output_map