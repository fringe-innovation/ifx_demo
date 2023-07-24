import numpy as np
from scipy import signal

'''
OS-CFAR算法中三个主要参数的设置依据如下:
1.滑动窗口大小(win_size)
  依据目标和噪声的相关性 - 相关性越大,需要更大的窗口提取更多独立参考单元
  依据噪声均值和方差 - 噪声方差大需要更多样例估计准确均值和方差
  一般设置为8-24不等,需trade off between环绕量和独立样例数
2.保护单元大小(guard_size)
  依据目标最大期望尺寸 - 需要充分排除目标本身影响
  一般设置为1-3不等
3.假警报率(pfa)
  依据对虚警率的容限 - pfa越小,丢失目标概率越小
  一般设置为10^-2 - 10^-4
  需要trade off between丢目标和虚警
综上,三个参数的设置需要综合考虑目标特征、噪声特征、检测要求等,通过trade off得到最优配置。具体数值需要通过大量仿真试验确定。但应遵循上述的设计依据和经验范围。
'''

class OSCFAR:

    def __init__(self, win_size, guard_size, pfa):
        self.win_size = win_size # 滑动窗口,即参考单元
        self.guard_size = guard_size # 保护单元，为了排除目标本身的影响,在滑动窗口中需要设置不参与计算的保护单元
        self.pfa = pfa # 假警报率,也就是检测算法本身产生错误警报的概率

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
