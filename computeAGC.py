import cmath


def compute_agc(datain):
    data_in_length = len(datain)
    lenblock = 20
    thresh = 2.0
    scale_value_sum = 0
    for indexTemp in range(lenblock, data_in_length - 1):
        sum_energy = 0
        for indexInner in range(1, lenblock + 1):
            index_curr = indexTemp - lenblock + indexInner
            sum_energy = sum_energy + datain[index_curr] * datain[index_curr]
        if sum_energy > thresh:
            scale_value = cmath.sqrt(thresh / sum_energy)
            scale_value_sum = scale_value + 1
            for indexInner in range(1, lenblock + 1):
                index_curr = indexTemp - lenblock + indexInner
                datain[index_curr] = datain[index_curr] * scale_value
    res = datain
    return res, scale_value_sum
