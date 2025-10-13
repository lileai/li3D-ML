import numpy as np
import json


def is_homogeneous(matrix):
    """
    检查矩阵是否为齐次变换矩阵
    :param matrix: 输入矩阵
    :return: 如果是齐次变换矩阵返回 True，否则返回 False
    """
    if matrix.shape != (4, 4):
        return False
    return np.allclose(matrix[3, :], [0, 0, 0, 1])


def to_homogeneous(matrix):
    """
    将矩阵转换为齐次变换矩阵
    :param matrix: 输入矩阵
    :return: 转换后的齐次变换矩阵
    """
    if matrix.shape == (3, 3):
        # 如果输入矩阵是 3x3 的旋转矩阵，转换为 4x4 的齐次变换矩阵
        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[:3, :3] = matrix
    elif matrix.shape == (3, 4):
        # 如果输入矩阵是 3x4 的变换矩阵，转换为 4x4 的齐次变换矩阵
        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[:3, :] = matrix
    else:
        raise ValueError("输入矩阵的维度不支持转换为齐次变换矩阵")
    return homogeneous_matrix


def left_multiply_matrices(matrices):
    """
    左乘多个矩阵
    :param matrices: 一个包含多个矩阵的列表，每个矩阵都是一个 NumPy 数组
    :return: 最终的左乘结果矩阵
    """
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.dot(matrix, result)
    return result


def save_matrix_as_json(matrix, filename):
    """
    将矩阵保存为 JSON 文件
    :param matrix: 要保存的矩阵
    :param filename: 保存的文件名
    """
    with open(filename, "w") as f:
        json.dump(matrix.tolist(), f, indent=4)


def main():
    # 示例：定义几个矩阵
    A = np.array([[0.9999936784236516, 4.441341249212944e-07, 0.00355571547472536],
                  [4.441341249212944e-07, 0.9999999687965295, -0.0002498134158006251],
                  [-0.00355571547472536, 0.0002498134158006251, 0.9999936472201811]])

    B = np.array(
        [[0.9999841093680584, 5.76650823153765e-05, -0.005637170044390312],
         [5.76650823153765e-05, 0.9997907407502324, 0.020456573145074083],
         [0.005637170044390312, -0.020456573145074083, 0.9997748501182908]])

    # 将矩阵列表传递给左乘函数
    matrices = [B, A]  # 注意：左乘的顺序是从右到左

    # 检查每个矩阵是否为齐次变换矩阵，如果不是则转换
    homogeneous_matrices = []
    for matrix in matrices:
        if is_homogeneous(matrix):
            homogeneous_matrices.append(matrix)
        else:
            homogeneous_matrices.append(to_homogeneous(matrix))

    # 计算左乘结果
    result_matrix = left_multiply_matrices(homogeneous_matrices)

    # 保存结果矩阵为 JSON 文件
    save_matrix_as_json(result_matrix, "homogeneous_transformation_matrix.json")

    print("最终的齐次变换矩阵已保存到 'homogeneous_transformation_matrix.json'")


if __name__ == "__main__":
    main()
