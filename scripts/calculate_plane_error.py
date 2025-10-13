"""
计算两个平面的误差，判断两个平面是否相同
"""
import numpy as np

def calculate_plane_error(a1, b1, c1, d1, a2, b2, c2, d2):
    # 计算法向量
    n1 = np.array([a1, b1, c1])
    n2 = np.array([a2, b2, c2])

    # 计算法向量之间的夹角
    cos_theta = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    theta = np.arccos(cos_theta)

    # 计算平面之间的距离
    if np.allclose(n1, n2) or np.allclose(n1, -n2):
        d = np.abs(d2 - d1) / np.linalg.norm(n1)
    else:
        # 选择一个点来计算距离
        point = np.array([0, 0, -d1 / c1])
        d = np.abs(a2 * point[0] + b2 * point[1] + c2 * point[2] + d2) / np.linalg.norm(n2)

    # 计算重建误差
    alpha = 0.1
    beta = 1
    E = alpha * theta + beta * d

    return E, theta, d

def are_planes_similar(E, theta, d, angle_threshold=0.01745, distance_threshold=0.01):
    """
    判断两个平面是否基本相同
    :param E: 综合误差
    :param theta: 角度误差（弧度）
    :param d: 距离误差（米）
    :param angle_threshold: 角度误差阈值（弧度）
    :param distance_threshold: 距离误差阈值（米）
    :return: 是否基本相同
    """
    return theta < angle_threshold and d < distance_threshold


if __name__ == '__main__':
    try:
        a1 = float(input("请输入第一个平面的 a1: "))
        b1 = float(input("请输入第一个平面的 b1: "))
        c1 = float(input("请输入第一个平面的 c1: "))
        d1 = float(input("请输入第一个平面的 d1: "))
    except ValueError:
        print("输入无效，请输入数字！")
    while True:
        try:
            a2 = float(input("请输入第二个平面的 a2: "))
            b2 = float(input("请输入第二个平面的 b2: "))
            c2 = float(input("请输入第二个平面的 c2: "))
            d2 = float(input("请输入第二个平面的 d2: "))
        except ValueError:
            print("输入无效，请输入数字！")
            continue

        E, theta, d = calculate_plane_error(a1, b1, c1, d1, a2, b2, c2, d2)
        print(f"综合误差: {E:.6f} 米")
        print(f"角度误差: {np.degrees(theta):.6f} 度")
        print(f"距离误差: {d:.6f} 米")

        if are_planes_similar(E, theta, d):
            print("两个平面基本相同")
        else:
            print("两个平面不相同")