import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --------------- 参数 ---------------
H = 5.5           # 安装高度 (m)
h_fov_deg = 60   # 水平 FOV
v_fov_deg = 6  # 垂直 FOV
phi_deg = 0    # 俯仰角：>0 表示向上，<0 向下（但通常雷达略下俯）
n_grid = 25       # 网格密度
MAX_RANGE = 300# 最大探测距离 (m)

# 转换为弧度
h_fov = np.radians(h_fov_deg)
v_fov = np.radians(v_fov_deg)
phi = np.radians(phi_deg)

# --------------- 新增：指定查询高度 ---------------
Z_target = 11.0  # 想要查询的 Z 高度 (m)

# --------------- 计算地面足迹（用于底面）---------------
α = np.linspace(-h_fov / 2, h_fov / 2, n_grid)
β = np.linspace(-v_fov / 2, v_fov / 2, n_grid)
α, β = np.meshgrid(α, β)

# 雷达坐标系方向向量
vx_r = np.sin(α) * np.cos(β)
vy_r = np.cos(α) * np.cos(β)
vz_r = np.sin(β)

# 绕 X 轴旋转 φ
cos_p, sin_p = np.cos(phi), np.sin(phi)
vx_w = vx_r
vy_w = vy_r * cos_p - vz_r * sin_p
vz_w = vy_r * sin_p + vz_r * cos_p

# 射线与地面交点：H + t * vz_w = 0 => t = -H / vz_w
t_ground = np.full_like(vz_w, np.nan)
mask_down = (vz_w < 0)  # 向下才能打到地面
t_ground[mask_down] = -H / vz_w[mask_down]
t_ground_clipped = np.clip(t_ground, None, MAX_RANGE)

X_ground = t_ground_clipped * vx_w
Y_ground = t_ground_clipped * vy_w
Z_ground = np.zeros_like(X_ground)

# 有效点范围
valid_mask = ~np.isnan(X_ground)
if not np.any(valid_mask):
    raise ValueError("No valid ground intersections. Check phi and FOV settings.")

x_min, x_max = X_ground[valid_mask].min(), X_ground[valid_mask].max()
y_min, y_max = Y_ground[valid_mask].min(), Y_ground[valid_mask].max()

padding_x = max(5, 0.1 * (x_max - x_min)) if x_max != x_min else 5
padding_y = max(5, 0.1 * (y_max - y_min)) if y_max != y_min else 5
padding_z = 200

x_min_adj, x_max_adj = x_min - padding_x, x_max + padding_x
y_min_adj, y_max_adj = max(0, y_min - padding_y), y_max + padding_y
z_min_adj, z_max_adj = -10, H + padding_z

# --------------- 计算 Z = Z_target 平面上的 FOV 截面 ---------------
# 我们固定 Z = Z_target，反推 t = (Z_target - H) / vz_w
# 然后 X = t * vx_w, Y = t * vy_w

t_slice = (Z_target - H) / vz_w  # 对每个方向计算到达 Z_target 所需距离
valid_at_z = (t_slice > 0) & (t_slice <= MAX_RANGE)  # 必须是正距离且不超 MAX_RANGE

X_slice = t_slice * vx_w
Y_slice = t_slice * vy_w
Z_slice = np.full_like(X_slice, Z_target)

# 提取有效点
X_valid = X_slice[valid_at_z]
Y_valid = Y_slice[valid_at_z]

if len(X_valid) == 0:
    print(f"⚠️  Warning: No FOV coverage at Z = {Z_target} m")
    x_min_z, x_max_z = 0, 0
    y_min_z, y_max_z = 0, 0
else:
    x_min_z, x_max_z = X_valid.min(), X_valid.max()
    y_min_z, y_max_z = Y_valid.min(), Y_valid.max()
    print(f"✅ 在高度 Z = {Z_target} m 处，FOV 覆盖范围：")
    print(f"   X ∈ [{x_min_z:.2f}, {x_max_z:.2f}] m")
    print(f"   Y ∈ [{y_min_z:.2f}, {y_max_z:.2f}] m")

# --------------- 绘图 ---------------
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 设置坐标轴范围
ax.set_xlim(x_min_adj, x_max_adj)
ax.set_ylim(y_min_adj, y_max_adj)
ax.set_zlim(z_min_adj, z_max_adj)

# === 绘制地面足迹 ===
ax.plot_surface(X_ground, Y_ground, Z_ground, color='skyblue', alpha=0.6,
                edgecolor='k', lw=0.2, shade=True, label='Ground Footprint')

# === 绘制 Z = Z_target 截面（线框）===
if len(X_valid) > 0:
    # 为了形成封闭轮廓，我们提取边界点（按 α 排序）
    # 沿着 β 的上下边界插值
    X_upper_z = []
    Y_upper_z = []
    X_lower_z = []
    Y_lower_z = []

    for i in range(n_grid):
        α_i = α[0, i]
        # 上边界 β = v_fov/2
        b_upper = v_fov / 2
        dx_r = np.sin(α_i) * np.cos(b_upper)
        dy_r = np.cos(α_i) * np.cos(b_upper)
        dz_r = np.sin(b_upper)
        dx_w = dx_r
        dy_w = dy_r * cos_p - dz_r * sin_p
        dz_w = dy_r * sin_p + dz_r * cos_p
        t = (Z_target - H) / dz_w
        if 0 < t <= MAX_RANGE:
            X_upper_z.append(t * dx_w)
            Y_upper_z.append(t * dy_w)

        # 下边界 β = -v_fov/2
        b_lower = -v_fov / 2
        dz_r_low = np.sin(b_lower)
        dz_w_low = dy_r * sin_p + dz_r_low * cos_p
        t_low = (Z_target - H) / dz_w_low
        if 0 < t_low <= MAX_RANGE:
            X_lower_z.append(t_low * dx_w)
            Y_lower_z.append(t_low * dy_w)

    # 绘制上边界线
    if X_upper_z:
        ax.plot(X_upper_z, Y_upper_z, Z_target, color='red', lw=2.5, label=f'Z={Z_target}m Upper')
    # 绘制下边界线
    if X_lower_z:
        ax.plot(X_lower_z, Y_lower_z, Z_target, color='orange', lw=2.5, label=f'Z={Z_target}m Lower')

    # 可选：连接左右端点（形成矩形框）
    if X_upper_z and X_lower_z:
        ax.plot([X_upper_z[0], X_lower_z[0]], [Y_upper_z[0], Y_lower_z[0]],
                [Z_target, Z_target], color='purple', lw=1.5)
        ax.plot([X_upper_z[-1], X_lower_z[-1]], [Y_upper_z[-1], Y_lower_z[-1]],
                [Z_target, Z_target], color='purple', lw=1.5)

# === 绘制边界射线（线框风格）===
corners = [
    (-h_fov / 2, -v_fov / 2),
    (-h_fov / 2,  v_fov / 2),
    ( h_fov / 2,  v_fov / 2),
    ( h_fov / 2, -v_fov / 2),
]

for a, b in corners:
    dx_r = np.sin(a) * np.cos(b)
    dy_r = np.cos(a) * np.cos(b)
    dz_r = np.sin(b)
    dx_w = dx_r
    dy_w = dy_r * cos_p - dz_r * sin_p
    dz_w = dy_r * sin_p + dz_r * cos_p

    # 射线终点：到地面或 MAX_RANGE
    t_end = MAX_RANGE
    if dz_w < 0:
        t_ground = -H / dz_w
        t_end = min(t_end, t_ground)

    t_line = np.linspace(0, t_end, 100)
    x_line = t_line * dx_w
    y_line = t_line * dy_w
    z_line = H + t_line * dz_w
    ax.plot(x_line, y_line, z_line, 'gray', lw=1.2, alpha=0.8)

# === 坐标轴 ===
L_axis = (y_max_adj - y_min_adj) * 0.7
ax.quiver(0, 0, H, 1, 0, 0, length=L_axis * 0.3, color='r', arrow_length_ratio=0.1)
ax.quiver(0, 0, H, 0, 1, 0, length=L_axis, color='g', arrow_length_ratio=0.1)
ax.quiver(0, 0, H, 0, 0, 1, length=L_axis * 0.6, color='b', arrow_length_ratio=0.1)
ax.text(L_axis * 0.3, 0, H, ' X', color='red', fontsize=10)
ax.text(0, L_axis, H, ' Y', color='green', fontsize=10)
ax.text(0, 0, z_max_adj - 5, ' Z', color='blue', fontsize=10)

# === 绘制旋转后的坐标轴 (Y') ===
# 绕X轴旋转φ后的Y轴方向
y_prime_dir = np.array([0, np.cos(phi), np.sin(phi)])
y_prime_length = L_axis
y_prime_start = np.array([0, 0, H])
y_prime_end = y_prime_start + y_prime_length * y_prime_dir

ax.quiver(y_prime_start[0], y_prime_start[1], y_prime_start[2],
          y_prime_dir[0], y_prime_dir[1], y_prime_dir[2],
          length=y_prime_length, color='magenta', arrow_length_ratio=0.1, label="Y' (rotated)")

ax.text(y_prime_end[0], y_prime_end[1], y_prime_end[2], " Y'", color='magenta', fontsize=10)

# === LiDAR 位置 ===
ax.scatter(0, 0, H, color='red', s=100, label='LiDAR')

# === 标题与设置 ===
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title(f'3D LiDAR FOV with Z={Z_target}m Slice\nφ = {phi_deg:.1f}°', fontsize=12)
ax.grid(True, alpha=0.5)
ax.legend()

# 长宽比
x_span = x_max_adj - x_min_adj
y_span = y_max_adj - y_min_adj
z_span = z_max_adj - z_min_adj
ax.set_box_aspect([x_span, y_span, z_span])
ax.view_init(elev=20, azim=-45)

plt.tight_layout()
plt.show()