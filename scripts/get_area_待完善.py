import open3d as o3d
import numpy as np
import open3d.visualization.gui as gui


def create_sample_point_cloud():
    """生成测试点云"""
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    xx, yy = np.meshgrid(x, y)
    zz = np.random.rand(*xx.shape) * 0.5
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


class PolygonCropper:
    def __init__(self, pcd):
        # ⚠️ 此处不再立即创建窗口！延迟到 setup() 中
        self.pcd = pcd
        self.window = None
        self.widget3d = None
        self.scene = None
        self.line_set = None
        self.polygon_points = []
        self.is_drawing = True

        # 延迟初始化 GUI（在 main 中确保 app 初始化后再调用 setup）
        self.setup()

    def setup(self):
        app = o3d.visualization.gui.Application.instance
        if app is None:
            raise RuntimeError("Open3D Application instance 未初始化！")

        # 创建窗口
        self.window = app.create_window("Interactive Polygon Crop", 1200, 800)
        self.widget3d = o3d.visualization.gui.SceneWidget()
        self.window.add_child(self.widget3d)

        # 创建场景
        self.scene = o3d.visualization.rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene = self.scene

        # 添加点云
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.point_size = 3
        mat.shader = "defaultUnlit"
        self.scene.add_geometry("pcd", self.pcd, mat)

        # 设置相机
        bbox = self.pcd.get_axis_aligned_bounding_box()
        center = self.pcd.get_center()
        self.widget3d.setup_camera(60.0, bbox, center)

        # 注册事件
        self.widget3d.set_on_mouse(self.on_mouse)
        self.window.set_on_key(self.on_key)

        print("✅ 点云已加载。左键添加顶点，按 C 完成裁剪，R 重置。")

    def on_mouse(self, event):
        if event.type != o3d.visualization.gui.MouseEvent.Type.BUTTON_DOWN:
            print("错误选点操作1")
            return False

        # 1. 不再判断 Ctrl，直接看是不是右键
        if event.buttons != o3d.visualization.gui.MouseButton.RIGHT:  # 这里改成 RIGHT
            print("错误选点操作2", event.buttons, o3d.visualization.gui.MouseButton.RIGHT)
            return False

        x, y = int(event.x), int(event.y)
        depth = self.widget3d.pick_depth(x, y)
        if np.isfinite(depth):
            world = self.widget3d.scene.camera.unproject(x, y, depth)
            self.polygon_points.append([world[0], world[1]])
            self.update_polygon()
            print("正确选点操作")
        return True

    def on_key(self, event):
        if event.type != o3d.visualization.gui.KeyEvent.DOWN:
            return False     # ← 原来是 EventCallbackResult.IGNORED

        if event.key == ord('C') or event.key == ord('c'):
            self.finish_crop()
            return True      # ← 原来是 EventCallbackResult.HANDLED
        elif event.key == ord('R') or event.key == ord('r'):
            self.reset_polygon()
            return True

        return False

    def update_polygon(self):
        if len(self.polygon_points) < 2:
            return

        z_mean = np.mean(np.asarray(self.pcd.points)[:, 2])
        pts_3d = [[p[0], p[1], z_mean] for p in self.polygon_points]
        lines = [[i, (i + 1) % len(self.polygon_points)] for i in range(len(self.polygon_points))]

        if self.line_set is None:
            self.line_set = o3d.geometry.LineSet()
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "unlitLine"
            mat.line_width = 3
            mat.color = [1, 0, 0]
            self.scene.add_geometry("polygon", self.line_set, mat)

        self.line_set.points = o3d.utility.Vector3dVector(pts_3d)
        self.line_set.lines = o3d.utility.Vector2iVector(lines)
        self.line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))
        self.widget3d.force_redraw()

    def finish_crop(self):
        if not self.is_drawing or len(self.polygon_points) < 3:
            print("至少需要3个点！")
            return

        from matplotlib.path import Path
        path = Path(self.polygon_points)
        xy = np.asarray(self.pcd.points)[:, :2]
        mask = path.contains_points(xy)

        inside = self.pcd.select_by_index(np.where(mask)[0])
        outside = self.pcd.select_by_index(np.where(~mask)[0])

        print(f"内部点: {len(inside.points)}, 外部点: {len(outside.points)}")

        self.scene.clear_geometry()
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.point_size = 3
        mat.shader = "defaultUnlit"

        if len(inside.points) > 0:
            inside.paint_uniform_color([1, 0, 0])
            self.scene.add_geometry("inside", inside, mat)
        if len(outside.points) > 0:
            outside.paint_uniform_color([0, 0, 1])
            self.scene.add_geometry("outside", outside, mat)

        self.is_drawing = False

    def reset_polygon(self):
        self.polygon_points = []
        if self.line_set:
            self.scene.remove_geometry("polygon")
            self.line_set = None
        self.widget3d.force_redraw()


# ==================== 主函数 ====================
def main():
    app = gui.Application.instance
    app.initialize()

    # === 现在才安全地创建对象 ===
    pcd = create_sample_point_cloud()
    cropper = PolygonCropper(pcd)

    # === 运行主循环 ===
    o3d.visualization.gui.Application.instance.run()


if __name__ == "__main__":
    main()