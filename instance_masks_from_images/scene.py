import numpy as np
import open3d as o3d


class Camera:
    def __init__(self, 
                 intrinsic_matrix, 
                 width,
                 height):
        self.intrinsic_matrix = intrinsic_matrix
        self.width = width
        self.height = height
    
    @property
    def resolution(self):
        return self.width, self.height


class Scene:
    def __init__(self, cfg) -> None:
        self.point_cloud = o3d.io.read_point_cloud(cfg.point_cloud_path)
        self.mesh = None
        if cfg.mesh_path:
            self.mesh = self.load_mesh(cfg.mesh_path)
        self.raycasting_scene = None
        self.visibility_threshold = cfg.visibility_threshold
    
    def get_point_cloud_homogeneous_coordinates(self):
        points = np.asarray(self.point_cloud.points)
        num_points = points.shape[0]
        return np.append(points, np.ones((num_points, 1)), axis = -1)

    @staticmethod
    def load_mesh(mesh_path):
        # return o3d.io.read_triangle_mesh(mesh_path)
        return o3d.io.read_triangle_model(mesh_path)

    def draw_mesh(self):
        submeshes = [submesh_info.mesh for submesh_info in self.mesh.meshes]
        o3d.visualization.draw_geometries(submeshes, mesh_show_back_face=True)

    def init_raycasting_scene(self):
        self.raycasting_scene = o3d.t.geometry.RaycastingScene()
        submeshes = [submesh_info.mesh for submesh_info in self.mesh.meshes]
        for submesh in submeshes:
            t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(submesh)
            mesh_id = self.raycasting_scene.add_triangles(t_mesh)

    def is_visible(self, img_position, points):
        if self.raycasting_scene is None:
            self.init_raycasting_scene()
        n_points = points.shape[0]
        directions = points - img_position
        rays = np.concatenate([img_position.reshape((1,3)).repeat(n_points, axis=0), directions], axis=1)
        ans = self.raycasting_scene.cast_rays(o3d.core.Tensor([rays], dtype=o3d.core.Dtype.Float32))
        return ans["t_hit"][0] >= self.visibility_threshold

    def get_visible_points(self, camera, img_extrinsic):
        n_points = len(self.point_cloud.points)
        X = self.get_point_cloud_homogeneous_coordinates()
        intrinsic = np.concatenate((camera.intrinsic_matrix, [[0],[0],[0]]), axis=1)

        projected_points = np.zeros((n_points, 2), dtype = int)
        
        # STEP 1: get the projected points
        # Get the coordinates of the projected points in the i-th view (i.e. the view with index idx)
        projected_points_not_norm = (intrinsic @ img_extrinsic @ X.T).T
        
        # Get the mask of the points which have a non-null third coordinate to avoid division by zero
        mask = (projected_points_not_norm[:, 2] != 0) # don't do the division for point with the third coord equal to zero
        # Get non homogeneous coordinates of valid points (2D in the image)
        projected_points[mask] = np.column_stack([[projected_points_not_norm[:, 0][mask] /  projected_points_not_norm[:, 2][mask], 
                projected_points_not_norm[:, 1][mask]/projected_points_not_norm[:, 2][mask]]]).T

        inside_mask = (projected_points[:,0] >= 0) * (projected_points[:,1] >= 0) \
                        * (projected_points[:,0] < camera.width) \
                        * (projected_points[:,1] < camera.height)
    
        # STEP 2: occlusions computation
        inside_points = np.array(self.point_cloud.points)[inside_mask]
        img_position = np.linalg.inv(img_extrinsic)[:3,3:].reshape((3,))
        visibility_mask_inside = self.is_visible(img_position, inside_points)
        visibility_mask = inside_mask.copy()
        visibility_mask[inside_mask] = visibility_mask_inside

        return inside_mask, visibility_mask, projected_points
