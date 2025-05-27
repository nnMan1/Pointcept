import glob
import open3d as o3d
import os.path as osp
import numpy as np
from sklearn.cluster import DBSCAN


class AnomalyDetectionModel:

    def __init__(self, segmentation_model, cls_of_int):
        self.segmentation_model = segmentation_model
        self.cls_of_int = cls_of_int

        self.points = None
        self.labels = None
        self.instance = None
        self.__instance_id = 0

    def call(self, data_dict):
        
        pred = self.segmentation_model(data_dict)

        # semantic_labels = pred[]

    def load_predicted_part(self, path, prefix):
        pcds = o3d.geometry.PointCloud()

        files = sorted(glob.glob(osp.join(path, f'{prefix}*.ply')))
        preds = sorted(glob.glob(osp.join(path, f'{prefix}*.npy')))

        points = []
        labels = []

        for file, pred in zip(files, preds):
            points.append(np.asarray(o3d.io.read_point_cloud(file).points))
            labels.append(np.load(pred))

        self.points = np.concatenate(points)
        self.labels = np.concatenate(labels)

    def to_o3d(self, color='semantic'):

        color_to_arr = {
            'semantic': self.labels,
            'instance': self.instance
        }

        if self.points is None or self.labels is None:
            raise ValueError("Points and labels must be loaded before converting to Open3D format.")
        
        colors = np.random.uniform(0, 1, (512, 3))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.colors = o3d.utility.Vector3dVector(colors[color_to_arr[color]])


        return pcd
    
    def ransac_line_fit_3d(self, points, distance_threshold=0.1, max_iterations=1000):
        """
        Fits a single line to 3D points using RANSAC.
        
        Parameters
        ----------
        points : (N, 3) ndarray
            A set of 3D points.
        distance_threshold : float
            Maximum distance for a point to be considered an inlier.
        max_iterations : int
            Number of random samples to try.

        Returns
        -------
        p0 : ndarray of shape (3,)
            A point on the best-fit line.
        direction : ndarray of shape (3,) or None
            A normalized direction vector of the best-fit line.
            If None, it means we couldn't find any valid line (not enough points).
        best_inlier_mask : (N,) boolean ndarray
            Boolean array of which points are inliers for the found line.
        """

        n = len(points)
        if n < 2:
            return None, None, np.zeros(n, dtype=bool)

        best_p0 = None
        best_dir = None
        best_inlier_mask = np.zeros(n, dtype=bool)
        best_inlier_count = 0

        for _ in range(max_iterations):
            # 1) Randomly pick two distinct points
            idx = np.random.choice(n, 2, replace=False)
            p1 = points[idx[0]]
            p2 = points[idx[1]]

            # If the points are extremely close, skip to avoid numerical instability
            if np.allclose(p1, p2):
                continue

            # 2) Define line by p0 and direction vector d (un-normalized first)
            p0_candidate = p1
            d_candidate = p2 - p1
            norm_d_candidate = np.linalg.norm(d_candidate)
            if norm_d_candidate < 1e-12:
                continue
            d_candidate = d_candidate / norm_d_candidate  # normalize

            # 3) Compute distances of all points to this candidate line
            # Distance from point p to line (p0, d) is:
            #   || (p - p0) x d || / ||d||, but d is normalized => ||d||=1
            #   so distance = || (p - p0) x d ||
            vecs = points - p0_candidate
            # cross product shape = (N,3)
            cross_prod = np.cross(vecs, d_candidate)
            distances = np.linalg.norm(cross_prod, axis=1)

            # 4) Determine inliers
            inlier_mask = distances < distance_threshold
            inlier_count = np.sum(inlier_mask)

            # 5) Update best if we found more inliers
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_p0 = p0_candidate
                best_dir = d_candidate
                best_inlier_mask = inlier_mask

        return best_p0, best_dir, best_inlier_mask

    def cluster_label(self, label):
        mask = np.where(self.labels == label)[0]

        if label == 3:
            self.__instance_id += 1
            self.instance[mask] = self.__instance_id
            return

        all_points = self.points[mask]

        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=5, min_samples=3)
        instances = dbscan.fit_predict(all_points)
        instances_ids = np.unique(instances)

        print(f"DBSCAN found {len(instances_ids) - (1 if -1 in instances_ids else 0)} clusters of class {label}")

        for instance_id in instances_ids:
            if instance_id == -1:
                continue  # Skip noise points

            self.__instance_id += 1
            self.instance[mask[instances == instance_id]] = self.__instance_id

    def find_instances(self):
        self.instance = np.zeros_like(self.labels)
        self.__instance_id = 0

        for label_id in np.unique(self.labels):
            if label_id != -1:
                self.cluster_label(label_id)



    # def cluster_class_to(self):

    #     self.__instance_id = 0


    #     for label in labels:
            

        

    #     # Separate points into clusters
    #     clusters = []
    #     for cluster_id in range(num_clusters):
    #         cluster_points = all_points[labels == cluster_id]
    #         clusters.append(cluster_points)

    #     all_points = [np.asarray(cluster).mean(0) for cluster in clusters]
    #     return all_points

    def detect_multiple_lines_ransac_3d(self, distance_threshold=0.1, 
                                        max_iterations=1000, 
                                        min_inliers=20, 
                                        max_lines=5):
        """
        Iteratively apply RANSAC to find multiple lines in 3D.

        Parameters
        ----------
        points : (N, 3) ndarray
            Input 3D point cloud.
        distance_threshold : float
            Distance threshold for a point to be an inlier of a line.
        max_iterations : int
            Number of RANSAC iterations per line detection.
        min_inliers : int
            Minimum number of inliers required to "accept" a detected line.
        max_lines : int
            Maximum number of lines to detect.

        Returns
        -------
        lines : list of dict
            A list where each element is a dictionary with keys:
            {
            "p0"       : ndarray(3,),  # a point on the line
            "direction": ndarray(3,),  # the line's direction (unit vector)
            "inliers"  : ndarray(M,3), # the subset of points that fit this line
            }
        remaining_points : (K, 3) ndarray
            The points that did not fit any of the detected lines (outliers).
        """

        remaining_points = self.points[self.labels == self.cls_of_int][::20]
        centroids = self.cluster_rivets_to_centroids(remaining_points)
        remaining_points = np.asarray(centroids)
        print(centroids)

        lines = []

        for _ in range(max_lines):
            if len(remaining_points) < 2:
                break

            p0, direction, inlier_mask = self.ransac_line_fit_3d(
                remaining_points, distance_threshold, max_iterations
            )

            inlier_count = np.sum(inlier_mask)

            if p0 is None or direction is None:
                # Could not find a valid line
                break

            if inlier_count < min_inliers:
                # Not enough inliers to be considered a good line
                break
            
            # Store the line parameters
            inlier_points = remaining_points[inlier_mask]
            lines.append({
                "p0": p0,
                "direction": direction,
                "inliers": inlier_points
            })
            
            # Remove the inliers from the point set
            remaining_points = remaining_points[~inlier_mask]
            
            if len(remaining_points) < 2:
                break

        return centroids, lines, remaining_points



anomaly_detection = AnomalyDetectionModel(None, 4)
anomaly_detection.load_predicted_part('exp/fuselage/result', 'panel5_2')

o3d.visualization.draw_geometries([anomaly_detection.to_o3d()])

anomaly_detection.find_instances()

o3d.visualization.draw_geometries([anomaly_detection.to_o3d('instance')])



# centorids, lines_found, leftovers = anomaly_detection.detect_multiple_lines_ransac_3d( distance_threshold=1,  # adjust based on noise
#                                                                              max_iterations=1000,
#                                                                              min_inliers=2,
#                                                                              max_lines=100)

for i, line in enumerate(lines_found, start=1):
    p0 = line["p0"]
    d  = line["direction"]
    inliers = line["inliers"]
    print(f"Line {i}:")
    print(f"  p0       = {p0}")
    print(f"  direction= {d}")
    print(f"  inliers  = {len(inliers)} points")
print(f"\nRemaining (unfit) points: {len(leftovers)}")


# Plot all original points
# ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], marker='o', label='All Points')

# For each detected line, plot the inliers in a distinct marker,
# and draw a segment for the line.
# Create an Open3D point cloud for the original points
pcd_all = o3d.geometry.PointCloud()
pcd_all.points = o3d.utility.Vector3dVector(centorids)
pcd_all.paint_uniform_color([0.5, 0.5, 0.5])  # gray for all points

o3d.visualization.draw_geometries([pcd_all])

# Create point clouds for each line's inliers
sphere_meshes = []
for i, line in enumerate(lines_found, start=1):
    inliers = line["inliers"]
    color = np.random.rand(3)  # random color for each line
    for point in inliers:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3)
        sphere.translate(point)
        sphere.paint_uniform_color(color)
        sphere_meshes.append(sphere)

o3d.visualization.draw_geometries(sphere_meshes)
# line_pcds = []
# for i, line in enumerate(lines_found, start=1):
#     inliers = line["inliers"]
#     pcd_inliers = o3d.geometry.PointCloud()
#     pcd_inliers.points = o3d.utility.Vector3dVector(inliers)
#     color = np.random.rand(3)  # random color for each line
#     pcd_inliers.paint_uniform_color(color)
#     line_pcds.append(pcd_inliers)

# o3d.visualization.draw_geometries(line_pcds)
