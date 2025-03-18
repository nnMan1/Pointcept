import numpy as np
import matplotlib.pyplot as plt

def ransac_line_fit_3d(points, distance_threshold=0.1, max_iterations=1000):
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


def detect_multiple_lines_ransac_3d(points, 
                                    distance_threshold=0.1, 
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
    remaining_points = np.copy(points)
    lines = []

    for _ in range(max_lines):
        if len(remaining_points) < 2:
            break

        p0, direction, inlier_mask = ransac_line_fit_3d(
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

    return lines, remaining_points


# ---------------- USAGE EXAMPLE ----------------
if __name__ == "__main__":
    np.random.seed(0)

    
    import open3d as o3d
    from sklearn.cluster import KMeans
    from sklearn.cluster import DBSCAN

    pcd = o3d.io.read_point_cloud('data/panel5/1-complete_assembly_1mm/rivet/5.ply')
    all_points = np.asarray(pcd.points)

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=4, min_samples=10)
    labels = dbscan.fit_predict(all_points)

    # Print the number of clusters found
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"DBSCAN found {num_clusters} clusters")
    

    # Separate points into clusters
    clusters = []
    for cluster_id in range(num_clusters):
        cluster_points = all_points[labels == cluster_id]
        clusters.append(cluster_points)

    print(clusters)

    all_points = [np.asarray(cluster).mean(0) for cluster in clusters]
    print(all_points)

    # # Print info about each cluster
    # for i, cluster in enumerate(clusters, start=1):
    #     print(f"Cluster {i}: {len(cluster)} points")

    
    
        
    # RANSAC to detect multiple lines
    lines_found, leftovers = detect_multiple_lines_ransac_3d(
        all_points,
        distance_threshold=3,  # adjust based on noise
        max_iterations=1000,
        min_inliers=3,
        max_lines=100
    )

    # Print info
    for i, line in enumerate(lines_found, start=1):
        p0 = line["p0"]
        d  = line["direction"]
        inliers = line["inliers"]
        print(f"Line {i}:")
        print(f"  p0       = {p0}")
        print(f"  direction= {d}")
        print(f"  inliers  = {len(inliers)} points")
    print(f"\nRemaining (unfit) points: {len(leftovers)}")

    # OPTIONAL: visualize in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all original points
    # ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], marker='o', label='All Points')

    # For each detected line, plot the inliers in a distinct marker,
    # and draw a segment for the line.
    # Create an Open3D point cloud for the original points
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(all_points)
    pcd_all.paint_uniform_color([0.5, 0.5, 0.5])  # gray for all points

    # Create point clouds for each line's inliers
    line_pcds = []
    for i, line in enumerate(lines_found, start=1):
        inliers = line["inliers"]
        pcd_inliers = o3d.geometry.PointCloud()
        pcd_inliers.points = o3d.utility.Vector3dVector(inliers)
        color = np.random.rand(3)  # random color for each line
        pcd_inliers.paint_uniform_color(color)
        line_pcds.append(pcd_inliers)

    o3d.visualization.draw_geometries(line_pcds)

    # # Create a point cloud for the leftover points
    # if len(leftovers) > 0:
    #     pcd_leftovers = o3d.geometry.PointCloud()
    #     pcd_leftovers.points = o3d.utility.Vector3dVector(leftovers)
    #     pcd_leftovers.paint_uniform_color([1, 0, 0])  # red for leftovers

    # # Visualize all point clouds together
    # o3d.visualization.draw_geometries([pcd_all] + line_pcds + ([pcd_leftovers] if len(leftovers) > 0 else []))
    # for i, line in enumerate(lines_found, start=1):
    #     inliers = line["inliers"]
    #     ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], marker='^',
    #                label=f'Line {i} Inliers')

    #     # Plot a small segment representing the line
    #     p0 = line["p0"]
    #     d  = line["direction"]
    #     # pick a range for t just for plotting
    #     t_vals = np.linspace(-1, 6, 2)
    #     line_pts = np.array([p0 + t*d for t in t_vals])
    #     ax.plot(line_pts[:,0], line_pts[:,1], line_pts[:,2], linestyle='--')

    # Plot leftover points
    # if len(leftovers) > 0:
    #     ax.scatter(leftovers[:, 0], leftovers[:, 1], leftovers[:, 2],
    #                marker='x', label='Unfit Points')
        
        
    # ax.legend()
    # ax.set_title("Multiple 3D Lines via Iterative RANSAC")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # plt.show()
