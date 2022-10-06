import argparse
import numpy as np
import open3d as o3d
import copy
import time
import logging
import os
import subprocess
import glob
import pathlib
import sys
import math
import random
import json
import scipy.interpolate as si

from functools import partial
from multiprocessing import Pool


logger = logging.getLogger(__name__)

try:
    from scipy.spatial import cKDTree as KDTree
except ImportError:
    from scipy.spatial import KDTree

try:
    from scipy.spatial import cKDTree as KDTree
except ImportError:
    from scipy.spatial import KDTree


class Spline:
    def __init__(self, controlPts, degree=3, setDefaultResl=False):
        self.ctrlVecs = controlPts
        self.setDefaultResl = setDefaultResl
        self.degree = degree
        self.SplineSeg = None
        self.SplinePts = None
    
    def genSpline(self, resolution=40):
        #Calculate n samples on a bspline
        #cv :      Array of control vertices
        #n  :      Number of samples to return
        #degree:   Curve degree
        #
        cv = np.asarray(self.ctrlVecs)
        count = cv.shape[0]

        # Prevent degree from exceeding count-1, otherwise splev will crash
        degree = np.clip(self.degree, 1, count-1)
        
        # Calculate knot vector
        kv = np.concatenate([np.repeat([0], degree), np.arange(count-degree+1), np.repeat([count-degree], degree)])
        
        # Calculate query range
        u = np.linspace(0, (count - self.degree), resolution)
        
        pts = np.array(si.splev(u, (kv, cv.T, self.degree))).T
        
        indices = [[idx, idx+1] for idx in range(0, len(pts)-1)]
        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(pts)
        lines.lines = o3d.utility.Vector2iVector(indices)
        
        self.SplineSeg = lines
        
        # Calculate result
        return pts, self.SplineSeg
    
    def showSpline(self, withSampledPoints=False):
        if self.SplineSeg == None:
            pts, self.SplineSeg = self.genSpline()
        
            if withSampledPoints == True:
                self.SplinePts = o3d.geometry.PointCloud()
                self.SplinePts.points = o3d.utility.Vector3dVector(pts)
                self.SplinePts.paint_uniform_color([1.0, 0.0, 0.0])
                o3d.visualization.draw_geometries([self.SplineSeg, self.SplinePts])
            else:
                o3d.visualization.draw_geometries([self.SplineSeg])
        
class Line:
    def __init__(self, StartPt, EndPt, setDefaultResl=False):
        self.StartPt = StartPt
        self.EndPt = EndPt
        self.setDefaultResl = setDefaultResl
        self.LineSeg = None
        self.LinePts = None
    
    def genLine(self, resolution=40):
        #parametric equations of the line segment defined by its endpoints.
        pts = []
        for t in np.arange(0., 1. + 1./resolution, 1./resolution):
            pts.append([((1.0 - t) * self.StartPt[0]) + (t * self.EndPt[0]), 
                       ((1.0 - t) * self.StartPt[1]) + (t * self.EndPt[1]), 
                       ((1.0 - t) * self.StartPt[2]) + (t * self.EndPt[2])])
        
        indices = [[idx, idx+1] for idx in range(0, len(pts)-1)]
        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(pts)
        lines.lines = o3d.utility.Vector2iVector(indices)
        self.LineSeg = lines
        
        return pts, self.LineSeg
        
    def showLine(self, withSampledPoints=False):
        if self.LineSeg == None:
            pts, self.LineSeg = self.genLine()
        
            if withSampledPoints == True:
                self.LinePts = o3d.geometry.PointCloud()
                self.LinePts.points = o3d.utility.Vector3dVector(pts)
                self.LinePts.paint_uniform_color([0.0, 0.0, 1.0])
                o3d.visualization.draw_geometries([self.LineSeg, self.LinePts])
            else:
                o3d.visualization.draw_geometries([self.LineSeg])
        
class Arc:
    def __init__(self, StartPt, EndPt, CenterPt, radiusLength, normal, orientation="CCW", setDefaultResl=False):
        self.StartPt = StartPt
        self.EndPt  = EndPt
        self.CenterPt = CenterPt
        self.r = radiusLength 
        self.setDefaultResl = setDefaultResl
        self.orientation = orientation
        self.normal = normal
        self.ArcSeg = None
        self.ArcPts = None
        
    def gen3DArc(self, resolution=40):
        pts = []
        if self.orientation == "CCW":
            U = (np.array(self.StartPt) - np.array(self.CenterPt))
            V = (np.array(self.EndPt) - np.array(self.CenterPt))
            uniU =  U / np.linalg.norm(U)
            uniV =  V / np.linalg.norm(V)
            
            angularDev=np.arccos(np.clip(np.dot(uniU, uniV), -1.0, 1.0))
            if np.linalg.norm(np.array(self.StartPt) - np.array(self.EndPt)) < 0.001 or angularDev == 0.0:
                angularDev = 2 * math.pi
            
            # NOTE --> https://www.physicsforums.com/threads/general-equation-of-a-circle-in-3d.123168/
            for t in np.arange(0, angularDev + angularDev/ resolution, angularDev/resolution):
                newPt = self.CenterPt + (self.r * math.cos(t) * uniU) + (self.r * math.sin(t) * np.cross(self.normal, uniU)) 
                pts.append([newPt[0], newPt[1], newPt[2]])
            
        indices = [[idx, idx+1] for idx in range(0, len(pts)-1)]
        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(pts)
        lines.lines = o3d.utility.Vector2iVector(indices)
        self.ArcSeg = lines
        
        return pts, self.ArcSeg
    
    def show3DArc(self, withSampledPoints=False):
        if self.ArcSeg == None:
            pts, self.ArcSeg = self.gen3DArc()
        
            if withSampledPoints == True:
                self.LinePts = o3d.geometry.PointCloud()
                self.LinePts.points = o3d.utility.Vector3dVector(pts)
                self.LinePts.paint_uniform_color([0.0, 1.0, 0.0])
                o3d.visualization.draw_geometries([self.LineSeg, self.LinePts])
            else:
                o3d.visualization.draw_geometries([self.LineSeg])


def detect_edge_lines_one(inpath, outpath, nag_rad_threshold, debug_view, fpath):
    def map_indices(indices_len, removed_indices):
        """Map old indices to new ones after removal of some indices.

        Args:
            indices_len (int): Length of old indices
            removed_indices (list or array): Removed indices
        Returns:
            array: Mapping old indices to new ones
        Example:
            >>> self.map_indices(6, [1, 2])
            array([0, -1, -1, 1, 2, 3])

        """
        new_indices = np.arange(indices_len)
        new_indices_cur = 0
        removed_indices = np.sort(removed_indices)
        removed_indices_len = len(removed_indices)
        removed_indices_cur = 0
        for old_indices_cur in range(indices_len):
            if (removed_indices_cur < removed_indices_len and
                    old_indices_cur == removed_indices[removed_indices_cur]):
                new_indices[old_indices_cur] = -1
                removed_indices_cur += 1
            else:
                new_indices[old_indices_cur] = new_indices_cur
                new_indices_cur += 1
        return new_indices
    
    def map_triangles_to_edges(mesh):
        """calculate a map of triangles for each edge in mesh.
            an edge is defined by indices of vertices of its ends
        """
        tris_per_edge = {}
        ordered_edge = lambda v1, v2: (v1, v2) if v1 < v2 else (v2, v1)
        tri_index_pairs = [(0, 1), (1, 2), (2, 0)]
        for tidx, tri in enumerate(mesh.triangles):
            for (i1, i2) in tri_index_pairs:
                p1 = tri[i1]
                p2 = tri[i2]
                tris_per_edge.setdefault(ordered_edge(p1, p2), []).append(tidx)

        return tris_per_edge
    
    def remove_unreferenced_points(lineset: o3d.geometry.LineSet, indices):
        """remove points and respectivec lines from lineset based on point indices"""
        if len(lineset.points) == len(indices):
            return None

        points = np.delete(lineset.points, indices, axis=0)
        lines = np.asarray(lineset.lines)
        if lineset.lines is not None:
            line_indices = np.where(
                np.any(np.isin(lines[:], indices, assume_unique=False), axis=1))[0]
            cpy_lines = np.delete(lines, line_indices, axis=0)

            if cpy_lines.size == 0:
                return None

            new_indices = map_indices(len(lineset.points), indices)
            cpy_lines = np.vectorize(lambda x: new_indices[x])(cpy_lines)
            if lineset.colors:
                cpy_colors = np.delete(np.asarray(
                    lineset.colors), line_indices, axis=0)
            lines = cpy_lines

        return o3d.geometry.LineSet(o3d.utility.Vector3dVector(points), o3d.utility.Vector2iVector(lines))
    
    """detect sharp edges based on CAD models face normals difference"""
    opath = os.path.splitext(fpath.replace(inpath, outpath))[0] + '.ply'
    if os.path.exists(opath):
        logger.info(f"already processed {opath}")
        return

    logger.info(f"loading {fpath}")
    inmesh = o3d.io.read_triangle_mesh(fpath)
    if not inmesh.has_vertex_normals():
        inmesh.compute_vertex_normals()

    # force split of all initial triangles into 4x to break the connectivity
    # of CAD borders
    mesh = inmesh.subdivide_midpoint(number_of_iterations=1)

    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()

    # if not mesh.is_edge_manifold:
    #     raise RuntimeError(inpath, 'not edge manifold')

    # calculate angle between normals pairwise
    normals_rad = lambda x, y: np.abs(
        np.clip(np.dot((x / np.linalg.norm(x)), (y / np.linalg.norm(y))), -1.0, 1.0))

    # generate all pairs from list items
    all_pairs = lambda lst: [(a, b) for idx, a in enumerate(lst)
                             for b in lst[idx + 1:]]

    tris_per_edge = map_triangles_to_edges(mesh)

    allow_boundary_edges = True
    sharp_edges = []
    sharp_norms = []  # TODO: if we need it further, should find a good way to store them

    nv = len(mesh.vertices)
    points_lbl = np.zeros(nv)

    tri_norms = np.asarray(mesh.triangle_normals)
    for (edge, tridices) in tris_per_edge.items():
        # if len(tridices) > 2:
        #     raise RuntimeError('non edge manifold')
        p1 = edge[0]
        p2 = edge[1]
        if len(tridices) == 1 and allow_boundary_edges:
            sharp_edges.append([p1, p2])
            sharp_norms.append(tri_norms[tridices[0]])
            points_lbl[p1] = 1
            points_lbl[p2] = 1

        # choose edge candidates based on angle between triangle normals
        dists = [normals_rad(tri_norms[t1], tri_norms[t2])
                 for (t1, t2) in all_pairs(tridices)]

        edge_candidates = np.argwhere(np.asarray(dists) < nag_rad_threshold)
        if edge_candidates.size > 0:
            sharp_edges.append([p1, p2])
            points_lbl[p1] = 1
            points_lbl[p2] = 1

            adj_tri_norms = [tri_norms[t] for t in tridices]
            edge_norm = sum(adj_tri_norms) / len(adj_tri_norms)
            normalized_edge_norm = edge_norm / np.linalg.norm(edge_norm)
            sharp_norms.append(normalized_edge_norm)

    if len(sharp_edges):
        edgesSet = o3d.geometry.LineSet(
            mesh.vertices, o3d.utility.Vector2iVector(sharp_edges))
        logger.info(f"{edgesSet} {len(edgesSet.points)} {len(edgesSet.lines)} {edgesSet.has_colors()}")

        rm_indices = np.argwhere(points_lbl == 0)

        logger.info(f"{len(rm_indices)} to clean up")
        edgesSet = remove_unreferenced_points(edgesSet, rm_indices)
        logger.info(f"{edgesSet} {len(edgesSet.points)} {len(edgesSet.lines)} {edgesSet.has_colors()}")

    if debug_view:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(inmesh)

        bbox = inmesh.get_axis_aligned_bounding_box()
        xtranslation = bbox.get_extent()[0]

        if not edgesSet.is_empty():
            #edgesSet.paint_uniform_color(np.array([1.0, 0.0, 0.0]))
            cpyEdgesSet = copy.copy(edgesSet)
            cpyEdgesSet.colors = o3d.utility.Vector3dVector(
                np.asarray(sharp_norms))
            vis.add_geometry(cpyEdgesSet.translate(
                np.array([1.1 * xtranslation, 0, 0])))

        vis.run()
        vis.destroy_window()

    if len(sharp_edges) and not edgesSet.is_empty():
        os.makedirs(os.path.dirname(opath), exist_ok=True)
        o3d.io.write_line_set(opath, edgesSet)
        logger.info(f"{edgesSet} {len(edgesSet.points)} {len(edgesSet.lines)} {edgesSet.has_colors()}")
    else:
        logger.warning(f"no sharp edges found {fpath}")
        os.makedirs(os.path.dirname(opath), exist_ok=True)
        with open(opath, 'w') as df:
            df.write("format ascii 1.0\n")
            df.write("comment VCGLIB generated\n")
            df.write("element vertex 0\n")
            df.write("property float x\n")
            df.write("property float y\n")
            df.write("property float z\n")
            df.write("element face 0\n")
            df.write("property list uchar int vertex_indices\n")
            df.write("end_header\n")
        logger.info(f"empty Edge Set")

def detect_edges(args):
    """detect sharp edges"""
    files = glob.glob(str(args.input_meshes) + '/**/*{}'.format(args.infmt_mesh), recursive=True)
    logger.info(f" found meshes {len(files)}")

    start = time.time()

    if args.nproc > 1:
        per_mesh_detect = partial(
            detect_edge_lines_one, str(args.input_meshes), str(args.output), args.normals_rad_threshold, False)
        pool = Pool(processes=args.nproc)
        pool.map(per_mesh_detect, files)
    else:
        for f in files:
            detect_edge_lines_one(str(args.input_meshes), str(
                args.output), args.normals_rad_threshold, False, f)

    logger.info(f"time elapsed: {time.time() - start}")

def genPointsfromJsonFile(AnnotFile, outFilePath, setDefaultResl=False, debug_view=False):
    def get_nested(data, *args):
        if args and data:
            element = args[0]
            if element:
                value = data.get(element)
                return value if len(args) == 1 else get_nested(value, *args[1:])
            
    def genParametricCurves(AnnotFile, outFilePath, setDefaultResl, debug_view):
        f = open(AnnotFile)
        json_dict = json.loads(f.read())
        DefaultSamplResl = 40
        
        countLineSeg = 0
        countArch = 0
        countSplineSeg = 0
        
        SplineSets = []
        LineSets = []
        ArcSets = []

        if "segments" in json_dict:
            for s in json_dict["segments"]:
                if s["stype"] == "LineSegment":
                    countLineSeg = countLineSeg + 1
                    x = 0.0
                    y = 0.0
                    z = 0.0
                    if "x" in s["line"]["start"]:
                        x = get_nested(s , "line", "start", "x")
                    if "y" in s["line"]["start"]:
                        y = get_nested(s , "line", "start", "y")
                    if "z" in s["line"]["start"]:
                        z = get_nested(s , "line", "start", "z")
                    startPt = [x, y, z]
                    
                    x = 0.0
                    y = 0.0
                    z = 0.0
                    if "x" in s["line"]["end"]:
                        x = get_nested(s , "line", "end", "x")
                    if "y" in s["line"]["end"]:
                        y = get_nested(s , "line", "end", "y")
                    if "z" in s["line"]["end"]:
                        z = get_nested(s , "line", "end", "z")
                    endPt = [x, y, z]
                    
                    if "occt" in s["line"]:
                        x = 0.0
                        y = 0.0
                        z = 0.0
                        if "x" in s["line"]["occt"]["direction"]:
                            x = get_nested(s , "line", "occt", "direction", "x")
                        if "y" in s["line"]["occt"]:
                            y = get_nested(s , "line", "occt", "direction", "y")
                        if "z" in s["line"]["occt"]:
                            z = get_nested(s , "line", "occt", "direction", "z")
                            
                        dirVec = [x,y,z]
                        
                        x = 0.0
                        y = 0.0
                        z = 0.0
                        if "x" in s["line"]["occt"]["direction"]:
                            x = get_nested(s , "line", "occt", "location", "x")
                        if "y" in s["line"]["occt"]:
                            y = get_nested(s , "line", "occt", "location", "y")
                        if "z" in s["line"]["occt"]:
                            z = get_nested(s , "line", "occt", "location", "z")
                            
                        location = [x,y,z]
                        lastParameter = get_nested(s, "line", "occt", "lastParameter")
                        firstParameter = get_nested(s, "line", "occt", "firstParameter")
                    
                    keypoints = []
                    if "keypoints" in s:
                        for k in get_nested(s, "keypoints"):
                            x = 0.0
                            y = 0.0
                            z = 0.0
                            if "x" in k:
                                x = get_nested(k, "x")
                            if "y" in k:
                                y = get_nested(k, "y")
                            if "z" in k:
                                z = get_nested(k, "z")
                            keypoints.append([x,y,z])
                        
                    distance = get_nested(s, "distance")
                    LineSets.append(Line(startPt, endPt, setDefaultResl))
                
                elif s["stype"] == "CircleSegment":
                    countArch = countArch + 1
                    
                    x = 0.0
                    y = 0.0
                    z = 0.0
                    if "x" in s["circle"]["center"]:
                        x = get_nested(s , "circle", "center", "x")
                    if "y" in s["circle"]["center"]:
                        y = get_nested(s , "circle", "center", "y")
                    if "z" in s["circle"]["center"]:
                        z = get_nested(s , "circle", "center", "z")
                    center = [x,y,z]
                        
                    x = 0.0
                    y = 0.0
                    z = 0.0
                    if "x" in s["circle"]["start"]:
                        x = get_nested(s , "circle", "start", "x")
                    if "y" in s["circle"]["start"]:
                        y = get_nested(s , "circle", "start", "y")
                    if "z" in s["circle"]["start"]:
                        z = get_nested(s , "circle", "start", "z")
                    startPt = [x, y, z]
                    
                    x = 0.0
                    y = 0.0
                    z = 0.0
                    if "x" in s["circle"]["end"]:
                        x = get_nested(s , "circle", "end", "x")
                    if "y" in s["circle"]["end"]:
                        y = get_nested(s , "circle", "end", "y")
                    if "z" in s["circle"]["end"]:
                        z = get_nested(s , "circle", "end", "z")
                    endPt = [x,y,z]

                    
                    x = 0.0
                    y = 0.0
                    z = 0.0
                    if "x" in s["circle"]["normal"]:
                        x = get_nested(s , "circle", "normal", "x")
                    if "y" in s["circle"]["normal"]:
                        y = get_nested(s , "circle", "normal", "y")
                    if "z" in s["circle"]["normal"]:
                        z = get_nested(s , "circle", "normal", "z")
                    normal = [x,y,z]
                    
                    radius = get_nested(s , "circle", "radius") 
                    
                    rotation = get_nested(s , "circle", "rotation") 
                    
                    occt_location = [get_nested(s, "circle", "occt", "location", "x"),   
                            get_nested(s, "circle", "occt", "location", "y"),
                            get_nested(s, "circle", "occt", "location", "z")]
                    
                    occt_normal = [get_nested(s, "circle", "occt", "normal", "x"),   
                                get_nested(s, "circle", "occt", "normal", "y"),
                                get_nested(s, "circle", "occt", "normal", "z")]
                    
                    occt_xAxis = [get_nested(s, "circle", "occt", "xAxis", "x"),   
                                get_nested(s, "circle", "occt", "xAxis", "y"),
                                get_nested(s, "circle", "occt", "xAxis", "z")]
                    
                    occt_yAxis = [get_nested(s, "circle", "occt", "yAxis", "x"),   
                                get_nested(s, "circle", "occt", "yAxis", "y"),
                                get_nested(s, "circle", "occt", "yAxis", "z")]
                    
                    occt_radius = get_nested(s, "circle", "occt", "radius")
                    occt_lastParameter = get_nested(s, "circle", "occt", "lastParameter")
                    occt_firstParameter = get_nested(s, "circle", "occt", "firstParameter")
                    
                    keypoints = []
                    if "keypoints" in s:
                        for k in get_nested(s, "keypoints"):
                            x = 0.0
                            y = 0.0
                            z = 0.0
                            if "x" in k:
                                x = get_nested(k, "x")
                            if "y" in k:
                                y = get_nested(k, "y")
                            if "z" in k:
                                z = get_nested(k, "z")
                            keypoints.append([x,y,z])
                        
                    distance = get_nested(s, "distance")
                    ArcSets.append(Arc(startPt, endPt, center, radius, normal, rotation, setDefaultResl))
                
                elif s["stype"] == "SplineSegment":
                    countSplineSeg = countSplineSeg + 1
                    
                    splineType = get_nested(s, "spline", "stype")
                    splineDegree = get_nested(s, "spline", "degree")
                    
                    ctrlPoints = []
                    for k in get_nested(s, "spline", "keypoints"):
                        x = 0.0
                        y = 0.0
                        z = 0.0
                        if "x" in k:
                            x = get_nested(k, "x")
                        if "y" in k:
                            y = get_nested(k, "y")
                        if "z" in k:
                            z = get_nested(k, "z")
                        ctrlPoints.append([x, y, z])
                    
                    firstParameter = get_nested(s, "spline", "firstParameter")
                    lastParameter = get_nested(s, "spline", "lastParameter")
                    
                    keypoints = []
                    if "keypoints" in s:
                        for k in get_nested(s, "keypoints"):
                            x = 0.0
                            y = 0.0
                            z = 0.0
                            if "x" in k:
                                x = get_nested(k, "x")
                            if "y" in k:
                                y = get_nested(k, "y")
                            if "z" in k:
                                z = get_nested(k, "z")
                            keypoints.append([x, y, z])
                            
                    SplineSets.append(Spline(ctrlPoints, splineDegree, setDefaultResl))
                
                else:
                    countOtherSeg = countOtherSeg + 1
                    LineSet_Segment_Dict.append({"o_" + str(countLineSeg + countSplineSeg + countArch + countOtherSeg) + "_" + str(countOtherSeg) : None})
                    continue
            
            logger.info(f" Number of Lines -- {countLineSeg}, Number of Arcs -- {countArch}, Number of Splines -- {countSplineSeg}")
            
            #NOTE -- Get Global scale "Dl" 
            #   [StartPt, EndPt] for Lines 
            # + [Control Points] for Splines
            # + [StartPt, EndPt] for Arcs
            
            pts = []
            for s in range(0, len(SplineSets)):
                for c in range(0, len(SplineSets[s].ctrlVecs)):
                    pts.append(SplineSets[s].ctrlVecs[c])
            for l in range(0, len(LineSets)):
                pts.append(LineSets[l].StartPt)
                pts.append(LineSets[l].EndPt)
            for a in range(0, len(ArcSets)):
                pts.append(ArcSets[a].StartPt)
                pts.append(ArcSets[a].EndPt)
            
            # print(np.array(pts).shape)
            
            Dl = np.linalg.norm(np.amax(np.array(pts), 1) - np.amin(np.array(pts), 1))
            
            #Combine All Splines
            SplineSets_combined = o3d.geometry.LineSet()
            for s in range(0, len(SplineSets)):
                Sl = np.linalg.norm(np.amax(SplineSets[s].ctrlVecs, 1) - np.amin(SplineSets[s].ctrlVecs, 1))
                pts, SplineSeg = SplineSets[s].genSpline(math.ceil(100 * Sl/Dl) + 2 if setDefaultResl == False else 40)
                SplineSets_combined += SplineSeg
                
            #Combine all Lines 
            LineSets_combined = o3d.geometry.LineSet()
            for l in range(0, len(LineSets)):
                Ll = np.linalg.norm(np.array(LineSets[l].StartPt) - np.array(LineSets[l].EndPt))
                pts, LineSeg = LineSets[l].genLine(math.ceil(1000 * Ll/Dl) + 2 if setDefaultResl == False else 40)
                LineSets_combined += LineSeg
                
            # Combine all Arcs
            ArcSets_combined = o3d.geometry.LineSet()
            for a in range(0, len(ArcSets)):
                Al = 2 * ArcSets[a].r
                pts, ArcSeg = ArcSets[a].gen3DArc(math.ceil(10000 * Al/Dl) + 2 if setDefaultResl == False else 40)
                ArcSets_combined += ArcSeg
                
            if debug_view == True:
                o3d.visualization.draw_geometries([SplineSets_combined, LineSets_combined, ArcSets_combined])

            if outFilePath is not None: 
                o3d.io.write_line_set(outFilePath, SplineSets_combined + LineSets_combined + ArcSets_combined)
            
            return SplineSets_combined, ArcSets_combined, LineSets_combined
        else:
            return None, None, None

    return genParametricCurves(AnnotFile, outFilePath, setDefaultResl, debug_view)

def genPointsfromJson_one(inpath, outpath, setDefaultResl, debug_view, fpath):
    """Generate Line Sets Given Parametric Curve Segments"""
    logger.info(f"loading {fpath}")
    
    if outpath is not None: 
        opath = os.path.splitext(fpath.replace(inpath, outpath))[0] + '.ply'
        if os.path.exists(opath):
            logger.info(f"already processed {opath}")
            return
        os.makedirs(os.path.dirname(opath), exist_ok=True)
    else:
        opath = None
    
    genPointsfromJsonFile(fpath, opath, setDefaultResl, debug_view)

def genPointsfromJsons(args):
    """Generate Directory Tree of Line Sets Given Directory Tree of Parametric Curve Segments"""
    files = glob.glob(str(args.input_jsons) + '/**/*{}'.format('json'), recursive=True)
    logger.info(f" found meshes {len(files)}")
    
    start = time.time()

    if args.nproc > 1:
        per_json_filter = partial(genPointsfromJson_one, str(args.input_jsons), str(args.output), args.defaultReslnFlag, args.debug_view)
        pool = Pool(processes=args.nproc)
        pool.map(per_json_filter, files)
        pool.close()
        pool.join()
    else:
        for f in files:
            genPointsfromJson_one(str(args.input_jsons), str(args.output), args.defaultReslnFlag, args.debug_view, f)

    logger.info(f"time elapsed:  {time.time() - start}")

def view(args):
    """view mesh and edges pairwise"""
    if args.selection is not None:
        selected = read_selection(args.selection)
        models = [os.path.join(args.input_meshes, s.replace('\\', os.sep))
                  for s in selected]
    else:
        models = glob.glob(str(args.input_meshes) +
                           '/**/*{}'.format(args.infmt_mesh), recursive=True)
    for fpath in models:
        logger.info(f"loading, {fpath}")
        m1 = o3d.io.read_triangle_mesh(fpath)
        if not m1.has_vertex_normals():
            m1.compute_vertex_normals()
        epath = str(pathlib.Path(fpath.replace(str(args.input_meshes), str(args.input_edges))).with_suffix(str(args.infmt_edges)))

        if str(args.infmt_edges) == '.ply':
            m2 = o3d.io.read_line_set(epath)
            abox = m2.get_axis_aligned_bounding_box()
            xtranslation = abox.get_extent()[0]
            o3d.visualization.draw_geometries([m1, m2.translate(np.array([1.5 * xtranslation, 0, 0]))])
        else:
            try:
                f_check = open(epath)
                m2_arcs, m2_splines, m2_lines  = genPointsfromJsonFile(epath, None, False, False)
                if m2_arcs is not None and m2_splines is not None and m2_lines is not None:  
                    m2 = m2_arcs + m2_splines + m2_lines
                    abox = m2.get_axis_aligned_bounding_box()
                    xtranslation = abox.get_extent()[0]
                    o3d.visualization.draw_geometries([m1, m2.translate(np.array([1.5 * xtranslation, 0, 0]))])
                else:
                    logger.warning(f"No sharp edges found with filename:  {epath}")
                    o3d.visualization.draw_geometries([m1])
            except IOError:
                logger.warning(f"No sharp edges found with filename:  {epath}")
                o3d.visualization.draw_geometries([m1])

def read_selection(selection):
    """read relative paths from .csv or .txt file"""
    selected = []
    if os.path.splitext(selection)[1] == '.csv':
        df = pd.read_csv(selection)
        selected = df.name
    else:
        with open(selection, 'r') as f:
            selected = f.read().splitlines()
        f.close()
    return selected

def _parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_detect_edges = subparsers.add_parser(
        "detect_edges",
        help="Detect edges in the Scans/CAD meshes.",
    )
    parser_detect_edges.add_argument("input_meshes", type=pathlib.Path, 
                                     help="Path to the directory with Scans/CAD meshes")
    parser_detect_edges.add_argument("output", type=pathlib.Path,  
                                     help="Path to the directory where the results will be saved")
    parser_detect_edges.add_argument("--infmt_mesh", choices=(['.ply', '.obj', '.stl', '.STL']), 
                                     default='.ply', help="Format of meshes to look for")
    parser_detect_edges.add_argument('--normals_rad_threshold', type=float, default=0.5, 
                                     help='threshold on the normals between angles')
    parser_detect_edges.add_argument('--nproc', type=int, default=1, 
                                     help='number of processes to use')
    parser_detect_edges.set_defaults(func=detect_edges)

    parser_view = subparsers.add_parser(
        "view",
        help="Visualize scans or CAD models and the edges as side by side",
    )
    parser_view.add_argument("input_meshes", type=pathlib.Path, 
                             help="Path to the directory with scans or CAD models")
    parser_view.add_argument("input_edges", type=pathlib.Path, 
                             help="Path to the directory with sharp edges")
    parser_view.add_argument("--infmt_mesh",  required=False, choices=(
        ['.ply', '.obj', '.stl', '.STL']), default='.ply', help="Format of meshes to look for. Default is 'ply'")
    parser_view.add_argument("--infmt_edges",  required=False, choices=(
        ['.ply', '.json']), default='.json', help="Format of sharp edges to look for. Default is 'json'")
    parser_view.add_argument('--selection', required=False,
                             help="Path to the txt with additional information")
    parser_view.set_defaults(func=view)
    

    parser_genPointsfromJsons = subparsers.add_parser(
        "genPointsfromJsons",
        help="Read sharp edges as parametric curves (Line, Arc and Spline segments) in .JSON format, generate  Line Sets, and save as .PLY",
    )
    parser_genPointsfromJsons.add_argument("input_jsons", type=pathlib.Path, 
                                           help="Path to the directory with .JSON file as Parametric Annotations")
    parser_genPointsfromJsons.add_argument("output", type=pathlib.Path, 
                                           help="Path to the directory where the results will be saved")
    parser_genPointsfromJsons.add_argument('--nproc', type=int, default=1, 
                                           help='number of processes to use')
    parser_genPointsfromJsons.add_argument("--defaultReslnFlag", type=bool, default=False, 
                                           help="Set it False if you need uniformly sampled Points on the Curves, Setting it to True may give non-uniform point sampling on the Curves")
    parser_genPointsfromJsons.add_argument("--debug_view", type=bool, default=False, 
                                           help="Set it to True visualizing the output (NOT recommended when files are Large in numbers)")
    parser_genPointsfromJsons.set_defaults(func=genPointsfromJsons)


    args = parser.parse_args()
    # Ensure the help message is displayed when no command is provided.
    if "func" not in args:
        parser.print_help()
        sys.exit(1)

    return args

def main():
    args = _parse_args()
    args.func(args)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
