"""Build the scene: a black cat riding an electric scooter along a road through hills.

Usage:
    blender -b --factory-startup --python scripts/build_scene.py

Writes build/cat_scooter.blend. Render it with:
    blender -b build/cat_scooter.blend -a                                  # MP4
    blender -b build/cat_scooter.blend -o //preview/f_#### -F PNG -f 1     # still
"""

import bisect
import math
import random
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector, kdtree, noise

ROOT = Path(__file__).resolve().parent.parent
BLEND_PATH = ROOT / "build" / "cat_scooter.blend"

FPS = 24
FRAME_END = 288
SEED = 7

TERRAIN_SIZE = 260.0
TERRAIN_SUBDIV = 400
ROAD_HALF_WIDTH = 1.3
ROAD_BLEND = 3.5
ROAD_SINK = 0.06  # terrain under the road sits this far below the road surface
ROAD_STEP = 0.25

# Scooter local frame: +X forward, +Y left, +Z up, origin on the ground below the deck.
WHEEL_R = 0.075
REAR_AXLE = Vector((-0.27, 0.0, WHEEL_R))
FRONT_AXLE = Vector((0.29, 0.0, WHEEL_R))
DECK_TOP = 0.117
FORK_Y = 0.036
FORK_DIR = Vector((-0.015, 0.0, 0.095))  # front axle to fork crown
SLIDER_TOP = 0.6  # fraction of FORK_DIR covered by the lower fork sliders
SWINGARM_PIVOT = Vector((-0.20, 0.0, 0.092))
SHOCK_Y = 0.045
SHOCK_TOP = Vector((-0.25, 0.0, 0.158))
LANE_OFFSET = -0.45  # metres left of the centre line (negative: right lane)
BASE_SPEED = 4.5  # m/s
RIDE_START_X = -22.0

# Speed bumps, as distances along the road from the ride start (m).
BUMPS = [8.0, 27.0, 46.0]
BUMP_HEIGHT = 0.05
BUMP_LENGTH = 0.35
ROAD_ROUGHNESS = 0.003

# Suspension: two springs on a body with heave and pitch.
SPRING_HZ = 3.0
SPRING_DAMPING = 0.25  # damping ratio
SUBSTEPS = 10

CAT_PIVOT = Vector((-0.04, 0.0, DECK_TOP))
NECK = Vector((0.09, 0.0, 0.52))
TAIL_BASE = Vector((-0.13, 0.0, 0.21))

SUN_ELEVATION = 4.0
SUN_AZIMUTH = 15.0  # degrees from +X towards +Y; the ride starts heading +X, into the sunset
CLOUD_SCALE = 1.2
CLOUD_COVER = 0.45  # noise threshold where clouds begin; lower is more cloud

# Camera path keys: (frame, angle from forward towards left in degrees, distance, height).
CAMERA_KEYS = [
    (1, 168.0, 2.3, 0.65),
    (70, 150.0, 2.0, 0.50),
    (140, 95.0, 1.7, 0.30),
    (200, 45.0, 1.9, 0.35),
    (240, 20.0, 2.2, 0.45),
    (288, -35.0, 2.6, 0.55),
]


def smoothstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def T(x, y, z):
    return Matrix.Translation((x, y, z))


def S(x, y, z):
    return Matrix.Diagonal((x, y, z, 1.0))


def R(angle, axis):
    return Matrix.Rotation(angle, 4, axis)


def align_z(direction):
    """Rotation matrix that maps local +Z onto direction."""
    return Vector(direction).to_track_quat('Z', 'Y').to_matrix().to_4x4()


# --------------------------------------------------------------------------- terrain


def fbm(x, y, octaves=5, z=0.37):
    total, amp, freq = 0.0, 1.0, 1.0
    for _ in range(octaves):
        total += amp * noise.noise(Vector((x * freq, y * freq, z)), noise_basis='PERLIN_ORIGINAL')
        amp *= 0.5
        freq *= 2.0
    return total


def base_height(x, y):
    h = 12.0 * fbm(x / 35.0, y / 35.0)
    ring = smoothstep(55.0, 125.0, math.hypot(x, y))
    return h + ring * (10.0 + 8.0 * fbm(x / 60.0 + 5.0, y / 60.0 + 5.0, octaves=3))


def centre_y(x):
    return 7.0 * math.sin(x / 16.0) + 2.5 * math.sin(x / 6.5 + 1.3)


def smooth(values, radius, passes):
    n = len(values)
    for _ in range(passes):
        prefix = [0.0]
        for v in values:
            prefix.append(prefix[-1] + v)
        values = [
            (prefix[min(n, i + radius + 1)] - prefix[max(0, i - radius)])
            / (min(n, i + radius + 1) - max(0, i - radius))
            for i in range(n)
        ]
    return values


class Road:
    """Road centre line sampled every ROAD_STEP metres; z follows the smoothed terrain."""

    def __init__(self):
        dense = [Vector((x, centre_y(x))) for x in (-122.0 + i * 0.02 for i in range(12201))]
        flat = [dense[0]]
        acc, target = 0.0, ROAD_STEP
        for a, b in zip(dense, dense[1:]):
            seg = (b - a).length
            while acc + seg >= target:
                flat.append(a.lerp(b, (target - acc) / seg))
                target += ROAD_STEP
            acc += seg

        zs = smooth([base_height(p.x, p.y) for p in flat], radius=24, passes=3)
        self.points = [Vector((p.x, p.y, z)) for p, z in zip(flat, zs)]
        n = len(self.points)

        self.tangents = []
        self.normals = []
        for i in range(n):
            t = (self.points[min(i + 1, n - 1)] - self.points[max(i - 1, 0)]).normalized()
            self.tangents.append(t)
            self.normals.append(Vector((-t.y, t.x, 0.0)).normalized())

        self.cum = [0.0]
        for a, b in zip(self.points, self.points[1:]):
            self.cum.append(self.cum[-1] + (b - a).length)

        self.kd = kdtree.KDTree(n)
        for i, p in enumerate(self.points):
            self.kd.insert((p.x, p.y, 0.0), i)
        self.kd.balance()
        self.bumps = []

    def bump(self, s):
        """Height of the road surface above the smooth centre line at arc length s."""
        h = ROAD_ROUGHNESS * noise.noise(Vector((s * 3.0, 0.0, 7.7)))
        for b in self.bumps:
            if abs(s - b) < BUMP_LENGTH / 2:
                h += BUMP_HEIGHT * math.cos(math.pi * (s - b) / BUMP_LENGTH) ** 2
        return h

    def sample(self, s):
        """Position, unit tangent and horizontal left normal at arc length s."""
        i = min(max(bisect.bisect_right(self.cum, s) - 1, 0), len(self.cum) - 2)
        t = (s - self.cum[i]) / (self.cum[i + 1] - self.cum[i])
        pos = self.points[i].lerp(self.points[i + 1], t)
        tan = self.tangents[i].lerp(self.tangents[i + 1], t).normalized()
        nrm = self.normals[i].lerp(self.normals[i + 1], t).normalized()
        return pos, tan, nrm

    def arc_length_at_x(self, x):
        i = next(i for i, p in enumerate(self.points) if p.x >= x)
        return self.cum[i]

    def distance(self, x, y):
        _, idx, dist = self.kd.find((x, y, 0.0))
        return idx, dist

    def terrain_height(self, x, y):
        idx, dist = self.distance(x, y)
        base = base_height(x, y)
        if dist >= ROAD_HALF_WIDTH + ROAD_BLEND:
            return base
        road = self.points[idx].z - ROAD_SINK
        return lerp(road, base, smoothstep(ROAD_HALF_WIDTH, ROAD_HALF_WIDTH + ROAD_BLEND, dist))


# --------------------------------------------------------------------------- helpers


def link(obj, collection, parent=None, location=(0.0, 0.0, 0.0)):
    collection.objects.link(obj)
    obj.parent = parent
    obj.location = location
    return obj


def make_empty(name, collection, parent=None, location=(0.0, 0.0, 0.0), size=0.1):
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = 'PLAIN_AXES'
    obj.empty_display_size = size
    return link(obj, collection, parent, location)


def make_collection(name):
    coll = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(coll)
    return coll


class MeshBuilder:
    """Accumulates primitives into one bmesh. Matrices are in the rig frame; geometry is
    stored relative to origin, which becomes the object's pivot."""

    def __init__(self, origin=(0.0, 0.0, 0.0)):
        self.bm = bmesh.new()
        self.to_local = Matrix.Translation(-Vector(origin))

    def _finish(self, verts, mat, smooth_shade, flat_ngons=False):
        for face in {f for v in verts for f in v.link_faces}:
            face.material_index = mat
            face.smooth = smooth_shade and not (flat_ngons and len(face.verts) > 4)

    def box(self, center, size, mat=0, rot=None):
        m = self.to_local @ Matrix.Translation(center) @ (rot or Matrix()) @ S(*size)
        res = bmesh.ops.create_cube(self.bm, size=1.0, matrix=m)
        self._finish(res['verts'], mat, False)

    def cylinder(self, matrix, radius, depth, mat=0, segments=24, radius2=None, smooth_shade=True):
        res = bmesh.ops.create_cone(
            self.bm, cap_ends=True, cap_tris=False, segments=segments, radius1=radius,
            radius2=radius if radius2 is None else radius2, depth=depth, matrix=self.to_local @ matrix,
        )
        self._finish(res['verts'], mat, smooth_shade, flat_ngons=True)

    def cone(self, matrix, radius, height, mat=0, segments=16, smooth_shade=True):
        """Cone with its base at the matrix origin, pointing along +Z."""
        self.cylinder(matrix @ T(0, 0, height / 2), radius, height, mat, segments, radius2=0.0,
                      smooth_shade=smooth_shade)

    def ellipsoid(self, center, radii, mat=0, rot=None, segments=24, rings=12):
        m = self.to_local @ Matrix.Translation(center) @ (rot or Matrix()) @ S(*radii)
        res = bmesh.ops.create_uvsphere(self.bm, u_segments=segments, v_segments=rings, radius=1.0, matrix=m)
        self._finish(res['verts'], mat, True)

    def ico(self, matrix, radius, mat=0, subdiv=1):
        res = bmesh.ops.create_icosphere(self.bm, subdivisions=subdiv, radius=radius, matrix=self.to_local @ matrix)
        self._finish(res['verts'], mat, False)

    def limb(self, p0, p1, r0, r1=None, mat=0):
        """Capsule between two points."""
        r1 = r0 if r1 is None else r1
        p0, p1 = Vector(p0), Vector(p1)
        d = p1 - p0
        self.cylinder(Matrix.Translation((p0 + p1) / 2) @ align_z(d), r0, d.length, mat, radius2=r1)
        self.ellipsoid(p0, (r0, r0, r0), mat, segments=16, rings=8)
        self.ellipsoid(p1, (r1, r1, r1), mat, segments=16, rings=8)

    def rod(self, p0, p1, r, mat=0, segments=12):
        p0, p1 = Vector(p0), Vector(p1)
        d = p1 - p0
        self.cylinder(Matrix.Translation((p0 + p1) / 2) @ align_z(d), r, d.length, mat, segments)

    def torus(self, matrix, major, minor, mat=0, seg_major=40, seg_minor=12):
        m = self.to_local @ matrix
        verts = []
        for i in range(seg_major):
            a = 2.0 * math.pi * i / seg_major
            for j in range(seg_minor):
                b = 2.0 * math.pi * j / seg_minor
                ring = major + minor * math.cos(b)
                verts.append(self.bm.verts.new(m @ Vector((ring * math.cos(a), ring * math.sin(a), minor * math.sin(b)))))
        for i in range(seg_major):
            i2 = (i + 1) % seg_major
            for j in range(seg_minor):
                j2 = (j + 1) % seg_minor
                self.bm.faces.new((verts[i * seg_minor + j], verts[i2 * seg_minor + j],
                                   verts[i2 * seg_minor + j2], verts[i * seg_minor + j2]))
        self._finish(verts, mat, True)

    def coil(self, radius, wire, length, turns, mat=0, ring=6):
        """Helical spring along +Z from the origin."""
        n = int(turns * 24)
        rings = []
        for i in range(n + 1):
            a = 2.0 * math.pi * turns * i / n
            centre = Vector((radius * math.cos(a), radius * math.sin(a), length * i / n))
            tangent = Vector((-math.sin(a) * radius * 2 * math.pi * turns, math.cos(a) * radius * 2 * math.pi * turns, length)).normalized()
            side = Vector((math.cos(a), math.sin(a), 0.0))
            up = tangent.cross(side)
            rings.append([self.bm.verts.new(self.to_local @ (centre + wire * (side * math.cos(b) + up * math.sin(b))))
                          for b in (2.0 * math.pi * k / ring for k in range(ring))])
        for r0, r1 in zip(rings, rings[1:]):
            for k in range(ring):
                self.bm.faces.new((r0[k], r0[(k + 1) % ring], r1[(k + 1) % ring], r1[k]))
        self._finish([v for r in rings for v in r], mat, True)

    def to_mesh(self, name, materials):
        mesh = bpy.data.meshes.new(name)
        self.bm.normal_update()
        self.bm.to_mesh(mesh)
        self.bm.free()
        for mat in materials:
            mesh.materials.append(mat)
        return mesh

    def to_object(self, name, materials, collection, parent=None, location=(0.0, 0.0, 0.0)):
        obj = bpy.data.objects.new(name, self.to_mesh(name, materials))
        return link(obj, collection, parent, location)


# --------------------------------------------------------------------------- materials


def principled_node(mat):
    nt = mat.node_tree
    for node in nt.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            return node
    out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None) or nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return bsdf


def set_inputs(node, values):
    for key, value in values.items():
        if key in node.inputs:
            node.inputs[key].default_value = value
        else:
            print(f"warning: {node.name} has no input {key!r}")


def make_material(name, color, roughness=0.5, metallic=0.0, emission=None, strength=0.0, inputs=None):
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None:
        mat.use_nodes = True
    bsdf = principled_node(mat)
    set_inputs(bsdf, {"Base Color": (*color, 1.0), "Roughness": roughness, "Metallic": metallic})
    if emission:
        set_inputs(bsdf, {"Emission Color": (*emission, 1.0), "Emission Strength": strength})
    set_inputs(bsdf, inputs or {})
    mat.diffuse_color = (*color, 1.0)
    return mat


def ramp_colors(ramp, low, high, positions=(0.0, 1.0)):
    elements = ramp.color_ramp.elements
    elements[0].color, elements[0].position = (*low, 1.0), positions[0]
    elements[1].color, elements[1].position = (*high, 1.0), positions[1]


def textured_material(name, low, high, patch_scale, bump_scale, roughness, bump_strength=0.2):
    """Two-colour noise patches with a fine bump, in object space."""
    mat = make_material(name, low, roughness)
    nt = mat.node_tree
    bsdf = principled_node(mat)
    coord = nt.nodes.new('ShaderNodeTexCoord')
    patches = nt.nodes.new('ShaderNodeTexNoise')
    patches.inputs['Scale'].default_value = patch_scale
    patches.inputs['Detail'].default_value = 6.0
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp_colors(ramp, low, high, (0.35, 0.65))
    fine = nt.nodes.new('ShaderNodeTexNoise')
    fine.inputs['Scale'].default_value = bump_scale
    bump = nt.nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = bump_strength
    nt.links.new(coord.outputs['Object'], patches.inputs['Vector'])
    nt.links.new(coord.outputs['Object'], fine.inputs['Vector'])
    nt.links.new(patches.outputs['Fac'], ramp.inputs['Fac'])
    nt.links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    nt.links.new(fine.outputs['Fac'], bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    return mat


def random_tint_material(name, low, high, roughness):
    """Colour varies per object instance through Object Info > Random."""
    mat = make_material(name, low, roughness)
    nt = mat.node_tree
    info = nt.nodes.new('ShaderNodeObjectInfo')
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp_colors(ramp, low, high)
    nt.links.new(info.outputs['Random'], ramp.inputs['Fac'])
    nt.links.new(ramp.outputs['Color'], principled_node(mat).inputs['Base Color'])
    return mat


def build_materials():
    m = {
        "grass": textured_material("Grass", (0.035, 0.12, 0.02), (0.20, 0.32, 0.06), 0.06, 4.0, 0.9),
        "asphalt": textured_material("Asphalt", (0.03, 0.03, 0.03), (0.05, 0.05, 0.05), 8.0, 40.0, 0.95, 0.1),
        "paint": make_material("RoadPaint", (0.85, 0.82, 0.70), 0.6),
        "wood": make_material("FenceWood", (0.20, 0.12, 0.06), 0.8),
        "wire": make_material("FenceWire", (0.4, 0.4, 0.42), 0.4, metallic=1.0),
        "bark": make_material("Bark", (0.10, 0.06, 0.035), 0.9),
        "leaves": random_tint_material("Leaves", (0.02, 0.10, 0.025), (0.12, 0.25, 0.04), 0.8),
        "blades": make_material("GrassBlades", (0.07, 0.20, 0.03), 0.8),
        "rock": textured_material("Rock", (0.08, 0.075, 0.07), (0.16, 0.15, 0.13), 3.0, 12.0, 0.8, 0.4),
        "frame": make_material("ScooterOrange", (0.90, 0.22, 0.02), 0.35, inputs={"Coat Weight": 0.6}),
        "rubber": make_material("Rubber", (0.015, 0.015, 0.015), 0.85),
        "alu": make_material("Aluminium", (0.65, 0.66, 0.68), 0.3, metallic=1.0),
        "headlight": make_material("Headlight", (1.0, 0.95, 0.85), 0.2, emission=(1.0, 0.95, 0.85), strength=30.0),
        "brakelight": make_material("BrakeLight", (0.8, 0.02, 0.02), 0.3, emission=(1.0, 0.03, 0.02), strength=12.0),
        "underglow": make_material("Underglow", (0.1, 0.8, 1.0), 0.3, emission=(0.1, 0.8, 1.0), strength=12.0),
        "screen": make_material("Display", (0.05, 0.4, 0.6), 0.2, emission=(0.1, 0.6, 1.0), strength=2.0),
        "fur": make_material("BlackFur", (0.008, 0.008, 0.009), 0.5,
                             inputs={"Sheen Weight": 0.15, "Sheen Tint": (0.6, 0.6, 0.65, 1.0)}),
        "eye": make_material("CatEye", (0.75, 0.62, 0.04), 0.05, emission=(0.75, 0.62, 0.04), strength=0.4),
        "pupil": make_material("Pupil", (0.0, 0.0, 0.0), 0.05),
        "nose": make_material("CatNose", (0.10, 0.04, 0.05), 0.4),
        "inner_ear": make_material("InnerEar", (0.12, 0.05, 0.06), 0.6),
        "whisker": make_material("Whisker", (0.75, 0.75, 0.72), 0.4),
        "collar": make_material("Collar", (0.7, 0.03, 0.05), 0.5),
        "gold": make_material("Bell", (1.0, 0.75, 0.3), 0.2, metallic=1.0),
        "bump_yellow": make_material("BumpYellow", (0.9, 0.65, 0.02), 0.6),
        "spring": make_material("Spring", (0.9, 0.75, 0.05), 0.3, metallic=1.0),
        "strands": strand_material(),
    }
    return m


def strand_material():
    """Black fur strands that turn slightly grey towards the tips."""
    mat = make_material("FurStrands", (0.006, 0.006, 0.007), 0.6)
    nt = mat.node_tree
    info = nt.nodes.new('ShaderNodeHairInfo')
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp_colors(ramp, (0.005, 0.005, 0.006), (0.035, 0.034, 0.036), (0.3, 1.0))
    nt.links.new(info.outputs['Intercept'], ramp.inputs['Fac'])
    nt.links.new(ramp.outputs['Color'], principled_node(mat).inputs['Base Color'])
    return mat


# --------------------------------------------------------------------------- environment


def build_terrain(road, mats, coll):
    n = TERRAIN_SUBDIV + 1
    half = TERRAIN_SIZE / 2.0
    step = TERRAIN_SIZE / TERRAIN_SUBDIV
    verts = []
    for j in range(n):
        y = -half + j * step
        for i in range(n):
            x = -half + i * step
            verts.append((x, y, road.terrain_height(x, y)))
    faces = [(j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i)
             for j in range(n - 1) for i in range(n - 1)]
    mesh = bpy.data.meshes.new("Terrain")
    mesh.from_pydata(verts, [], faces)
    mesh.shade_smooth()
    mesh.materials.append(mats["grass"])
    link(bpy.data.objects.new("Terrain", mesh), coll)


def build_road(road, mats, coll):
    w = ROAD_HALF_WIDTH
    verts, faces = [], []
    for p, nrm in zip(road.points, road.normals):
        low = Vector((0, 0, -0.25))
        verts += [p + nrm * (w + 0.15) + low, p + nrm * w, p - nrm * w, p - nrm * (w + 0.15) + low]
    for i in range(len(road.points) - 1):
        a, b = 4 * i, 4 * (i + 1)
        for k in range(3):
            faces.append((a + k, a + k + 1, b + k + 1, b + k))
    mesh = bpy.data.meshes.new("Road")
    mesh.from_pydata(verts, [], faces)
    mesh.materials.append(mats["asphalt"])
    link(bpy.data.objects.new("Road", mesh), coll)

    # Dashed centre line: 0.9 m dashes every 2 m.
    verts, faces = [], []
    dash, period = int(0.9 / ROAD_STEP), int(2.0 / ROAD_STEP)
    up = Vector((0, 0, 0.004))
    for start in range(0, len(road.points) - dash, period):
        base = len(verts)
        for i in range(start, start + dash + 1):
            p, nrm = road.points[i] + up, road.normals[i]
            verts += [p + nrm * 0.035, p - nrm * 0.035]
        for k in range(dash):
            faces.append((base + 2 * k, base + 2 * k + 2, base + 2 * k + 3, base + 2 * k + 1))
    mesh = bpy.data.meshes.new("RoadLine")
    mesh.from_pydata(verts, [], faces)
    mesh.materials.append(mats["paint"])
    link(bpy.data.objects.new("RoadLine", mesh), coll)


def build_speed_bumps(road, mats, coll):
    """Rubber humps across the road, striped yellow and black."""
    stripes, steps = 10, 14
    verts, faces, face_mats = [], [], []
    for b in road.bumps:
        base = len(verts)
        for k in range(steps + 1):
            ds = -BUMP_LENGTH / 2 + BUMP_LENGTH * k / steps
            centre, _, nrm = road.sample(b + ds)
            centre.z += road.bump(b + ds) - ROAD_ROUGHNESS + 0.001
            for j in range(stripes + 1):
                verts.append(centre + nrm * lerp(-ROAD_HALF_WIDTH, ROAD_HALF_WIDTH, j / stripes))
        for k in range(steps):
            for j in range(stripes):
                a = base + k * (stripes + 1) + j
                faces.append((a, a + 1, a + stripes + 2, a + stripes + 1))
                face_mats.append(j % 2)
        for j in (0, stripes):
            side = [base + k * (stripes + 1) + j for k in range(steps + 1)]
            faces.append(tuple(side if j else reversed(side)))
            face_mats.append(1)
    mesh = bpy.data.meshes.new("SpeedBumps")
    mesh.from_pydata(verts, [], faces)
    mesh.polygons.foreach_set("material_index", face_mats)
    mesh.materials.append(mats["bump_yellow"])
    mesh.materials.append(mats["rubber"])
    link(bpy.data.objects.new("SpeedBumps", mesh), coll)


def build_fence(road, mats, coll):
    """Posts every 3 m on the left verge, with two wires."""
    b = MeshBuilder()
    posts = []
    for i in range(0, len(road.points), int(3.0 / ROAD_STEP)):
        p = road.points[i] + road.normals[i] * (ROAD_HALF_WIDTH + 1.0)
        p.z = road.terrain_height(p.x, p.y) - 0.1
        posts.append(p)
        b.box(p + Vector((0, 0, 0.5)), (0.07, 0.07, 1.0), mat=0, rot=R(random.uniform(-0.05, 0.05), 'X'))
    for a, c in zip(posts, posts[1:]):
        for h in (0.55, 0.85):
            b.rod(a + Vector((0, 0, h)), c + Vector((0, 0, h)), 0.004, mat=1, segments=5)
    b.to_object("Fence", [mats["wood"], mats["wire"]], coll)


def conifer_mesh(mats):
    b = MeshBuilder()
    b.cylinder(T(0, 0, 0.6), 0.18, 1.2, mat=0, segments=8, smooth_shade=False)
    for radius, height, z in ((1.4, 2.2, 0.8), (1.1, 1.9, 1.9), (0.75, 1.6, 2.9)):
        b.cone(T(0, 0, z), radius, height, mat=1, segments=9, smooth_shade=False)
    return b.to_mesh("Conifer", [mats["bark"], mats["leaves"]])


def broadleaf_mesh(mats):
    b = MeshBuilder()
    b.cylinder(T(0, 0, 0.9), 0.15, 1.8, mat=0, segments=8, smooth_shade=False)
    b.ico(T(0, 0, 2.6) @ S(1, 1, 0.85), 1.4, mat=1)
    b.ico(T(0.6, 0.3, 3.1), 1.0, mat=1)
    b.ico(T(-0.5, -0.4, 2.9), 0.9, mat=1)
    return b.to_mesh("Broadleaf", [mats["bark"], mats["leaves"]])


def scatter_trees(road, mats, coll, rng):
    templates = [conifer_mesh(mats), broadleaf_mesh(mats)]
    placed = 0
    while placed < 450:
        x, y = rng.uniform(-120, 120), rng.uniform(-120, 120)
        if road.distance(x, y)[1] < 6.0 + rng.uniform(0, 4):
            continue
        if noise.noise(Vector((x / 30.0, y / 30.0, 3.3))) < -0.05:
            continue
        mesh = templates[0] if rng.random() < 0.55 else templates[1]
        obj = link(bpy.data.objects.new(f"Tree.{placed:03d}", mesh), coll,
                   location=(x, y, road.terrain_height(x, y) - 0.1))
        obj.rotation_euler.z = rng.uniform(0, 2 * math.pi)
        obj.scale = (rng.uniform(0.7, 1.6),) * 3
        placed += 1


def scatter_ground_detail(road, mats, coll, rng, ride_x):
    """Grass tufts and rocks near the stretch of road the ride covers, merged into two meshes."""
    lo, hi = ride_x[0] - 10.0, ride_x[1] + 15.0
    tufts = MeshBuilder()
    count = 0
    while count < 900:
        x, y = rng.uniform(lo, hi), rng.uniform(-20, 20)
        d = road.distance(x, y)[1]
        if not ROAD_HALF_WIDTH + 0.1 < d < 7.0:
            continue
        z = road.terrain_height(x, y)
        for _ in range(7):
            tilt = R(rng.uniform(-0.35, 0.35), 'X') @ R(rng.uniform(-0.35, 0.35), 'Y')
            offset = T(x + rng.uniform(-0.06, 0.06), y + rng.uniform(-0.06, 0.06), z - 0.02)
            tufts.cone(offset @ tilt, 0.012, rng.uniform(0.12, 0.28), mat=0, segments=4, smooth_shade=False)
        count += 1
    tufts.to_object("GrassTufts", [mats["blades"]], coll)

    rocks = MeshBuilder()
    count = 0
    while count < 80:
        x, y = rng.uniform(lo, hi), rng.uniform(-25, 25)
        d = road.distance(x, y)[1]
        if not ROAD_HALF_WIDTH + 1.2 < d < 14.0:
            continue
        size = rng.uniform(0.15, 0.6)
        m = (T(x, y, road.terrain_height(x, y) - size * 0.3) @ R(rng.uniform(0, 6.3), 'Z')
             @ S(size * rng.uniform(0.8, 1.4), size, size * rng.uniform(0.5, 0.8)))
        rocks.ico(m, 1.0, mat=0)
        count += 1
    rocks.to_object("Rocks", [mats["rock"]], coll)


def socket(sockets, name, kind):
    """Socket by name and type, for nodes such as Mix that have several sockets per name."""
    return next(s for s in sockets if s.name == name and s.type == kind)


def math_node(nt, operation, a, b):
    node = nt.nodes.new('ShaderNodeMath')
    node.operation = operation
    for sock, value in zip(node.inputs, (a, b)):
        if isinstance(value, float):
            sock.default_value = value
        else:
            nt.links.new(value, sock)
    return node.outputs[0]


def mix_colour(nt, factor, a, b, blend='MIX'):
    node = nt.nodes.new('ShaderNodeMix')
    node.data_type = 'RGBA'
    node.blend_type = blend
    if isinstance(factor, float):
        node.inputs['Factor'].default_value = factor
    else:
        nt.links.new(factor, node.inputs['Factor'])
    for name, value in (("A", a), ("B", b)):
        sock = socket(node.inputs, name, 'RGBA')
        if isinstance(value, tuple):
            sock.default_value = (*value, 1.0)
        else:
            nt.links.new(value, sock)
    return socket(node.outputs, "Result", 'RGBA')


def build_world(scene):
    """Sunset sky with a cloud layer: noise projected onto a horizontal plane above the scene.
    Clouds take the sky colour, tinted orange towards the sun and violet away from it."""
    world = bpy.data.worlds.new("Sky")
    scene.world = world
    if world.node_tree is None:
        world.use_nodes = True
    nt = world.node_tree
    bg = next((n for n in nt.nodes if n.type == 'BACKGROUND'), None) or nt.nodes.new('ShaderNodeBackground')
    out = next((n for n in nt.nodes if n.type == 'OUTPUT_WORLD'), None) or nt.nodes.new('ShaderNodeOutputWorld')
    el, az = math.radians(SUN_ELEVATION), math.radians(SUN_AZIMUTH)
    to_sun = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))

    sky = nt.nodes.new('ShaderNodeTexSky')
    sky.sky_type = 'MULTIPLE_SCATTERING'
    sky.sun_disc = True
    sky.sun_size = math.radians(1.5)
    sky.sun_elevation = el
    sky.sun_rotation = math.atan2(to_sun.x, to_sun.y)

    coord = nt.nodes.new('ShaderNodeTexCoord')
    xyz = nt.nodes.new('ShaderNodeSeparateXYZ')
    nt.links.new(coord.outputs['Generated'], xyz.inputs[0])
    height = math_node(nt, 'MAXIMUM', xyz.outputs['Z'], 0.03)
    plane = nt.nodes.new('ShaderNodeCombineXYZ')
    nt.links.new(math_node(nt, 'DIVIDE', xyz.outputs['X'], height), plane.inputs['X'])
    nt.links.new(math_node(nt, 'DIVIDE', xyz.outputs['Y'], height), plane.inputs['Y'])
    clouds = nt.nodes.new('ShaderNodeTexNoise')
    clouds.inputs['Scale'].default_value = CLOUD_SCALE
    clouds.inputs['Detail'].default_value = 8.0
    clouds.inputs['Roughness'].default_value = 0.6
    nt.links.new(plane.outputs[0], clouds.inputs['Vector'])
    density = nt.nodes.new('ShaderNodeMapRange')
    density.interpolation_type = 'SMOOTHSTEP'
    density.inputs['From Min'].default_value = CLOUD_COVER
    density.inputs['From Max'].default_value = CLOUD_COVER + 0.18
    nt.links.new(clouds.outputs['Fac'], density.inputs['Value'])
    horizon = nt.nodes.new('ShaderNodeMapRange')
    horizon.interpolation_type = 'SMOOTHSTEP'
    horizon.inputs['From Min'].default_value = 0.01
    horizon.inputs['From Max'].default_value = 0.2
    nt.links.new(xyz.outputs['Z'], horizon.inputs['Value'])
    cover = math_node(nt, 'MULTIPLY', density.outputs['Result'], horizon.outputs['Result'])

    sunward = nt.nodes.new('ShaderNodeVectorMath')
    sunward.operation = 'DOT_PRODUCT'
    nt.links.new(coord.outputs['Generated'], sunward.inputs[0])
    sunward.inputs[1].default_value = to_sun
    glow = nt.nodes.new('ShaderNodeMapRange')
    glow.inputs['From Min'].default_value = -0.3
    glow.inputs['From Max'].default_value = 1.0
    nt.links.new(sunward.outputs['Value'], glow.inputs['Value'])
    lit = math_node(nt, 'POWER', glow.outputs['Result'], 2.0)
    tint = mix_colour(nt, lit, (0.8, 0.5, 0.75), (3.0, 1.3, 0.5))
    cloud_colour = mix_colour(nt, 1.0, sky.outputs['Color'], tint, blend='MULTIPLY')
    colour = mix_colour(nt, cover, sky.outputs['Color'], cloud_colour)

    bg.inputs['Strength'].default_value = 0.18
    nt.links.new(colour, bg.inputs['Color'])
    nt.links.new(bg.outputs['Background'], out.inputs['Surface'])

    light = bpy.data.lights.new("Sun", 'SUN')
    light.energy = 3.0
    light.angle = math.radians(1.5)
    light.color = (1.0, 0.55, 0.28)
    sun = link(bpy.data.objects.new("Sun", light), scene.collection)
    sun.rotation_mode = 'QUATERNION'
    sun.rotation_quaternion = to_sun.to_track_quat('Z', 'Y')


# --------------------------------------------------------------------------- scooter and cat


class Scooter:
    """Objects of the scooter rig. ScooterRig follows the smooth road; Body carries the frame
    and the cat and moves on the suspension; the wheel carriers follow the bumps."""

    def __init__(self, rig, body, front, rear, wheels, arm, springs, target):
        self.rig, self.body, self.front, self.rear = rig, body, front, rear
        self.wheels, self.arm, self.springs, self.target = wheels, arm, springs, target


def body_matrix(heave, slope):
    return T(0, 0, heave) @ R(-math.atan(slope), 'Y')


def spring_ends(heave, slope, u_front, u_rear):
    """Bottom and top of each spring in the rig frame: (front left, front right, rear left, rear right)."""
    m = body_matrix(heave, slope)
    front_axle = FRONT_AXLE + Vector((0, 0, u_front))
    rear_axle = REAR_AXLE + Vector((0, 0, u_rear))
    ends = []
    for y in (FORK_Y, -FORK_Y):
        bottom = front_axle + FORK_DIR * SLIDER_TOP + Vector((0, y, 0))
        ends.append((bottom, m @ (FRONT_AXLE + FORK_DIR * 0.93 + Vector((0, y, 0)))))
    for y in (SHOCK_Y, -SHOCK_Y):
        ends.append((rear_axle + Vector((0, y, 0)), m @ (SHOCK_TOP + Vector((0, y, 0)))))
    return ends


def build_scooter_lights(coll, body):
    """Headlight beam, tail light and underglow, carried by the sprung body."""
    def lamp(name, kind, location, energy, colour, direction=None, **settings):
        data = bpy.data.lights.new(name, kind)
        data.energy = energy
        data.color = colour
        for key, value in settings.items():
            setattr(data, key, value)
        obj = link(bpy.data.objects.new(name, data), coll, body, location)
        if direction:
            obj.rotation_mode = 'QUATERNION'
            obj.rotation_quaternion = Vector(direction).to_track_quat('-Z', 'Y')
        return obj

    lamp("HeadlightBeam", 'SPOT', (0.276, 0, 0.47), 6.0, (1.0, 0.92, 0.8), (1.0, 0.0, -0.18),
         spot_size=math.radians(55), spot_blend=0.6, shadow_soft_size=0.01)
    lamp("TailLight", 'POINT', (-0.36, 0, 0.16), 0.15, (1.0, 0.05, 0.03), shadow_soft_size=0.005)
    lamp("Underglow", 'AREA', (0, 0, 0.065), 0.6, (0.1, 0.8, 1.0), shape='RECTANGLE', size=0.4, size_y=0.12)


def build_scooter(mats, coll):
    rig = make_empty("ScooterRig", coll, size=0.3)
    rig.rotation_mode = 'QUATERNION'
    body = make_empty("ScooterBody", coll, rig, size=0.2)
    materials = [mats[k] for k in ("frame", "rubber", "alu", "headlight", "brakelight", "screen", "spring", "underglow")]
    FRAME, RUBBER, ALU, HEAD, BRAKE, SCREEN, SPRING, GLOW = range(8)
    crown = FRONT_AXLE + FORK_DIR

    b = MeshBuilder()
    b.box((0, 0, 0.10), (0.46, 0.13, 0.03), FRAME)
    b.box((0, 0, DECK_TOP - 0.0015), (0.42, 0.11, 0.003), RUBBER)
    b.box((0, 0, 0.078), (0.36, 0.10, 0.02), RUBBER)
    b.box((-0.205, 0, 0.085), (0.03, 0.09, 0.03), ALU)
    for y in (-0.064, 0.064):
        b.box((0, y, 0.086), (0.40, 0.004, 0.005), GLOW)
    for y in (-FORK_Y, FORK_Y):
        b.rod(FRONT_AXLE + FORK_DIR * 0.2 + Vector((0, y, 0)), crown + Vector((0, y, 0)), 0.007, ALU)
    b.rod((-0.22, 0, 0.11), (-0.25, 0, 0.165), 0.008, ALU)
    b.box((-0.27, 0, 0.165), (0.16, 0.11, 0.008), FRAME)
    b.box((-0.352, 0, 0.16), (0.006, 0.03, 0.01), BRAKE)
    b.limb((0.21, 0, 0.10), (0.27, 0, 0.165), 0.02, mat=FRAME)
    b.box(crown + Vector((0, 0, 0.002)), (0.03, 2 * FORK_Y + 0.03, 0.015), ALU)
    b.rod(crown, (0.23, 0, 0.56), 0.017, ALU, segments=16)
    b.rod((0.23, -0.16, 0.56), (0.23, 0.16, 0.56), 0.010, ALU)
    for sign in (-1, 1):
        b.rod((0.23, sign * 0.105, 0.56), (0.23, sign * 0.165, 0.56), 0.0145, RUBBER)
    b.box((0.235, 0, 0.572), (0.03, 0.05, 0.012), RUBBER)
    b.box((0.236, 0, 0.5785), (0.024, 0.04, 0.002), SCREEN)
    b.cylinder(T(0.262, 0, 0.47) @ R(math.pi / 2, 'Y'), 0.018, 0.02, ALU, segments=20)
    b.cylinder(T(0.2725, 0, 0.47) @ R(math.pi / 2, 'Y'), 0.014, 0.003, HEAD, segments=20)
    b.to_object("ScooterFrame", materials, coll, body)

    front = make_empty("FrontSuspension", coll, rig, FRONT_AXLE, size=0.05)
    rear = make_empty("RearSuspension", coll, rig, REAR_AXLE, size=0.05)
    wheels = []
    for name, axle, carrier in (("WheelFront", FRONT_AXLE, front), ("WheelRear", REAR_AXLE, rear)):
        w = MeshBuilder(axle)
        axis = Matrix.Translation(axle) @ R(math.pi / 2, 'X')
        w.torus(axis, WHEEL_R - 0.022, 0.022, RUBBER)
        w.cylinder(axis, 0.034, 0.03, ALU, segments=24)
        for k in range(5):
            for y in (-0.016, 0.016):
                w.box(axle + Vector((0, y, 0)), (0.05, 0.004, 0.008), FRAME, rot=R(k * 2 * math.pi / 5, 'Y') @ T(0.022, 0, 0))
        wheels.append(w.to_object(name, materials, coll, carrier))

    sliders = MeshBuilder(FRONT_AXLE)
    for y in (-FORK_Y, FORK_Y):
        offset = Vector((0, y, 0))
        sliders.rod(FRONT_AXLE + offset, FRONT_AXLE + FORK_DIR * SLIDER_TOP + offset, 0.011, ALU)
    sliders.to_object("ForkSliders", materials, coll, front)

    a = MeshBuilder(SWINGARM_PIVOT)
    for y in (-0.035, 0.035):
        a.rod(SWINGARM_PIVOT + Vector((0, y, 0)), REAR_AXLE + Vector((0, y, 0)), 0.008, ALU)
    arm = a.to_object("RearArm", materials, coll, body, SWINGARM_PIVOT)
    arm.rotation_mode = 'QUATERNION'

    springs = []
    names = ("FrontSpringL", "FrontSpringR", "RearShockL", "RearShockR")
    for i, (bottom, top) in enumerate(spring_ends(0.0, 0.0, 0.0, 0.0)):
        length = (top - bottom).length
        s = MeshBuilder()
        if i < 2:
            s.coil(0.0115, 0.0022, length, 4, SPRING)
        else:
            s.coil(0.009, 0.0022, length, 6, SPRING)
        spring = s.to_object(names[i], materials, coll, rig, bottom)
        spring.rotation_mode = 'QUATERNION'
        spring["rest_length"] = length
        springs.append(spring)

    build_scooter_lights(coll, body)
    target = make_empty("CameraTarget", coll, rig, (0.03, 0.0, 0.36))
    return Scooter(rig, body, front, rear, wheels, arm, springs, target)


def remesh(obj, voxel_size, smooth_iterations):
    """Merge overlapping primitives into one closed surface, so fur grows only on the outside."""
    rm = obj.modifiers.new("Remesh", 'REMESH')
    rm.mode = 'VOXEL'
    rm.voxel_size = voxel_size
    sm = obj.modifiers.new("Smooth", 'SMOOTH')
    sm.factor = 0.5
    sm.iterations = smooth_iterations
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph))
    old = obj.data
    name, materials = old.name, list(old.materials)
    obj.modifiers.clear()
    obj.data = mesh
    bpy.data.meshes.remove(old)
    mesh.name = name
    if not mesh.materials:
        for mat in materials:
            mesh.materials.append(mat)
    mesh.shade_smooth()


def add_fur(obj, mats, count, length, direction, density_group=None):
    """Particle hair of the given length along the normal; direction, in units of that length,
    is added to comb the strands. Blender grows hair for 4 time units at the emission velocity
    (hair_length is an alias for 4 * normal_factor)."""
    obj.data.materials.append(mats["strands"])
    ps = bpy.data.particles.new(f"{obj.name}Fur")
    ps.type = 'HAIR'
    ps.count = count
    ps.emit_from = 'FACE'
    ps.use_advanced_hair = True
    ps.normal_factor = length / 4.0
    ps.object_align_factor = [c * length / 4.0 for c in direction]
    ps.factor_random = 0.3 * length / 4.0
    ps.hair_step = 4
    ps.display_step = 2
    ps.render_step = 3
    ps.child_type = 'INTERPOLATED'
    ps.child_percent = 1
    ps.rendered_child_count = 25
    ps.clump_factor = 0.25
    ps.roughness_1 = 0.002
    ps.roughness_endpoint = 0.0015
    ps.root_radius = 1.0
    ps.tip_radius = 0.15
    ps.radius_scale = 0.0006
    ps.use_close_tip = True
    obj.modifiers.new("Fur", 'PARTICLE_SYSTEM')
    psys = obj.particle_systems[-1]
    psys.settings = ps
    ps.material_slot = mats["strands"].name  # the enum lists the slots of the objects using ps
    psys.seed = SEED
    if density_group:
        psys.vertex_group_density = density_group


def build_cat(mats, coll, scooter_body):
    cat_rig = make_empty("CatRig", coll, scooter_body, CAT_PIVOT)
    FUR = 0

    b = MeshBuilder(CAT_PIVOT)
    b.ellipsoid((-0.005, 0, 0.34), (0.085, 0.08, 0.16), FUR, rot=R(math.radians(34), 'Y'))
    b.ellipsoid((0.055, 0, 0.44), (0.075, 0.072, 0.08), FUR)
    for s in (-1, 1):
        b.ellipsoid((-0.07, s * 0.045, 0.20), (0.075, 0.045, 0.07), FUR)
        b.limb((-0.03, s * 0.05, 0.17), (-0.075, s * 0.05, 0.135), 0.024, 0.018, FUR)
        b.ellipsoid((-0.035, s * 0.05, 0.129), (0.048, 0.022, 0.014), FUR)
        b.limb((0.07, s * 0.05, 0.455), (0.215, s * 0.13, 0.572), 0.024, 0.019, FUR)
        b.ellipsoid((0.225, s * 0.132, 0.574), (0.026, 0.022, 0.02), FUR)
    body = b.to_object("CatBody", [mats["fur"]], coll, cat_rig)
    remesh(body, 0.0035, 8)
    add_fur(body, mats, 5000, 0.01, (-1.0, 0.0, -0.6))

    c = MeshBuilder(CAT_PIVOT)
    neck_dir = Vector((0.055, 0, 0.16)).normalized()
    collar_c = Vector((0.085, 0, 0.515))
    c.torus(Matrix.Translation(collar_c) @ align_z(neck_dir), 0.054, 0.01, 0, seg_minor=8)
    front = (Vector((1, 0, 0)) - neck_dir * neck_dir.x).normalized()
    c.ellipsoid(collar_c + front * 0.068 - neck_dir * 0.012, (0.012,) * 3, 1, segments=16, rings=8)
    c.to_object("CatCollar", [mats["collar"], mats["gold"]], coll, cat_rig)

    tail_curve = build_tail(mats, coll, cat_rig)
    head_rig, ears, eyes = build_head(mats, coll, cat_rig)
    return cat_rig, head_rig, tail_curve, ears, eyes


TAIL_REST = [(0, 0, 0), (-0.09, 0, 0.02), (-0.16, 0, 0.09), (-0.19, 0, 0.19), (-0.17, 0, 0.27), (-0.12, 0, 0.30)]


def tail_tube(length, r0, r1, rings=48, ring=12):
    """Tapered tube along +X with a rounded tip."""
    bm = bmesh.new()
    loops = []
    for i in range(rings):
        x = length * i / rings
        r = lerp(r0, r1, i / rings) * math.sqrt(max(0.0, 1.0 - (max(0.0, x - (length - r1)) / r1) ** 2))
        loops.append([bm.verts.new((x, r * math.cos(2 * math.pi * k / ring), r * math.sin(2 * math.pi * k / ring)))
                      for k in range(ring)])
    tip = bm.verts.new((length, 0.0, 0.0))
    for l0, l1 in zip(loops, loops[1:]):
        for k in range(ring):
            bm.faces.new((l0[k], l0[(k + 1) % ring], l1[(k + 1) % ring], l1[k]))
    for k in range(ring):
        bm.faces.new((loops[-1][k], loops[-1][(k + 1) % ring], tip))
    bm.faces.new(list(reversed(loops[0])))
    bm.normal_update()
    mesh = bpy.data.meshes.new("CatTail")
    bm.to_mesh(mesh)
    bm.free()
    mesh.shade_smooth()
    return mesh


def build_tail(mats, coll, cat_rig):
    """A tube mesh bent along an animated curve, so the fur follows the sway."""
    cu = bpy.data.curves.new("TailCurve", 'CURVE')
    cu.dimensions = '3D'
    cu.resolution_u = 12
    cu.twist_mode = 'MINIMUM'
    spline = cu.splines.new('NURBS')
    spline.points.add(len(TAIL_REST) - 1)
    for point, co in zip(spline.points, TAIL_REST):
        point.co = (*co, 1.0)
    spline.order_u = 3
    spline.use_endpoint_u = True
    curve = link(bpy.data.objects.new("TailCurve", cu), coll, cat_rig, TAIL_BASE - CAT_PIVOT)
    curve.hide_render = True

    mesh = tail_tube(spline.calc_length() * 0.97, 0.021, 0.012)
    mesh.materials.append(mats["fur"])
    tail = link(bpy.data.objects.new("CatTail", mesh), coll, cat_rig, TAIL_BASE - CAT_PIVOT)
    bend = tail.modifiers.new("Bend", 'CURVE')
    bend.object = curve
    bend.deform_axis = 'POS_X'
    add_fur(tail, mats, 900, 0.012, (0.8, 0.0, 0.0))
    return curve


def build_head(mats, coll, cat_rig):
    head_rig = make_empty("HeadRig", coll, cat_rig, NECK - CAT_PIVOT, size=0.05)
    head_rig.scale = (1.12, 1.12, 1.12)
    b = MeshBuilder(NECK)
    b.ellipsoid((0.115, 0, 0.60), (0.075, 0.08, 0.068), 0, segments=32, rings=16)
    for s in (-1, 1):
        b.ellipsoid((0.168, s * 0.022, 0.582), (0.022, 0.028, 0.024), 0)
    b.ellipsoid((0.165, 0, 0.568), (0.02, 0.022, 0.014), 0)
    head = b.to_object("CatHead", [mats["fur"]], coll, head_rig)
    remesh(head, 0.003, 6)
    # No fur over the eyes and the nose.
    bare = [(Vector((0.172, s * 0.032, 0.617)), 0.022, 0.032) for s in (-1, 1)]
    bare.append((Vector((0.193, 0, 0.596)), 0.014, 0.024))
    group = head.vertex_groups.new(name="FurDensity")
    for v in head.data.vertices:
        p = v.co + NECK
        group.add([v.index], min(smoothstep(r0, r1, (p - c).length) for c, r0, r1 in bare), 'REPLACE')
    add_fur(head, mats, 2500, 0.006, (-1.2, 0.0, 0.0), density_group=group.name)

    f = MeshBuilder(NECK)
    for s in (-1, 1):
        for end in ((0.205, 0.115, 0.598), (0.207, 0.118, 0.583), (0.2, 0.11, 0.568)):
            f.rod((0.183, s * 0.028, 0.585), (end[0], s * end[1], end[2]), 0.0012, 1, segments=4)
    f.ellipsoid((0.193, 0, 0.596), (0.007, 0.011, 0.007), 0)
    f.to_object("CatFace", [mats["nose"], mats["whisker"]], coll, head_rig)

    ears = []
    for s in (-1, 1):
        base = Vector((0.10, s * 0.048, 0.648))
        e = MeshBuilder(base)
        e.cone(Matrix.Translation(base) @ S(0.55, 1, 1), 0.032, 0.06, 0)
        e.cone(Matrix.Translation(base + Vector((0.009, 0, 0.004))) @ S(0.35, 1, 1), 0.022, 0.044, 1)
        ear = e.to_object("EarL" if s > 0 else "EarR", [mats["fur"], mats["inner_ear"]], coll, head_rig, base - NECK)
        ear.rotation_euler = (-s * 0.35, -0.15, 0.0)
        ears.append(ear)

    eyes = []
    for s in (-1, 1):
        c = Vector((0.172, s * 0.032, 0.617))
        e = MeshBuilder(c)
        e.ellipsoid(c, (0.0175,) * 3, 0)
        e.ellipsoid(c + Vector((0.0158, 0, 0)), (0.003, 0.0045, 0.0135), 1, segments=16, rings=8)
        eyes.append(e.to_object("EyeL" if s > 0 else "EyeR", [mats["eye"], mats["pupil"]], coll, head_rig, c - NECK))
    return head_rig, ears, eyes


# --------------------------------------------------------------------------- animation


def catmull_rom(keys, frame):
    """Interpolate tuples (frame, *values) with Catmull-Rom splines."""
    frames = [k[0] for k in keys]
    i = min(max(bisect.bisect_right(frames, frame) - 1, 0), len(keys) - 2)
    p0, p1, p2, p3 = (keys[max(i - 1, 0)], keys[i], keys[i + 1], keys[min(i + 2, len(keys) - 1)])
    t = (frame - p1[0]) / (p2[0] - p1[0])
    out = []
    for k in range(1, len(p1)):
        a, b, c, d = p0[k], p1[k], p2[k], p3[k]
        out.append(0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t * t
                          + (-a + 3 * b - 3 * c + d) * t * t * t))
    return out


def heading(tangent):
    return math.atan2(tangent.y, tangent.x)


class Suspension:
    """Heave and pitch of the scooter body on one spring per axle (unit mass), driven by the
    road height under each wheel. The cat rides on the body as a second spring."""

    AXLES = (FRONT_AXLE.x, REAR_AXLE.x)

    def __init__(self):
        self.k = (2 * math.pi * SPRING_HZ) ** 2 / 2
        self.c = SPRING_DAMPING * math.sqrt(2 * self.k)
        self.inertia = 0.28 ** 2
        self.heave = self.heave_v = self.slope = self.slope_v = 0.0
        self.cat_k = (2 * math.pi * 2.5) ** 2
        self.cat_c = 2 * 0.3 * 2 * math.pi * 2.5
        self.cat = self.cat_v = 0.0

    def step(self, u, u_v, dt):
        forces = [self.k * (ui - (self.heave + self.slope * x)) + self.c * (vi - (self.heave_v + self.slope_v * x))
                  for ui, vi, x in zip(u, u_v, self.AXLES)]
        accel = sum(forces)
        self.heave_v += accel * dt
        self.slope_v += sum(f * x for f, x in zip(forces, self.AXLES)) / self.inertia * dt
        self.heave += self.heave_v * dt
        self.slope += self.slope_v * dt
        self.cat_v += (-self.cat_k * self.cat - self.cat_c * self.cat_v - accel) * dt
        self.cat += self.cat_v * dt

    def advance(self, road, s, ds, v):
        dt = ds / v / SUBSTEPS
        for k in range(SUBSTEPS):
            sk = s + ds * (k + 1) / SUBSTEPS
            u = [road.bump(sk + x) for x in self.AXLES]
            u_v = [(road.bump(sk + x + 0.01) - road.bump(sk + x - 0.01)) / 0.02 * v for x in self.AXLES]
            self.step(u, u_v, dt)


def pose_suspension(scooter, susp, road, s, frame):
    u_front, u_rear = (road.bump(s + x) for x in Suspension.AXLES)
    body = scooter.body
    body.location = (0.0, 0.0, susp.heave)
    body.rotation_euler = (0.0, -math.atan(susp.slope), 0.0)
    scooter.front.location = FRONT_AXLE + Vector((0, 0, u_front))
    scooter.rear.location = REAR_AXLE + Vector((0, 0, u_rear))
    axle = body_matrix(susp.heave, susp.slope).inverted() @ scooter.rear.location
    scooter.arm.rotation_quaternion = (REAR_AXLE - SWINGARM_PIVOT).rotation_difference(axle - SWINGARM_PIVOT)
    for obj in (body, scooter.front, scooter.rear):
        obj.keyframe_insert("location", frame=frame)
    body.keyframe_insert("rotation_euler", frame=frame)
    scooter.arm.keyframe_insert("rotation_quaternion", frame=frame)
    for spring, (bottom, top) in zip(scooter.springs, spring_ends(susp.heave, susp.slope, u_front, u_rear)):
        d = top - bottom
        spring.location = bottom
        spring.rotation_quaternion = d.to_track_quat('Z', 'Y')
        spring.scale = (1.0, 1.0, d.length / spring["rest_length"])
        for path in ("location", "rotation_quaternion", "scale"):
            spring.keyframe_insert(path, frame=frame)


def animate(scene, road, scooter, cam, cat_rig, head_rig, tail, ears, eyes):
    bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
    rig, wheels = scooter.rig, scooter.wheels
    s = road.arc_length_at_x(RIDE_START_X)
    v = BASE_SPEED
    lean = 0.0
    travelled = 0.0
    cam_heading = None
    prev_q = None
    ride_x = []
    susp = Suspension()
    bump_frames = []

    for frame in range(1, FRAME_END + 1):
        t = (frame - 1) / FPS
        centre, tan, nrm = road.sample(s)
        pos = centre + nrm * LANE_OFFSET
        ride_x.append(pos.x)

        slope = tan.z / max(Vector((tan.x, tan.y)).length, 1e-6)
        v += (BASE_SPEED * min(max(1.0 - 1.8 * slope, 0.55), 1.5) - v) * 0.06
        kappa = wrap_angle(heading(road.sample(s + 1.0)[1]) - heading(road.sample(s - 1.0)[1])) / 2.0
        lean += (min(max(math.atan(v * v * kappa / 9.81), -0.4), 0.4) - lean) * 0.15

        left = Vector((0, 0, 1)).cross(tan).normalized()
        up = tan.cross(left)
        basis = Matrix((tan, left, up)).transposed()
        q = (Matrix.Rotation(-lean, 3, tan) @ basis).to_quaternion()
        if prev_q is not None and q.dot(prev_q) < 0:
            q.negate()
        prev_q = q
        rig.location = pos
        rig.rotation_quaternion = q
        rig.keyframe_insert("location", frame=frame)
        rig.keyframe_insert("rotation_quaternion", frame=frame)

        for wheel in wheels:
            wheel.rotation_euler = (0.0, travelled / WHEEL_R, 0.0)
            wheel.keyframe_insert("rotation_euler", frame=frame)

        pose_suspension(scooter, susp, road, s, frame)
        if any(abs(s + FRONT_AXLE.x - b) < v / FPS / 2 for b in road.bumps):
            bump_frames.append(frame)

        # The cat absorbs the bumps in its legs: squash and stretch about its feet.
        squash = 1.0 + min(max(susp.cat / 0.4, -0.08), 0.06)
        cat_rig.location = CAT_PIVOT + Vector((0, 0, 0.003 * math.sin(2 * math.pi * 1.6 * t)))
        cat_rig.scale = (1.0 / math.sqrt(squash), 1.0 / math.sqrt(squash), squash)
        cat_rig.keyframe_insert("location", frame=frame)
        cat_rig.keyframe_insert("scale", frame=frame)
        # Cats keep their head level: counter the scooter's lean and look into the turn.
        head_rig.rotation_euler = (lean * 0.6, 0.05 * math.sin(2 * math.pi * 0.5 * t),
                                   min(max(kappa * 1.5, -0.25), 0.25))
        head_rig.keyframe_insert("rotation_euler", frame=frame)

        for i, point in enumerate(tail.data.splines[0].points):
            frac = i / (len(TAIL_REST) - 1)
            x, y, z = TAIL_REST[i]
            sway = 0.07 * frac ** 1.5 * math.sin(2 * math.pi * 0.7 * t - 1.2 * i)
            point.co = (x, y + sway, z, 1.0)
            point.keyframe_insert("co", frame=frame)

        cam_h = heading(tan)
        cam_heading = cam_h if cam_heading is None else cam_heading + wrap_angle(cam_h - cam_heading) * 0.08
        angle, dist, height = catmull_rom(CAMERA_KEYS, frame)
        a = cam_heading + math.radians(angle)
        cam_pos = pos + Vector((math.cos(a), math.sin(a), 0.0)) * dist + Vector((0, 0, height))
        cam_pos.z = max(cam_pos.z, road.terrain_height(cam_pos.x, cam_pos.y) + 0.25)
        cam.location = cam_pos
        cam.keyframe_insert("location", frame=frame)

        susp.advance(road, s, v / FPS, v)
        travelled += v / FPS
        s += v / FPS

    for ear, sign, start in ((ears[1], 1, 60), (ears[0], -1, 150), (ears[1], 1, 205)):
        rest = ear.rotation_euler.copy()
        for f, extra in ((start, 0.0), (start + 3, 0.45), (start + 7, 0.0)):
            ear.rotation_euler = (rest.x - sign * extra, rest.y - extra * 0.3, rest.z)
            ear.keyframe_insert("rotation_euler", frame=f)
        ear.rotation_euler = rest

    for eye in eyes:
        for f, sz in ((1, 1.0), (88, 1.0), (90, 0.1), (93, 1.0), (176, 1.0), (178, 0.1), (181, 1.0)):
            eye.scale = (1.0, 1.0, sz)
            eye.keyframe_insert("scale", frame=f)

    print(f"ride: x {ride_x[0]:.1f} -> {ride_x[-1]:.1f}, distance {travelled:.1f} m, bumps at frames {bump_frames}")
    return ride_x[0], ride_x[-1]


# --------------------------------------------------------------------------- scene


def add_bloom(scene):
    """Compositor glare so the scooter lights and the sun bloom."""
    tree = bpy.data.node_groups.new("Compositing", 'CompositorNodeTree')
    tree.interface.new_socket("Image", in_out='OUTPUT', socket_type='NodeSocketColor')
    layers = tree.nodes.new('CompositorNodeRLayers')
    glare = tree.nodes.new('CompositorNodeGlare')
    glare.inputs['Type'].default_value = 'Bloom'
    glare.inputs['Threshold'].default_value = 1.0
    glare.inputs['Strength'].default_value = 0.5
    out = tree.nodes.new('NodeGroupOutput')
    tree.links.new(layers.outputs['Image'], glare.inputs['Image'])
    tree.links.new(glare.outputs['Image'], out.inputs[0])
    scene.compositing_node_group = tree


def configure_render(scene):
    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.taa_render_samples = 64
    if hasattr(scene.eevee, "use_raytracing"):
        scene.eevee.use_raytracing = True
    scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    scene.render.fps = FPS
    scene.frame_start, scene.frame_end = 1, FRAME_END
    scene.render.hair_type = 'STRIP'
    scene.render.hair_subdiv = 1
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.5
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.look = 'AgX - Medium High Contrast'
    add_bloom(scene)
    scene.render.filepath = "//render/cat_scooter_"
    scene.render.image_settings.media_type = 'VIDEO'
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'HIGH'


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    rng = random.Random(SEED)
    random.seed(SEED)
    noise.seed_set(SEED)

    mats = build_materials()
    road = Road()
    env = make_collection("Environment")
    actors = make_collection("Actors")

    start = road.arc_length_at_x(RIDE_START_X)
    road.bumps = [start + d for d in BUMPS]

    scooter = build_scooter(mats, actors)
    cat_rig, head_rig, tail, ears, eyes = build_cat(mats, actors, scooter.body)

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 32
    cam_data.clip_start = 0.05
    cam_data.clip_end = 1000.0
    cam = link(bpy.data.objects.new("Camera", cam_data), scene.collection)
    track = cam.constraints.new('TRACK_TO')
    track.target = scooter.target
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    scene.camera = cam

    ride_x = animate(scene, road, scooter, cam, cat_rig, head_rig, tail, ears, eyes)

    build_terrain(road, mats, env)
    build_road(road, mats, env)
    build_speed_bumps(road, mats, env)
    build_fence(road, mats, env)
    scatter_trees(road, mats, env, rng)
    scatter_ground_detail(road, mats, env, rng, ride_x)
    build_world(scene)
    configure_render(scene)

    zs = [p.z for p in road.points if ride_x[0] <= p.x <= ride_x[1]]
    print(f"road height along ride: {min(zs):.1f} .. {max(zs):.1f} m")
    scene.frame_set(1)
    BLEND_PATH.parent.mkdir(exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))


if __name__ == "__main__":
    main()
