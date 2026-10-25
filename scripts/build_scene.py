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
import numpy as np
from mathutils import Matrix, Vector, kdtree, noise

ROOT = Path(__file__).resolve().parent.parent
BLEND_PATH = ROOT / "build" / "cat_scooter.blend"
TEXTURES = ROOT / "assets" / "textures"  # fetched by scripts/fetch_textures.py

FPS = 24
FRAME_END = 288
SEED = 7

TERRAIN_SIZE = 260.0
TERRAIN_SUBDIV = 400
FAR_RADIUS = 30000.0  # the far terrain is a polar grid from the edge of the near terrain to here
RANGE_START = (10000.0, 16000.0)  # the peaks rise over this distance band
PEAK_HEIGHT = 2400.0
HAZE_DISTANCE = 20000.0  # aerial perspective e-folding distance on the far terrain
TREELINE = 1300.0
SNOWLINE = 1700.0
SUN_GAP = (20.0, 45.0)  # degrees either side of the sun azimuth: no peaks inside, full beyond
ROAD_HALF_WIDTH = 1.3
ROAD_BLEND = 3.5
ROAD_SINK = 0.06  # terrain under the road sits this far below the road surface
ROAD_STEP = 0.25
GRASS_DENSITY = 160.0  # clumps per square metre at the road verge
GRASS_CLUMPS = 6  # clump variants

# Wind: gusts are noise blobs drifting downwind; grass clumps and trees lean downwind with the
# gust strength and sway around that lean.
WIND_HEADING = 120.0  # degrees from +X towards +Y the wind blows to
GUST_SPEED = 6.0  # m/s
GUST_SIZE = 18.0  # metres
WIND_CALM = 0.2  # lean between gusts, as a fraction of the lean in a full gust
GRASS_LEAN = 0.35  # radians a grass clump leans in a full gust
TREE_LEAN = 0.02  # radians a fir leans in a full gust: 0.24 m at the top of a 12 m fir

# Scooter local frame: +X forward, +Y left, +Z up, origin on the ground below the deck.
# Modelled on the Segway Ninebot Max G2 (1.26 m tall, 20 cm deck, 10-inch tyres), scaled to the
# cat: SCOOTER_SCALE of the real size puts the handlebar where the cat's paws are.
SCOOTER_SCALE = 0.443
WHEEL_R = 0.127 * SCOOTER_SCALE
REAR_AXLE = Vector((-0.15, 0.0, WHEEL_R))
FRONT_AXLE = Vector((0.33, 0.0, WHEEL_R))
DECK_TOP = 0.20 * SCOOTER_SCALE
DECK_X = (-0.092, 0.22)
FORK_Y = 0.025
FORK_DIR = Vector((-0.027, 0.0, 0.13))  # front axle to fork crown, raked like the stem
SLIDER_TOP = 0.6  # fraction of FORK_DIR covered by the lower fork sliders
SWINGARM_PIVOT = Vector((-0.085, 0.0, 0.068))
SHOCK_Y = 0.032
SHOCK_TOP = Vector((-0.12, 0.0, 0.13))
# The cat is modelled standing on a deck 0.117 m high; its paws grip the bar at BAR_DESIGN,
# under the chin, so the head clears the bar when the cat leans into a climb.
CAT_OFFSET = Vector((0.0, 0.0, DECK_TOP - 0.117))
BAR_DESIGN = Vector((0.25, 0.0, 0.465))
MAX_LEAN = 0.25  # radians the cat leans forward into a climb; more brings the chin to the bar
BAR = BAR_DESIGN + CAT_OFFSET
BAR_HALF_WIDTH = 0.14
LANE_OFFSET = -0.45  # metres left of the centre line (negative: right lane)
BASE_SPEED = 4.5  # m/s
RIDE_START_X = -22.0

# Speed bumps, as distances along the road from the ride start (m).
BUMPS = [8.0, 27.0, 46.0]
BUMP_HEIGHT = 0.04
BUMP_LENGTH = 0.35
ROAD_ROUGHNESS = 0.003

# Suspension: two springs on a body with heave and pitch.
SPRING_HZ = 3.0
SPRING_DAMPING = 0.25  # damping ratio
SUBSTEPS = 10

CAT_PIVOT = Vector((-0.04, 0.0, 0.117))  # in the cat's modelling frame; see CAT_OFFSET
NECK = Vector((0.09, 0.0, 0.52))
TAIL_BASE = Vector((-0.13, 0.0, 0.21))

SUN_ELEVATION = 4.0
SUN_AZIMUTH = 15.0  # degrees from +X towards +Y; the ride starts heading +X, into the sunset
HAZE_DENSITY = 0.006  # at the haze base; falls off with height
HAZE_HEIGHT = 20.0  # metres per e-fold of the density
DOF_FSTOP = 2.8
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


def away_from_sun(x, y):
    """0 in the direction of the setting sun, 1 beyond SUN_GAP degrees either side of it."""
    gap = abs(wrap_angle(math.atan2(y, x) - math.radians(SUN_AZIMUTH)))
    return smoothstep(math.radians(SUN_GAP[0]), math.radians(SUN_GAP[1]), gap)


def base_height(x, y):
    """Rolling ground along the ride, rising into foothills that ring the valley. The foothills
    stay low towards the sunset and along the road, which runs through a notch."""
    h = 12.0 * fbm(x / 35.0, y / 35.0)
    ring = smoothstep(55.0, 125.0, math.hypot(x, y)) * smoothstep(8.0, 35.0, abs(y - centre_y(x)))
    height = lerp(10.0, 30.0, away_from_sun(x, y))
    return h + ring * height * (0.75 + 0.35 * fbm(x / 60.0 + 5.0, y / 60.0 + 5.0, octaves=3))


def far_height(x, y):
    """Foothills rising away from the near terrain into an alpine range, with a low gap where
    the sun sets. ridged_multi_fractal spans about 0.15-1.9 (median 1.16)."""
    r = math.hypot(x, y)
    open_side = away_from_sun(x, y)
    foothills = 30.0 + 600.0 * smoothstep(250.0, 8000.0, r) * (0.3 + 0.7 * open_side) \
        * (0.6 + 0.4 * fbm(x / 2500.0, y / 2500.0, octaves=4, z=11.1))
    ridges = noise.ridged_multi_fractal(Vector((x / 6000.0, y / 6000.0, 0.5)), 1.0, 2.0, 8, 1.0, 2.0,
                                        noise_basis='PERLIN_ORIGINAL')
    peaks = min(max((ridges - 0.6) / 1.2, 0.0), 1.0) ** 1.5
    return foothills + PEAK_HEIGHT * open_side * smoothstep(*RANGE_START, r) * peaks


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

    def arc_band(self, centre, radius, width, thickness, a0, a1, mat=0, steps=20):
        """Curved strip around the Y axis through centre, from angle a0 to a1 (0 = +X, 90 = +Z):
        a mudguard."""
        rings = []
        for i in range(steps + 1):
            a = math.radians(lerp(a0, a1, i / steps))
            out = Vector((math.cos(a), 0.0, math.sin(a)))
            rings.append([self.bm.verts.new(self.to_local @ (Vector(centre) + out * r + Vector((0, y, 0))))
                          for r, y in ((radius, -width / 2), (radius, width / 2),
                                       (radius + thickness, width / 2), (radius + thickness, -width / 2))])
        for r0, r1 in zip(rings, rings[1:]):
            for k in range(4):
                self.bm.faces.new((r0[k], r0[(k + 1) % 4], r1[(k + 1) % 4], r1[k]))
        self.bm.faces.new(rings[0])
        self.bm.faces.new(list(reversed(rings[-1])))
        self._finish([v for r in rings for v in r], mat, True)

    def bellows(self, radius, length, folds, mat=0):
        """Rubber gaiter along +Z from the origin: overlapping rings over a core tube."""
        pitch = length / folds
        for k in range(folds):
            self.torus(T(0, 0, (k + 0.5) * pitch), radius, pitch * 0.55, mat, seg_major=16, seg_minor=8)
        self.cylinder(T(0, 0, length / 2), radius * 0.9, length, mat, segments=16)

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


def build_materials():
    m = {
        "terrain": terrain_material(),
        "asphalt": asphalt_material(),
        "alpine": alpine_material(),
        "paint": make_material("RoadPaint", (0.85, 0.82, 0.70), 0.6),
        "wood": make_material("FenceWood", (0.20, 0.12, 0.06), 0.8),
        "wire": make_material("FenceWire", (0.4, 0.4, 0.42), 0.4, metallic=1.0),
        "blades": grass_blade_material(),
        "rock": textured_material("Rock", (0.08, 0.075, 0.07), (0.16, 0.15, 0.13), 3.0, 12.0, 0.8, 0.4),
        "frame": make_material("ScooterPaint", (0.045, 0.047, 0.05), 0.35, metallic=0.6, inputs={"Coat Weight": 0.5}),
        "plastic": make_material("BlackPlastic", (0.02, 0.02, 0.021), 0.5),
        "amber": make_material("Indicator", (0.9, 0.45, 0.02), 0.2, emission=(1.0, 0.45, 0.02), strength=0.3),
        "reflector": make_material("Reflector", (0.9, 0.35, 0.02), 0.15),
        "rubber": make_material("Rubber", (0.015, 0.015, 0.015), 0.85),
        "alu": make_material("Aluminium", (0.65, 0.66, 0.68), 0.3, metallic=1.0),
        "headlight": make_material("Headlight", (1.0, 0.95, 0.85), 0.2, emission=(1.0, 0.95, 0.85), strength=30.0),
        "brakelight": make_material("BrakeLight", (0.8, 0.02, 0.02), 0.3, emission=(1.0, 0.03, 0.02), strength=12.0),
        "underglow": make_material("Underglow", (0.1, 0.8, 1.0), 0.3, emission=(0.1, 0.8, 1.0), strength=12.0),
        "screen": make_material("Display", (0.05, 0.4, 0.6), 0.2, emission=(0.1, 0.6, 1.0), strength=2.0),
        "fur": make_material("BlackFur", (0.008, 0.008, 0.009), 0.9),  # skin under the fur strands
        "eye": make_material("CatEye", (0.75, 0.62, 0.04), 0.05, emission=(0.75, 0.62, 0.04), strength=0.4),
        "pupil": make_material("Pupil", (0.0, 0.0, 0.0), 0.05),
        "nose": make_material("CatNose", (0.10, 0.04, 0.05), 0.4),
        "inner_ear": make_material("InnerEar", (0.12, 0.05, 0.06), 0.6),
        "whisker": make_material("Whisker", (0.75, 0.75, 0.72), 0.4),
        "collar": make_material("Collar", (0.7, 0.03, 0.05), 0.5),
        "gold": make_material("Bell", (1.0, 0.75, 0.3), 0.2, metallic=1.0),
        "bump_yellow": make_material("BumpYellow", (0.9, 0.65, 0.02), 0.6),
        "spring": make_material("Spring", (0.12, 0.12, 0.13), 0.3, metallic=1.0),
        "strands": strand_material(),
    }
    return m


def load_image(asset, name):
    img = bpy.data.images.load(str(TEXTURES / asset / f"{name}.jpg"), check_existing=True)
    if name != "diff":
        img.colorspace_settings.name = 'Non-Color'
    return img


def texture_set(nt, asset, size, uv):
    """Diffuse, roughness and normal-map colour sockets of a Poly Haven texture repeating every
    size metres; uv is in metres."""
    mapping = nt.nodes.new('ShaderNodeMapping')
    mapping.inputs['Scale'].default_value = (1.0 / size,) * 3
    nt.links.new(uv, mapping.inputs['Vector'])
    sockets = []
    for name in ("diff", "rough", "nor_gl"):
        tex = nt.nodes.new('ShaderNodeTexImage')
        tex.image = load_image(asset, name)
        nt.links.new(mapping.outputs['Vector'], tex.inputs['Vector'])
        sockets.append(tex.outputs['Color'])
    return sockets


def smooth_range(nt, value, lo, hi):
    """0 below lo, 1 above hi, smoothstep in between (reversed when lo > hi)."""
    node = nt.nodes.new('ShaderNodeMapRange')
    node.interpolation_type = 'SMOOTHSTEP'
    node.inputs['From Min'].default_value = lo
    node.inputs['From Max'].default_value = hi
    nt.links.new(value, node.inputs['Value'])
    return node.outputs['Result']


def apply_texture(nt, bsdf, sockets):
    colour, rough, normal = sockets
    nt.links.new(colour, bsdf.inputs['Base Color'])
    nt.links.new(rough, bsdf.inputs['Roughness'])
    normal_map = nt.nodes.new('ShaderNodeNormalMap')
    normal_map.uv_map = "UVMap"
    nt.links.new(normal, normal_map.inputs['Color'])
    nt.links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])


def terrain_material():
    """Grass ground, with dry ground on crests, rock on steep slopes, and darker hollows.
    Crests and hollows come from the mesh attribute 'curvature' (positive on crests)."""
    mat = make_material("Terrain", (0.1, 0.12, 0.05), 0.9)
    nt = mat.node_tree
    bsdf = principled_node(mat)
    uv = nt.nodes.new('ShaderNodeUVMap')
    uv.uv_map = "UVMap"
    grass = texture_set(nt, "grass_ground", 2.5, uv.outputs['UV'])
    dry = texture_set(nt, "dry_ground_01", 4.0, uv.outputs['UV'])
    rock = texture_set(nt, "aerial_rocks_02", 20.0, uv.outputs['UV'])
    # grass_ground averages olive-brown (0.43, 0.38, 0.24); shift it towards green.
    greener = nt.nodes.new('ShaderNodeHueSaturation')
    greener.inputs['Hue'].default_value = 0.56
    greener.inputs['Saturation'].default_value = 1.3
    nt.links.new(grass[0], greener.inputs['Color'])
    grass[0] = greener.outputs['Color']

    coord = nt.nodes.new('ShaderNodeTexCoord')
    breakup = nt.nodes.new('ShaderNodeTexNoise')
    breakup.inputs['Scale'].default_value = 0.12
    breakup.inputs['Detail'].default_value = 6.0
    nt.links.new(coord.outputs['Object'], breakup.inputs['Vector'])
    jitter = math_node(nt, 'SUBTRACT', breakup.outputs['Fac'], 0.5)

    geometry = nt.nodes.new('ShaderNodeNewGeometry')
    normal = nt.nodes.new('ShaderNodeSeparateXYZ')
    nt.links.new(geometry.outputs['Normal'], normal.inputs[0])
    steep = math_node(nt, 'SUBTRACT', 1.0, normal.outputs['Z'])
    rock_mask = smooth_range(nt, math_node(nt, 'ADD', steep, math_node(nt, 'MULTIPLY', jitter, 0.2)), 0.25, 0.38)

    curvature = nt.nodes.new('ShaderNodeAttribute')
    curvature.attribute_name = "curvature"
    # Curvature is about ±0.5 at the 5th/95th percentile.
    crest = math_node(nt, 'ADD', curvature.outputs['Fac'], math_node(nt, 'MULTIPLY', jitter, 0.6))
    dry_mask = smooth_range(nt, crest, 0.45, 0.9)
    hollow = smooth_range(nt, curvature.outputs['Fac'], -0.7, -0.2)

    layers = []
    for g, d, r in zip(grass, dry, rock):
        layers.append(mix_colour(nt, rock_mask, mix_colour(nt, dry_mask, g, d), r))
    shade = mix_colour(nt, hollow, (0.6, 0.6, 0.6), (1.0, 1.0, 1.0))
    macro = nt.nodes.new('ShaderNodeTexNoise')
    macro.inputs['Scale'].default_value = 0.02
    nt.links.new(coord.outputs['Object'], macro.inputs['Vector'])
    macro_shade = nt.nodes.new('ShaderNodeValToRGB')
    ramp_colors(macro_shade, (0.75, 0.75, 0.7), (1.15, 1.15, 1.1), (0.3, 0.7))
    nt.links.new(macro.outputs['Fac'], macro_shade.inputs['Fac'])
    layers[0] = mix_colour(nt, 1.0, mix_colour(nt, 1.0, layers[0], shade, 'MULTIPLY'),
                           macro_shade.outputs['Color'], 'MULTIPLY')
    apply_texture(nt, bsdf, layers)
    return mat


def alpine_material():
    """Far terrain: forest below the treeline, rock above it and on steep faces, snow above the
    snowline on gentler slopes. Fades to a haze colour with distance (aerial perspective), as
    the haze volume only reaches a few hundred metres."""
    mat = make_material("Alpine", (0.17, 0.15, 0.14), 0.85)
    nt = mat.node_tree
    bsdf = principled_node(mat)
    out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
    coord = nt.nodes.new('ShaderNodeTexCoord')
    position = nt.nodes.new('ShaderNodeSeparateXYZ')
    nt.links.new(coord.outputs['Object'], position.inputs[0])
    geometry = nt.nodes.new('ShaderNodeNewGeometry')
    normal = nt.nodes.new('ShaderNodeSeparateXYZ')
    nt.links.new(geometry.outputs['Normal'], normal.inputs[0])
    patches = nt.nodes.new('ShaderNodeTexNoise')
    patches.inputs['Scale'].default_value = 0.002
    patches.inputs['Detail'].default_value = 8.0
    nt.links.new(coord.outputs['Object'], patches.inputs['Vector'])
    jitter = math_node(nt, 'MULTIPLY', math_node(nt, 'SUBTRACT', patches.outputs['Fac'], 0.5), 400.0)
    height = math_node(nt, 'ADD', position.outputs['Z'], jitter)

    flat = smooth_range(nt, normal.outputs['Z'], 0.55, 0.75)
    forest = math_node(nt, 'MULTIPLY', smooth_range(nt, height, TREELINE + 150.0, TREELINE - 150.0), flat)
    snow = math_node(nt, 'MULTIPLY', smooth_range(nt, height, SNOWLINE - 100.0, SNOWLINE + 100.0),
                     smooth_range(nt, normal.outputs['Z'], 0.45, 0.65))
    rock = mix_colour(nt, patches.outputs['Fac'], (0.10, 0.09, 0.085), (0.24, 0.21, 0.19))
    colour = mix_colour(nt, snow, mix_colour(nt, forest, rock, (0.02, 0.04, 0.02)), (0.85, 0.87, 0.9))
    nt.links.new(colour, bsdf.inputs['Base Color'])
    nt.links.new(mix_colour(nt, snow, (0.85, 0.85, 0.85), (0.45, 0.45, 0.45)), bsdf.inputs['Roughness'])

    camera = nt.nodes.new('ShaderNodeCameraData')
    falloff = math_node(nt, 'EXPONENT', math_node(nt, 'DIVIDE', camera.outputs['View Distance'], -HAZE_DISTANCE), 0.0)
    haze = math_node(nt, 'MINIMUM', math_node(nt, 'SUBTRACT', 1.0, falloff), 0.85)
    glow = nt.nodes.new('ShaderNodeEmission')
    glow.inputs['Color'].default_value = (0.5, 0.55, 0.72, 1.0)
    glow.inputs['Strength'].default_value = 0.25
    mix = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(haze, mix.inputs['Fac'])
    nt.links.new(bsdf.outputs['BSDF'], mix.inputs[1])
    nt.links.new(glow.outputs['Emission'], mix.inputs[2])
    nt.links.new(mix.outputs['Shader'], out.inputs['Surface'])
    return mat


def asphalt_material():
    """Asphalt texture; the road UVs run along the road in metres."""
    mat = make_material("Asphalt", (0.04, 0.04, 0.04), 0.9)
    nt = mat.node_tree
    uv = nt.nodes.new('ShaderNodeUVMap')
    uv.uv_map = "UVMap"
    apply_texture(nt, principled_node(mat), texture_set(nt, "asphalt_02", 3.0, uv.outputs['UV']))
    return mat


def grass_blade_material():
    """Dark at the root, lighter and yellower at the tip (UV v), varied per clump instance,
    and partly translucent for backlight from the low sun."""
    mat = make_material("GrassBlades", (0.07, 0.16, 0.03), 0.6)
    nt = mat.node_tree
    bsdf = principled_node(mat)
    out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
    uv = nt.nodes.new('ShaderNodeUVMap')
    split = nt.nodes.new('ShaderNodeSeparateXYZ')
    nt.links.new(uv.outputs['UV'], split.inputs[0])
    info = nt.nodes.new('ShaderNodeObjectInfo')
    tip = mix_colour(nt, info.outputs['Random'], (0.10, 0.20, 0.03), (0.30, 0.28, 0.08))
    colour = mix_colour(nt, smooth_range(nt, split.outputs['Y'], 0.0, 0.8), (0.015, 0.035, 0.008), tip)
    nt.links.new(colour, bsdf.inputs['Base Color'])
    translucent = nt.nodes.new('ShaderNodeBsdfTranslucent')
    nt.links.new(colour, translucent.inputs['Color'])
    mix = nt.nodes.new('ShaderNodeMixShader')
    mix.inputs['Fac'].default_value = 0.35
    nt.links.new(bsdf.outputs['BSDF'], mix.inputs[1])
    nt.links.new(translucent.outputs['BSDF'], mix.inputs[2])
    nt.links.new(mix.outputs['Shader'], out.inputs['Surface'])
    return mat


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


def wind_direction():
    heading = math.radians(WIND_HEADING)
    return (math.cos(heading), math.sin(heading), 0.0)


def wind_strength(nt, position, phase, sway_hz):
    """Geometry nodes field: the wind at position as a fraction of the lean in a full gust,
    swaying at sway_hz with the given phase (radians)."""
    seconds = nt.nodes.new('GeometryNodeInputSceneTime').outputs['Seconds']

    # Gusts drift downwind at GUST_SPEED and slowly change shape.
    drift = nt.nodes.new('ShaderNodeVectorMath')
    drift.operation = 'SCALE'
    drift.inputs[0].default_value = wind_direction()
    nt.links.new(math_node(nt, 'MULTIPLY', seconds, -GUST_SPEED), drift.inputs['Scale'])
    shifted = nt.nodes.new('ShaderNodeVectorMath')
    shifted.operation = 'ADD'
    nt.links.new(position, shifted.inputs[0])
    nt.links.new(drift.outputs['Vector'], shifted.inputs[1])
    gusts = nt.nodes.new('ShaderNodeTexNoise')
    gusts.noise_dimensions = '4D'
    gusts.inputs['Scale'].default_value = 1.0 / GUST_SIZE
    gusts.inputs['Detail'].default_value = 1.0
    nt.links.new(shifted.outputs['Vector'], gusts.inputs['Vector'])
    nt.links.new(math_node(nt, 'MULTIPLY', seconds, 0.1), gusts.inputs['W'])
    gust = smooth_range(nt, gusts.outputs['Fac'], 0.42, 0.68)

    cycles = math_node(nt, 'MULTIPLY', seconds, 2 * math.pi * sway_hz)
    sway = math_node(nt, 'SINE', math_node(nt, 'ADD', cycles, phase), 0.0)
    return math_node(nt, 'MULTIPLY', math_node(nt, 'ADD', gust, WIND_CALM),
                     math_node(nt, 'ADD', math_node(nt, 'MULTIPLY', sway, 0.4), 1.0))


def lean_downwind(nt, rotation, angle):
    """Rotation socket: rotation (an Euler vector socket) followed by a lean of angle radians
    downwind, about the instance origin."""
    tilt = nt.nodes.new('FunctionNodeAxisAngleToRotation')
    dx, dy, _ = wind_direction()
    tilt.inputs['Axis'].default_value = (-dy, dx, 0.0)  # up × wind direction: tops go downwind
    nt.links.new(angle, tilt.inputs['Angle'])
    rotate = nt.nodes.new('FunctionNodeRotateRotation')
    rotate.rotation_space = 'GLOBAL'
    nt.links.new(rotation, rotate.inputs['Rotation'])
    nt.links.new(tilt.outputs['Rotation'], rotate.inputs['Rotate By'])
    return rotate.outputs['Rotation']


def build_terrain(road, mats, coll, ride_x):
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
    mesh.materials.append(mats["terrain"])
    set_uvs(mesh, [(x, y) for x, y, _ in verts])

    # Negative Laplacian of the height over a 4-cell stencil: positive on crests.
    k = 4
    curvature = []
    for j in range(n):
        for i in range(n):
            z = verts[j * n + i][2]
            around = [verts[min(max(jj, 0), n - 1) * n + min(max(ii, 0), n - 1)][2]
                      for jj, ii in ((j, i - k), (j, i + k), (j - k, i), (j + k, i))]
            curvature.append(-(sum(around) - 4 * z) / (k * step) ** 2)
    ranked = sorted(curvature)
    print("curvature percentiles 5/50/95:", [round(ranked[int(len(ranked) * p)], 4) for p in (0.05, 0.5, 0.95)])
    set_attribute(mesh, "curvature", curvature)
    set_attribute(mesh, "grass_density", [grass_density(road, x, y, ride_x) for x, y, _ in verts])
    return link(bpy.data.objects.new("Terrain", mesh), coll)


def build_far_terrain(mats, coll):
    """Polar grid from just inside the near terrain's edge out to FAR_RADIUS, with radial
    spacing growing geometrically. Its inner edge follows base_height so the two meshes meet."""
    inner, rings, segments = 120.0, 260, 1200
    verts = []
    for i in range(rings + 1):
        r = inner * (FAR_RADIUS / inner) ** (i / rings)
        blend = smoothstep(inner, 400.0, r)
        for k in range(segments):
            a = 2.0 * math.pi * k / segments
            x, y = r * math.cos(a), r * math.sin(a)
            near = base_height(x, y) - 0.5
            verts.append((x, y, lerp(near, far_height(x, y), blend) if blend > 0 else near))
    faces = [(i * segments + k, i * segments + (k + 1) % segments,
              (i + 1) * segments + (k + 1) % segments, (i + 1) * segments + k)
             for i in range(rings) for k in range(segments)]
    mesh = bpy.data.meshes.new("FarTerrain")
    mesh.from_pydata(verts, [], faces)
    mesh.shade_smooth()
    mesh.materials.append(mats["alpine"])
    print(f"far terrain: highest point {max(v[2] for v in verts):.0f} m")
    return link(bpy.data.objects.new("FarTerrain", mesh), coll)


def set_uvs(mesh, per_vertex):
    layer = mesh.uv_layers.new(name="UVMap")
    index = [0] * len(mesh.loops)
    mesh.loops.foreach_get("vertex_index", index)
    layer.data.foreach_set("uv", [c for i in index for c in per_vertex[i]])


def set_attribute(mesh, name, values):
    attribute = mesh.attributes.new(name, 'FLOAT', 'POINT')
    attribute.data.foreach_set("value", values)


def grass_density(road, x, y, ride_x):
    """Relative grass density: full on the verge, thinning away from the road and outside the
    stretch the camera sees."""
    edge = road.distance(x, y)[1] - ROAD_HALF_WIDTH
    along = smoothstep(ride_x[0] - 18.0, ride_x[0] - 8.0, x) * (1.0 - smoothstep(ride_x[1] + 10.0, ride_x[1] + 20.0, x))
    return along * smoothstep(0.05, 0.3, edge) * lerp(1.0, 0.06, smoothstep(2.0, 20.0, edge))


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
    across = (w + 0.15, w, -w, -w - 0.15)
    set_uvs(mesh, [(s, v) for s in road.cum for v in across])
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


def load_tree_meshes():
    """Meshes of the Poly Haven fir_tree_01 variants a, b and c, as {lod: [mesh, ...]}.
    The model ships its own levels of detail: LOD1 is ~70-180k triangles, LOD2 ~30-70k."""
    path = ROOT / "assets" / "models" / "fir_tree_01" / "fir_tree_01.blend"
    names = {lod: [f"fir_tree_01_{v}_{lod}" for v in "abc"] for lod in ("LOD1", "LOD2")}
    with bpy.data.libraries.load(str(path)) as (src, dst):
        dst.objects = [n for group in names.values() for n in group]
    loaded = {o.name: o.data for o in dst.objects}
    for obj in dst.objects:
        bpy.data.objects.remove(obj)
    # Alpha-tested shadows of the twig cards cost ~50 s per frame on the Radeon 680M and look
    # the same as opaque card shadows at the distances the trees are seen from.
    for mat in {m for mesh in loaded.values() for m in mesh.materials if m and "twig" in m.name}:
        mat.use_transparent_shadow = False
    return {lod: [loaded[n] for n in group] for lod, group in names.items()}


def scatter_trees(road, coll, rng, ride_x):
    """Fir trees in clusters away from the road: full LOD1 detail near the ride, LOD2 further out.
    The trees are instanced by geometry nodes on the vertices of a point mesh, so each one can
    lean with the wind about its base. Displacing the tree vertices in the shader instead made
    the Radeon 680M time out and reset during animation renders."""
    meshes = load_tree_meshes()
    variants = bpy.data.collections.new("TreeVariants")  # not linked to the scene: instanced only
    for k, mesh in enumerate(meshes["LOD1"] + meshes["LOD2"]):
        variants.objects.link(bpy.data.objects.new(f"TreeVariant.{k}", mesh))
    points, variant, yaw, scale = [], [], [], []
    while len(points) < 300:
        x, y = rng.uniform(-120, 120), rng.uniform(-120, 120)
        distance = road.distance(x, y)[1]
        if distance < 6.0 + rng.uniform(0, 4):
            continue
        if noise.noise(Vector((x / 30.0, y / 30.0, 3.3))) < -0.05:
            continue
        near = distance < 30.0 and ride_x[0] - 30.0 < x < ride_x[1] + 30.0
        variant.append(rng.randrange(3) + (0 if near else 3))
        points.append((x, y, road.terrain_height(x, y) - 0.1))
        yaw.append(rng.uniform(0, 2 * math.pi))
        scale.append(rng.uniform(0.35, 0.8))
    mesh = bpy.data.meshes.new("TreePoints")
    mesh.from_pydata(points, [], [])
    for name, values in (("variant", variant), ("yaw", yaw), ("scale", scale)):
        set_attribute(mesh, name, values)

    tree = bpy.data.node_groups.new("TreeScatter", 'GeometryNodeTree')
    tree.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    nodes, links = tree.nodes, tree.links

    def attribute(name):
        node = nodes.new('GeometryNodeInputNamedAttribute')
        node.data_type = 'FLOAT'
        node.inputs['Name'].default_value = name
        return node.outputs['Attribute']

    collection = nodes.new('GeometryNodeCollectionInfo')
    collection.inputs['Collection'].default_value = variants
    collection.inputs['Separate Children'].default_value = True
    collection.inputs['Reset Children'].default_value = True
    heading = nodes.new('ShaderNodeCombineXYZ')
    links.new(attribute("yaw"), heading.inputs['Z'])
    wind = wind_strength(tree, nodes.new('GeometryNodeInputPosition').outputs['Position'],
                         random_value(nodes, 'FLOAT', 0.0, 2 * math.pi, 5), 0.35)
    rotation = lean_downwind(tree, heading.outputs['Vector'], math_node(tree, 'MULTIPLY', wind, TREE_LEAN))
    instances = nodes.new('GeometryNodeInstanceOnPoints')
    instances.inputs['Pick Instance'].default_value = True
    links.new(nodes.new('NodeGroupInput').outputs[0], instances.inputs['Points'])
    links.new(collection.outputs['Instances'], instances.inputs['Instance'])
    links.new(attribute("variant"), instances.inputs['Instance Index'])
    links.new(rotation, instances.inputs['Rotation'])
    links.new(attribute("scale"), instances.inputs['Scale'])
    links.new(instances.outputs['Instances'], nodes.new('NodeGroupOutput').inputs[0])

    trees = link(bpy.data.objects.new("Trees", mesh), coll)
    trees.modifiers.new("Scatter", 'NODES').node_group = tree


def grass_clump(rng, mat):
    """A clump of curved, tapering blades; UV v runs from root (0) to tip (1)."""
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    for _ in range(rng.randint(10, 18)):
        base = Vector((rng.gauss(0, 0.025), rng.gauss(0, 0.025), -0.01))
        height = rng.uniform(0.07, 0.22)
        width = rng.uniform(0.003, 0.006)
        a = rng.uniform(0, 2 * math.pi)
        lean = Vector((math.cos(a), math.sin(a), 0.0))
        side = Vector((0, 0, 1)).cross(lean)
        bend = rng.uniform(0.15, 0.5) * height
        rows = []
        for t in (0.0, 0.35, 0.7):
            centre = base + Vector((0, 0, height * t)) + lean * bend * t * t
            half = width * (1.0 - t) ** 0.7
            rows.append(((bm.verts.new(centre - side * half), bm.verts.new(centre + side * half)), t))
        tip = bm.verts.new(base + Vector((0, 0, height)) + lean * bend)
        for ((l0, r0), t0), ((l1, r1), t1) in zip(rows, rows[1:]):
            face = bm.faces.new((l0, r0, r1, l1))
            for loop, v in zip(face.loops, (t0, t0, t1, t1)):
                loop[uv].uv = (0.5, v)
        (l, r), t = rows[-1]
        face = bm.faces.new((l, r, tip))
        for loop, v in zip(face.loops, (t, t, 1.0)):
            loop[uv].uv = (0.5, v)
    mesh = bpy.data.meshes.new("GrassClump")
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(mat)
    return mesh


def random_value(nodes, kind, lo, hi, seed):
    node = nodes.new('FunctionNodeRandomValue')
    node.data_type = kind
    sock = 'INT' if kind == 'INT' else 'VALUE'
    socket(node.inputs, "Min", sock).default_value = lo
    socket(node.inputs, "Max", sock).default_value = hi
    node.inputs['Seed'].default_value = seed
    return socket(node.outputs, "Value", sock)


def build_grass(mats, coll, terrain, rng):
    """Grass clumps instanced by geometry nodes over the terrain, with the density taken from
    the terrain attribute 'grass_density'."""
    clumps = bpy.data.collections.new("GrassClumps")  # not linked to the scene: instanced only
    for k in range(GRASS_CLUMPS):
        clumps.objects.link(bpy.data.objects.new(f"GrassClump.{k}", grass_clump(rng, mats["blades"])))

    tree = bpy.data.node_groups.new("GrassScatter", 'GeometryNodeTree')
    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    nodes, links = tree.nodes, tree.links
    source = nodes.new('GeometryNodeObjectInfo')
    source.inputs['Object'].default_value = terrain
    density = nodes.new('GeometryNodeInputNamedAttribute')
    density.data_type = 'FLOAT'
    density.inputs['Name'].default_value = "grass_density"
    scaled = nodes.new('ShaderNodeMath')
    scaled.operation = 'MULTIPLY'
    scaled.inputs[1].default_value = GRASS_DENSITY
    links.new(density.outputs['Attribute'], scaled.inputs[0])
    points = nodes.new('GeometryNodeDistributePointsOnFaces')
    points.distribute_method = 'RANDOM'
    points.inputs['Seed'].default_value = SEED
    links.new(source.outputs['Geometry'], points.inputs['Mesh'])
    links.new(scaled.outputs[0], points.inputs['Density'])
    variants = nodes.new('GeometryNodeCollectionInfo')
    variants.inputs['Collection'].default_value = clumps
    variants.inputs['Separate Children'].default_value = True
    variants.inputs['Reset Children'].default_value = True
    yaw = nodes.new('ShaderNodeCombineXYZ')
    links.new(random_value(nodes, 'FLOAT', 0.0, 2 * math.pi, 1), yaw.inputs['Z'])
    # Each clump tilts downwind about its root, by the wind at its root. Tilting whole clumps
    # here costs one evaluation per clump; displacing the 53M blade vertices in the shader
    # stalled the GPU until the driver reset it.
    wind = wind_strength(tree, nodes.new('GeometryNodeInputPosition').outputs['Position'],
                         random_value(nodes, 'FLOAT', 0.0, 2 * math.pi, 4), 1.2)
    rotation = lean_downwind(tree, yaw.outputs['Vector'], math_node(tree, 'MULTIPLY', wind, GRASS_LEAN))
    instances = nodes.new('GeometryNodeInstanceOnPoints')
    instances.inputs['Pick Instance'].default_value = True
    links.new(points.outputs['Points'], instances.inputs['Points'])
    links.new(variants.outputs['Instances'], instances.inputs['Instance'])
    links.new(random_value(nodes, 'INT', 0, GRASS_CLUMPS - 1, 2), instances.inputs['Instance Index'])
    links.new(rotation, instances.inputs['Rotation'])
    links.new(random_value(nodes, 'FLOAT', 0.6, 1.4, 3), instances.inputs['Scale'])
    out = nodes.new('NodeGroupOutput')
    links.new(instances.outputs['Instances'], out.inputs[0])

    grass = link(bpy.data.objects.new("Grass", bpy.data.meshes.new("Grass")), coll)
    grass.modifiers.new("Scatter", 'NODES').node_group = tree


def scatter_rocks(road, mats, coll, rng, ride_x):
    """Rocks near the stretch of road the ride covers, merged into one mesh."""
    lo, hi = ride_x[0] - 10.0, ride_x[1] + 15.0
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


def build_haze(coll):
    """Aerial perspective: a box of forward-scattering volume over the terrain, thinning with
    height. A world volume does not work in EEVEE: it is infinitely deep and blocks the sun."""
    half, bottom, top = TERRAIN_SIZE / 2 - 5.0, -15.0, 80.0
    b = MeshBuilder()
    b.box((0, 0, (bottom + top) / 2), (2 * half, 2 * half, top - bottom))
    mat = bpy.data.materials.new("Haze")
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    for node in [n for n in nt.nodes if n.type != 'OUTPUT_MATERIAL']:
        nt.nodes.remove(node)
    out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
    volume = nt.nodes.new('ShaderNodeVolumePrincipled')
    volume.inputs['Color'].default_value = (1.0, 0.88, 0.8, 1.0)
    volume.inputs['Anisotropy'].default_value = 0.6
    coord = nt.nodes.new('ShaderNodeTexCoord')
    xyz = nt.nodes.new('ShaderNodeSeparateXYZ')
    nt.links.new(coord.outputs['Object'], xyz.inputs[0])
    falloff = math_node(nt, 'EXPONENT', math_node(nt, 'DIVIDE', xyz.outputs['Z'], -HAZE_HEIGHT), 0.0)
    nt.links.new(math_node(nt, 'MULTIPLY', falloff, HAZE_DENSITY), volume.inputs['Density'])
    nt.links.new(volume.outputs['Volume'], out.inputs['Volume'])
    haze = b.to_object("Haze", [mat], coll)
    haze.display_type = 'BOUNDS'


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

    lamp("HeadlightBeam", 'SPOT', BAR + Vector((0.031, 0, -0.004)), 6.0, (1.0, 0.92, 0.8), (1.0, 0.0, -0.25),
         spot_size=math.radians(55), spot_blend=0.6, shadow_soft_size=0.005)
    lamp("TailLight", 'POINT', (-0.228, 0, 0.083), 0.15, (1.0, 0.05, 0.03), shadow_soft_size=0.005)
    lamp("Underglow", 'AREA', ((DECK_X[0] + DECK_X[1]) / 2, 0, DECK_TOP - 0.036), 0.5, (0.1, 0.8, 1.0),
         shape='RECTANGLE', size=0.26, size_y=0.06)


def build_wheel(name, axle, carrier, coll, materials, hub_radius, hub_width, hub_mat):
    """10-inch pneumatic tyre with tread blocks on a five double-spoke rim. The front hub holds the
    drum brake, the rear hub the motor."""
    RUBBER, SPRING = 1, 6
    w = MeshBuilder(axle)
    across = Matrix.Translation(axle) @ R(math.pi / 2, 'X')
    w.torus(across @ S(1, 1, 1.1), WHEEL_R - 0.013, 0.013, RUBBER, seg_major=48, seg_minor=12)
    for k in range(44):
        a = 2 * math.pi * k / 44
        out = Vector((math.cos(a), 0, math.sin(a)))
        w.box(axle + out * (WHEEL_R - 0.0012) + Vector((0, 0.005 * (-1) ** k, 0)), (0.004, 0.02, 0.003), RUBBER,
              rot=R(-(a + math.pi / 2), 'Y'))
    w.torus(across, 0.031, 0.0035, SPRING, seg_major=40, seg_minor=8)
    for k in range(5):
        a = 2 * math.pi * k / 5
        out = Vector((math.cos(a), 0, math.sin(a)))
        for y in (-0.004, 0.004):
            w.box(axle + out * 0.022 + Vector((0, y, 0)), (0.02, 0.004, 0.005), SPRING, rot=R(-a, 'Y'))
    w.cylinder(across, hub_radius, hub_width, hub_mat, segments=24)
    w.cylinder(across, 0.004, hub_width + 0.012, 2, segments=12)
    return w.to_object(name, materials, coll, carrier)


def build_brake_cable(mats, coll, body, crown, stem):
    """Cable from the left brake lever, looping in front of the stem, down to the front drum."""
    cu = bpy.data.curves.new("BrakeCable", 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = 0.0016
    cu.bevel_resolution = 2
    spline = cu.splines.new('NURBS')
    points = [BAR + Vector((0.016, 0.075, 0.0)), BAR + Vector((0.05, 0.05, -0.04)),
              crown + stem * 0.16 + Vector((0.02, 0.02, 0)), crown + Vector((0.016, 0.02, -0.005)),
              FRONT_AXLE + FORK_DIR * 0.62 + Vector((0.008, FORK_Y + 0.006, 0))]
    spline.points.add(len(points) - 1)
    for point, co in zip(spline.points, points):
        point.co = (*co, 1.0)
    spline.order_u = 3
    spline.use_endpoint_u = True
    cu.materials.append(mats["plastic"])
    link(bpy.data.objects.new("BrakeCable", cu), coll, body)


def build_scooter(mats, coll):
    rig = make_empty("ScooterRig", coll, size=0.3)
    rig.rotation_mode = 'QUATERNION'
    body = make_empty("ScooterBody", coll, rig, size=0.2)
    materials = [mats[k] for k in ("frame", "rubber", "alu", "headlight", "brakelight", "screen", "spring",
                                   "underglow", "plastic", "amber", "reflector")]
    FRAME, RUBBER, ALU, HEAD, BRAKE, SCREEN, SPRING, GLOW, PLASTIC, AMBER, REFLECTOR = range(11)
    crown = FRONT_AXLE + FORK_DIR
    stem = (BAR - crown).normalized()
    deck_mid, deck_len = (DECK_X[0] + DECK_X[1]) / 2, DECK_X[1] - DECK_X[0]

    b = MeshBuilder()
    # Deck: a slab with rounded sides, a ribbed grip mat, the battery cover underneath, LED strips.
    b.box((deck_mid, 0, DECK_TOP - 0.015), (deck_len, 0.05, 0.03), FRAME)
    for y in (-0.025, 0.025):
        b.rod((DECK_X[0], y, DECK_TOP - 0.015), (DECK_X[1], y, DECK_TOP - 0.015), 0.0148, FRAME, segments=16)
    b.box((deck_mid, 0, DECK_TOP + 0.0005), (deck_len - 0.02, 0.066, 0.002), RUBBER)
    for k in range(14):
        b.box((DECK_X[0] + 0.02 + k * (deck_len - 0.04) / 13, 0, DECK_TOP + 0.0018), (0.004, 0.058, 0.0012), RUBBER)
    b.box((deck_mid, 0, DECK_TOP - 0.032), (deck_len - 0.04, 0.055, 0.008), PLASTIC)
    for y in (-0.04, 0.04):
        b.box((deck_mid, y, DECK_TOP - 0.026), (deck_len - 0.04, 0.003, 0.004), GLOW)
    # Neck up to the steering head, with the fork crown and the upper fork tubes.
    b.limb((DECK_X[1] - 0.02, 0, DECK_TOP - 0.012), crown - Vector((0, 0, 0.012)), 0.018, mat=FRAME)
    b.box(crown, (0.024, 2 * FORK_Y + 0.018, 0.012), FRAME)
    for y in (-FORK_Y, FORK_Y):
        b.rod(FRONT_AXLE + FORK_DIR * 0.2 + Vector((0, y, 0)), crown + Vector((0, y, 0)), 0.0045, ALU)
    # Stem with the folding joint and its lever.
    b.rod(crown, BAR - stem * 0.012, 0.0115, FRAME, segments=20)
    hinge = Matrix.Translation(crown + stem * 0.035) @ align_z(stem)
    b.cylinder(hinge, 0.017, 0.035, FRAME, segments=20)
    b.box(crown + stem * 0.035 + Vector((0, -0.019, 0)), (0.006, 0.005, 0.034), PLASTIC, rot=align_z(stem))
    # Handlebar: dashboard with display, headlight, grips, indicators, brake lever, thumb throttle.
    b.rod(BAR + Vector((0, -BAR_HALF_WIDTH, 0)), BAR + Vector((0, BAR_HALF_WIDTH, 0)), 0.0055, FRAME)
    b.box(BAR + Vector((0.006, 0, 0.006)), (0.036, 0.055, 0.02), PLASTIC)
    b.box(BAR + Vector((0.0, 0, 0.0165)), (0.02, 0.026, 0.001), SCREEN, rot=R(math.radians(-20), 'Y'))
    b.cylinder(T(*(BAR + Vector((0.025, 0, -0.004)))) @ R(math.pi / 2, 'Y'), 0.009, 0.01, PLASTIC, segments=20)
    b.cylinder(T(*(BAR + Vector((0.0305, 0, -0.004)))) @ R(math.pi / 2, 'Y'), 0.0075, 0.002, HEAD, segments=20)
    for s in (-1, 1):
        b.rod(BAR + Vector((0, s * 0.09, 0)), BAR + Vector((0, s * BAR_HALF_WIDTH, 0)), 0.0085, RUBBER)
        b.rod(BAR + Vector((0, s * BAR_HALF_WIDTH, 0)), BAR + Vector((0, s * (BAR_HALF_WIDTH + 0.008), 0)), 0.0088, AMBER)
    b.box(BAR + Vector((0.012, 0.075, 0.004)), (0.012, 0.01, 0.01), PLASTIC)
    b.box(BAR + Vector((0.022, 0.105, -0.002)), (0.004, 0.05, 0.004), ALU)
    b.box(BAR + Vector((0.004, -0.08, 0.012)), (0.012, 0.008, 0.012), PLASTIC)
    # Rear: struts from the deck to the shock mounts, the mudguard with the tail light, kickstand.
    for s in (-1, 1):
        b.rod((DECK_X[0] + 0.005, s * 0.028, DECK_TOP - 0.01), SHOCK_TOP + Vector((0, s * 0.028, -0.004)), 0.004, FRAME)
        b.box(SHOCK_TOP + Vector((0, s * 0.03, 0)), (0.012, 0.016, 0.01), FRAME)
    b.arc_band(REAR_AXLE, WHEEL_R + 0.015, 0.044, 0.0025, 55, 185, FRAME)
    b.box((-0.226, 0, 0.083), (0.004, 0.026, 0.009), BRAKE)
    b.rod((0.02, 0.04, DECK_TOP - 0.03), (-0.05, 0.05, DECK_TOP - 0.045), 0.0035, FRAME)
    b.to_object("ScooterFrame", materials, coll, body)
    build_brake_cable(mats, coll, body, crown, stem)

    front = make_empty("FrontSuspension", coll, rig, FRONT_AXLE, size=0.05)
    rear = make_empty("RearSuspension", coll, rig, REAR_AXLE, size=0.05)
    wheels = [build_wheel("WheelFront", FRONT_AXLE, front, coll, materials, 0.016, 0.024, ALU),
              build_wheel("WheelRear", REAR_AXLE, rear, coll, materials, 0.024, 0.03, PLASTIC)]

    lower = MeshBuilder(FRONT_AXLE)
    for s in (-1, 1):
        offset = Vector((0, s * FORK_Y, 0))
        lower.rod(FRONT_AXLE + offset, FRONT_AXLE + FORK_DIR * SLIDER_TOP + offset, 0.0068, PLASTIC)
        lower.box(FRONT_AXLE + FORK_DIR * 0.35 + Vector((0, s * (FORK_Y + 0.0075), 0)), (0.012, 0.002, 0.006), REFLECTOR)
    lower.arc_band(FRONT_AXLE, WHEEL_R + 0.007, 0.034, 0.002, 25, 150, PLASTIC)
    lower.to_object("ForkLower", materials, coll, front)

    a = MeshBuilder(SWINGARM_PIVOT)
    for s in (-1, 1):
        a.rod(SWINGARM_PIVOT + Vector((0, s * 0.025, 0)), REAR_AXLE + Vector((0, s * 0.025, 0)), 0.005, FRAME)
    arm = a.to_object("RearArm", materials, coll, body, SWINGARM_PIVOT)
    arm.rotation_mode = 'QUATERNION'

    # Front: rubber gaiters over the hydraulic fork. Rear: coil-over shocks beside the wheel.
    springs = []
    names = ("FrontGaiterL", "FrontGaiterR", "RearShockL", "RearShockR")
    for i, (bottom, top) in enumerate(spring_ends(0.0, 0.0, 0.0, 0.0)):
        length = (top - bottom).length
        s = MeshBuilder()
        if i < 2:
            s.bellows(0.0068, length, 6, RUBBER)
        else:
            s.coil(0.0075, 0.0015, length, 7, SPRING)
            s.cylinder(T(0, 0, length / 2), 0.0035, length, ALU, segments=12)
        spring = s.to_object(names[i], materials, coll, rig, bottom)
        spring.rotation_mode = 'QUATERNION'
        spring["rest_length"] = length
        springs.append(spring)

    build_scooter_lights(coll, body)
    target = make_empty("CameraTarget", coll, rig, (0.03, 0.0, 0.33))
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


def segment(b, p0, p1, width, depth, overhang, mat):
    """Ellipsoid spanning p0..p1 (plus overhang at each end): a muscle mass along a bone."""
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    b.ellipsoid((p0 + p1) / 2, (depth, width, d.length / 2 + overhang), mat, rot=align_z(d))


def toes(b, front, side, y, z, radius, mat):
    """Four toes in an arc, the middle two furthest forward."""
    for k, dy in enumerate((-0.017, -0.006, 0.006, 0.017)):
        reach = 0.004 if k in (1, 2) else 0.0
        b.ellipsoid((front + reach, side * (y + dy), z), (radius * 1.1, radius, radius * 0.85), mat, segments=12, rings=6)


class Cat:
    """Animated parts of the cat. Ears, eyes and whiskers are ordered (right, left)."""

    def __init__(self, rig, head, tail, ears, eyes, whiskers):
        self.rig, self.head, self.tail = rig, head, tail
        self.ears, self.eyes, self.whiskers = ears, eyes, whiskers


def hind_joints(s):
    return {"hip": Vector((-0.075, s * 0.052, 0.225)), "knee": Vector((0.012, s * 0.066, 0.168)),
            "hock": Vector((-0.088, s * 0.056, 0.13)), "ball": Vector((-0.012, s * 0.052, 0.128))}


def front_joints(s):
    return {"shoulder": Vector((0.07, s * 0.055, 0.455)), "elbow": Vector((0.12, s * 0.105, 0.40)),
            "wrist": BAR_DESIGN + Vector((-0.025, s * 0.126, -0.005)),
            "toes": BAR_DESIGN + Vector((0.013, s * 0.135, 0.009))}


def cat_bones():
    """Bones in the cat's modelling frame: name -> (head, tail, parent, connected)."""
    bones = {"pelvis": (Vector((-0.075, 0, 0.20)), Vector((-0.03, 0, 0.29)), None, False),
             "spine": (Vector((-0.03, 0, 0.29)), Vector((0.055, 0, 0.44)), "pelvis", True),
             "neck": (Vector((0.055, 0, 0.44)), NECK.copy(), "spine", True)}
    for s, side in ((1, "L"), (-1, "R")):
        h, f = hind_joints(s), front_joints(s)
        bones[f"thigh.{side}"] = (h["hip"], h["knee"], "pelvis", False)
        bones[f"shin.{side}"] = (h["knee"], h["hock"], f"thigh.{side}", True)
        bones[f"foot.{side}"] = (h["hock"], h["ball"] + Vector((0.02, 0, 0)), None, False)
        bones[f"upperarm.{side}"] = (f["shoulder"], f["elbow"], "spine", False)
        bones[f"forearm.{side}"] = (f["elbow"], f["wrist"], f"upperarm.{side}", True)
        bones[f"paw.{side}"] = (f["wrist"], f["toes"], None, False)
    return bones


def build_cat_armature(coll, scooter_body):
    """Armature in the cat's modelling frame, with its origin at CAT_PIVOT. The feet and paws have
    no parent bone, so they stay planted on the deck and on the grips; IK on the shins and
    forearms keeps the legs attached while the pelvis and spine move."""
    data = bpy.data.armatures.new("CatRig")
    rig = link(bpy.data.objects.new("CatRig", data), coll, scooter_body, CAT_PIVOT + CAT_OFFSET)
    bones = cat_bones()
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    for name, (head, tail, _, _) in bones.items():
        bone = data.edit_bones.new(name)
        bone.head, bone.tail = head - CAT_PIVOT, tail - CAT_PIVOT
    for name, (_, _, parent, connected) in bones.items():
        if parent:
            data.edit_bones[name].parent = data.edit_bones[parent]
            data.edit_bones[name].use_connect = connected
            if parent == "spine":
                data.edit_bones[name].inherit_scale = 'NONE'  # breathing scales the chest only
    bpy.ops.object.mode_set(mode='OBJECT')
    for chain, joints, end in (("shin", hind_joints, "hock"), ("forearm", front_joints, "wrist")):
        for s, side in ((1, "L"), (-1, "R")):
            target = make_empty(f"{chain.title()}Target.{side}", coll, rig, joints(s)[end] - CAT_PIVOT, size=0.02)
            ik = rig.pose.bones[f"{chain}.{side}"].constraints.new('IK')
            ik.target = target
            ik.chain_count = 2
    return rig


def skin(obj, rig, smoothing=30):
    """Weights from the distance to each bone (inverse fourth power), smoothed over the surface
    so the influence changes gradually between bones, then an Armature modifier. Call before
    adding particle hair so the hair follows the deformation."""
    mesh = obj.data
    n = len(mesh.vertices)
    co = np.empty(n * 3)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    names = [b.name for b in rig.data.bones]
    weights = np.empty((n, len(names)))
    for k, bone in enumerate(rig.data.bones):
        head, axis = np.array(bone.head_local), np.array(bone.tail_local - bone.head_local)
        t = np.clip((co - head) @ axis / (axis @ axis), 0.0, 1.0)
        distance = np.linalg.norm(co - (head + t[:, None] * axis), axis=1)
        weights[:, k] = 1.0 / (distance + 0.004) ** 4
    weights /= weights.sum(axis=1, keepdims=True)

    edges = np.empty(len(mesh.edges) * 2, dtype=np.int64)
    mesh.edges.foreach_get("vertices", edges)
    a, b = edges[0::2], edges[1::2]
    degree = np.bincount(np.concatenate((a, b)), minlength=n)[:, None]
    for _ in range(smoothing):
        neighbours = np.zeros_like(weights)
        np.add.at(neighbours, a, weights[b])
        np.add.at(neighbours, b, weights[a])
        weights = 0.5 * weights + 0.5 * neighbours / np.maximum(degree, 1)

    for k, name in enumerate(names):
        group = obj.vertex_groups.new(name=name)
        for index in np.nonzero(weights[:, k] > 0.01)[0]:
            group.add([int(index)], float(weights[index, k]), 'REPLACE')
    obj.modifiers.new("Rig", 'ARMATURE').object = rig


def parent_to_bone(obj, rig, bone_name):
    """Parent obj to a bone without moving it: Blender places bone children relative to the
    bone's tail, so the parent inverse is the inverse of the rest tail matrix."""
    bone = rig.data.bones[bone_name]
    obj.parent = rig
    obj.parent_type = 'BONE'
    obj.parent_bone = bone_name
    obj.matrix_parent_inverse = (Matrix.Translation(bone.tail_local) @ bone.matrix_local.to_3x3().to_4x4()).inverted()


def hind_leg(b, s, mat):
    """Standing on the hind legs: thigh forward to the knee, shin back down to the hock,
    and the foot flat on the deck from the hock to the toes."""
    hip, knee, hock, ball = hind_joints(s).values()
    b.ellipsoid((-0.085, s * 0.04, 0.215), (0.06, 0.04, 0.06), mat)  # haunch
    segment(b, hip, knee, 0.03, 0.04, 0.015, mat)
    b.limb(knee, hock, 0.02, 0.013, mat)
    b.ellipsoid(hock, (0.016, 0.013, 0.014), mat, segments=12, rings=6)
    b.limb(hock + Vector((0.01, 0, 0)), ball, 0.0125, 0.015, mat)
    toes(b, ball.x + 0.012, s, 0.052, 0.126, 0.0085, mat)


def front_leg(b, s, mat):
    """Shoulder blade, upper arm out to the elbow, forearm up to the wrist, and a paw whose
    toes curl over the front of the handlebar grip."""
    shoulder, elbow, wrist, _ = front_joints(s).values()
    b.ellipsoid((0.05, s * 0.058, 0.475), (0.03, 0.022, 0.048), mat, rot=R(math.radians(30), 'Y'))
    segment(b, shoulder, elbow, 0.026, 0.028, 0.01, mat)
    b.limb(elbow, wrist, 0.019, 0.014, mat)
    b.ellipsoid(BAR_DESIGN + Vector((-0.008, s * 0.135, 0.014)), (0.022, 0.024, 0.016), mat)
    toes(b, BAR_DESIGN.x + 0.009, s, 0.135, BAR_DESIGN.z + 0.009, 0.0085, mat)


def build_cat(mats, coll, scooter_body):
    cat_rig = build_cat_armature(coll, scooter_body)
    FUR = 0

    b = MeshBuilder(CAT_PIVOT)
    b.ellipsoid((-0.005, 0, 0.34), (0.085, 0.08, 0.16), FUR, rot=R(math.radians(34), 'Y'))
    b.ellipsoid((0.055, 0, 0.44), (0.075, 0.072, 0.08), FUR)
    for s in (-1, 1):
        hind_leg(b, s, FUR)
        front_leg(b, s, FUR)
    body = b.to_object("CatBody", [mats["fur"]], coll, cat_rig)
    remesh(body, 0.0035, 8)
    skin(body, cat_rig)
    add_fur(body, mats, 5000, 0.012, (-1.0, 0.0, -0.6))

    c = MeshBuilder(CAT_PIVOT)
    neck_dir = Vector((0.055, 0, 0.16)).normalized()
    collar_c = Vector((0.085, 0, 0.515))
    c.torus(Matrix.Translation(collar_c) @ align_z(neck_dir), 0.054, 0.01, 0, seg_minor=8)
    front = (Vector((1, 0, 0)) - neck_dir * neck_dir.x).normalized()
    c.ellipsoid(collar_c + front * 0.068 - neck_dir * 0.012, (0.012,) * 3, 1, segments=16, rings=8)
    parent_to_bone(c.to_object("CatCollar", [mats["collar"], mats["gold"]], coll, cat_rig), cat_rig, "neck")

    tail_curve = build_tail(mats, coll, cat_rig)
    head_rig, ears, eyes, whiskers = build_head(mats, coll, cat_rig)
    return Cat(cat_rig, head_rig, tail_curve, ears, eyes, whiskers)


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
    add_fur(tail, mats, 900, 0.0144, (0.8, 0.0, 0.0))
    parent_to_bone(curve, cat_rig, "pelvis")
    parent_to_bone(tail, cat_rig, "pelvis")
    return curve


def build_head(mats, coll, cat_rig):
    head_rig = make_empty("HeadRig", coll, cat_rig, NECK - CAT_PIVOT, size=0.05)
    head_rig.scale = (1.12, 1.12, 1.12)
    parent_to_bone(head_rig, cat_rig, "neck")
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
    add_fur(head, mats, 2500, 0.0072, (-1.2, 0.0, 0.0), density_group=group.name)

    f = MeshBuilder(NECK)
    f.ellipsoid((0.193, 0, 0.596), (0.007, 0.011, 0.007), 0)
    f.to_object("CatFace", [mats["nose"]], coll, head_rig)

    whiskers = []
    for s in (-1, 1):
        root = Vector((0.183, s * 0.028, 0.585))
        w = MeshBuilder(root)
        for end in ((0.205, 0.115, 0.598), (0.207, 0.118, 0.583), (0.2, 0.11, 0.568)):
            w.rod(root, (end[0], s * end[1], end[2]), 0.0012, 0, segments=4)
        whiskers.append(w.to_object("WhiskersL" if s > 0 else "WhiskersR", [mats["whisker"]], coll, head_rig,
                                    root - NECK))

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
    return head_rig, ears, eyes, whiskers


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
    road height under each wheel. The cat rides on the body as a second spring (its legs), and
    its head on the cat as a third (the neck), so the head nods a moment after each bump."""

    AXLES = (FRONT_AXLE.x, REAR_AXLE.x)

    def __init__(self):
        self.k = (2 * math.pi * SPRING_HZ) ** 2 / 2
        self.c = SPRING_DAMPING * math.sqrt(2 * self.k)
        self.inertia = 0.28 ** 2
        self.heave = self.heave_v = self.slope = self.slope_v = 0.0
        self.cat_k = (2 * math.pi * 2.5) ** 2
        self.cat_c = 2 * 0.3 * 2 * math.pi * 2.5
        self.cat = self.cat_v = 0.0
        self.head_k = (2 * math.pi * 3.5) ** 2
        self.head_c = 2 * 0.25 * 2 * math.pi * 3.5
        self.head = self.head_v = 0.0

    def step(self, u, u_v, dt):
        forces = [self.k * (ui - (self.heave + self.slope * x)) + self.c * (vi - (self.heave_v + self.slope_v * x))
                  for ui, vi, x in zip(u, u_v, self.AXLES)]
        accel = sum(forces)
        self.heave_v += accel * dt
        self.slope_v += sum(f * x for f, x in zip(forces, self.AXLES)) / self.inertia * dt
        self.heave += self.heave_v * dt
        self.slope += self.slope_v * dt
        cat_accel = -self.cat_k * self.cat - self.cat_c * self.cat_v - accel
        self.cat_v += cat_accel * dt
        self.cat += self.cat_v * dt
        self.head_v += (-self.head_k * self.head - self.head_c * self.head_v - (accel + cat_accel)) * dt
        self.head += self.head_v * dt

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


def pose_cat(rig, lean, drop, breath, frame):
    """Lower the pelvis by drop and lean the spine forward by lean (radians) about its base, both
    in the armature frame; widen the chest by the fraction breath. Keys the bones' channels."""
    pelvis, spine = rig.pose.bones["pelvis"], rig.pose.bones["spine"]
    rest = pelvis.bone.matrix_local
    pelvis.matrix_basis = rest.inverted() @ T(0, 0, drop) @ rest
    # The spine's parent moves by the same translation, so its basis is the rotation, then the
    # breathing scale across the bone (its local X and Z); its children do not inherit scale.
    rest, hinge = spine.bone.matrix_local, spine.bone.head_local
    spine.matrix_basis = rest.inverted() @ T(*hinge) @ R(lean, 'Y') @ T(*-hinge) @ rest @ S(1 + breath, 1, 1 + breath)
    for bone in (pelvis, spine):
        for path in ("location", "rotation_quaternion", "scale"):
            bone.keyframe_insert(path, frame=frame)


EAR_TWITCHES = {0: [150], 1: [60, 205]}  # ear index (right, left) -> start frames


def twitch(frame, starts):
    """Ear twitch envelope: up to 1 three frames after each start, back to 0 after seven."""
    return max((max(0.0, 1.0 - abs(frame - start - 3) / 4.0) for start in starts), default=0.0)


def animate(scene, road, scooter, cam, cat):
    bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
    rig, wheels = scooter.rig, scooter.wheels
    head_rig, tail = cat.head, cat.tail
    ear_rest = [ear.rotation_euler.copy() for ear in cat.ears]
    breath_phase = 0.0
    s = road.arc_length_at_x(RIDE_START_X)
    v = BASE_SPEED
    lean = 0.0
    travelled = 0.0
    cam_heading = None
    prev_q = None
    ride_x = []
    susp = Suspension()
    bump_frames = []
    body_lean = 0.0

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

        # The cat leans into climbs and back on descents, crouches a little uphill, and absorbs
        # the bumps by dropping its hips; IK keeps the paws on the grips and the feet planted.
        pitch = math.atan(slope)
        body_lean += (min(max(1.2 * pitch, -0.12), MAX_LEAN) - body_lean) * 0.08
        drop = (0.003 * math.sin(2 * math.pi * 1.6 * t) + min(max(susp.cat * 0.5, -0.025), 0.02)
                - 0.04 * max(pitch, 0.0))
        # Breathing: about 0.45 Hz on the flat, up to twice that on the climb.
        breath_phase += 2 * math.pi * (0.45 + 0.45 * min(max(pitch, 0.0) / 0.25, 1.0)) / FPS
        pose_cat(cat.rig, body_lean, drop, 0.015 * math.sin(breath_phase), frame)
        # Cats keep their head level: counter the scooter's lean and the body's lean, and look
        # into the turn. The neck spring adds a nod that lags each bump.
        nod = min(max(susp.head * 15.0, -0.2), 0.08)  # chin down only a little: the bar is below it
        head_rig.rotation_euler = (lean * 0.6, 0.05 * math.sin(2 * math.pi * 0.5 * t) - body_lean * 0.7 + nod,
                                   min(max(kappa * 1.5, -0.25), 0.25))
        head_rig.keyframe_insert("rotation_euler", frame=frame)

        # Ears fold back with speed and flutter in the wind; whiskers sweep back and flutter.
        wind = v / BASE_SPEED
        for k, (ear, rest, sign) in enumerate(zip(cat.ears, ear_rest, (-1, 1))):
            flick = 0.45 * twitch(frame, EAR_TWITCHES[k])
            flutter = 0.03 * wind * math.sin(2 * math.pi * 6.0 * t + 1.7 * k) + 0.015 * wind * math.sin(2 * math.pi * 9.3 * t)
            ear.rotation_euler = (rest.x - sign * (flick + flutter), rest.y - 0.15 * wind - 0.3 * flick, rest.z)
            ear.keyframe_insert("rotation_euler", frame=frame)
        for k, (whisker, sign) in enumerate(zip(cat.whiskers, (-1, 1))):
            sweep = 0.3 * wind + 0.05 * wind * math.sin(2 * math.pi * 8.0 * t + 2.3 * k)
            whisker.rotation_euler = (0.03 * wind * math.sin(2 * math.pi * 11.0 * t + k), 0.0, sign * sweep)
            whisker.keyframe_insert("rotation_euler", frame=frame)

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

    for eye in cat.eyes:
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
    scene.eevee.volumetric_end = 300.0
    scene.eevee.use_volumetric_shadows = True
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
    cat = build_cat(mats, actors, scooter.body)

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 32
    cam_data.clip_start = 0.05
    cam_data.clip_end = FAR_RADIUS * 1.5
    cam = link(bpy.data.objects.new("Camera", cam_data), scene.collection)
    track = cam.constraints.new('TRACK_TO')
    track.target = scooter.target
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    cam_data.dof.use_dof = True
    cam_data.dof.focus_object = scooter.target
    cam_data.dof.aperture_fstop = DOF_FSTOP
    scene.camera = cam

    ride_x = animate(scene, road, scooter, cam, cat)

    terrain = build_terrain(road, mats, env, ride_x)
    build_far_terrain(mats, env)
    build_road(road, mats, env)
    build_speed_bumps(road, mats, env)
    build_fence(road, mats, env)
    scatter_trees(road, env, rng, ride_x)
    scatter_rocks(road, mats, env, rng, ride_x)
    build_grass(mats, env, terrain, rng)
    build_world(scene)
    build_haze(env)
    configure_render(scene)

    zs = [p.z for p in road.points if ride_x[0] <= p.x <= ride_x[1]]
    print(f"road height along ride: {min(zs):.1f} .. {max(zs):.1f} m")
    scene.frame_set(1)
    BLEND_PATH.parent.mkdir(exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))


if __name__ == "__main__":
    main()
