import trimesh
import numpy as np
import os


data_dirs = ["train", "validation", "test"]
shape_count = [50, 10, 10]

shape_directories = ["annulus", "capsule", "cone", "cube", "cylinder", "sphere"]

os.makedirs("data", exist_ok=True)
for shape_dir in shape_directories:
    os.makedirs(os.path.join("data", shape_dir), exist_ok=True)
    sub_directories = ["train", "validation", "test"]
    for sub_dir in sub_directories:
        path = os.path.join("data", shape_dir, sub_dir)
        os.makedirs(path, exist_ok=True)


for j, data_dir in enumerate(data_dirs):
    for i in range(shape_count[j]):
        # Cube
        cube_dimensions = np.random.uniform(low=0.5, high=2.0, size=(3,))
        cube = trimesh.creation.box(cube_dimensions)

        # Sphere
        sphere_radius = np.random.uniform(low=0.5, high=1.0)
        sphere_count = np.random.randint(low=5, high=15)
        sphere = trimesh.creation.uv_sphere(radius=sphere_radius, count=[sphere_count, sphere_count])

        # Cone
        cone_height = np.random.uniform(low=0.5, high=2.0)
        cone_radius = np.random.uniform(low=0.5, high=1.0)
        cone_sections = np.random.randint(low=10, high=30)
        cone = trimesh.creation.cone(radius=cone_radius, height=cone_height, sections=cone_sections)

        # Cylinder
        cylinder_height = np.random.uniform(low=0.5, high=2.0)
        cylinder_radius = np.random.uniform(low=0.5, high=2.0)
        cylinder_sections = np.random.randint(low=10, high=30)
        cylinder = trimesh.creation.cylinder(radius=cylinder_radius, height=cylinder_height, sections=cylinder_sections)

        # Annulus
        annulus_radius_in = np.random.uniform(low=0.5, high=1.0)
        annulus_radius_out = annulus_radius_in + np.random.uniform(low=0, high=1.0)
        annulus_sections = np.random.randint(low=10, high=30)
        annulus_height = np.random.uniform(low=0.5, high=2.0)
        annulus = trimesh.creation.annulus(r_min=annulus_radius_in, r_max=annulus_radius_out, height=annulus_height, sections=annulus_sections)

        # Capsule
        capsule_height = np.random.uniform(low=0.5, high=2.0)
        capsule_radius = np.random.uniform(low=0.5, high=1.0)
        capsule_count = np.random.randint(low=5, high=15)
        capsule = trimesh.creation.capsule(height=capsule_height, radius=capsule_radius, count=[capsule_count, capsule_count])

        # Saving
        annulus.export('data/annulus/%s/%i.obj'%(data_dir, i))
        capsule.export('data/capsule/%s/%i.obj'%(data_dir, i))
        cube.export('data/cube/%s/%i.obj'%(data_dir, i))
        sphere.export('data/sphere/%s/%i.obj'%(data_dir, i))
        cone.export('data/cone/%s/%i.obj'%(data_dir, i))
        cylinder.export('data/cylinder/%s/%i.obj'%(data_dir, i))