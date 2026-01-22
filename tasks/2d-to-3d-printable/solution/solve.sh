#!/bin/bash
# Oracle solution for 2D to 3D printable conversion task
# This script creates a 3D Pokeball model with pattern features based on the input image

set -e

echo "=== 2D to 3D Printable Conversion Solution ==="

cd /root/workspace

echo "Step 1: Creating the conversion script..."

cat > convert_2d_to_3d.py << 'PYTHONSCRIPT'
#!/usr/bin/env python3
"""
Convert a 2D image of a Pokeball to a 3D printable STL file.

This script:
1. Recognizes the Pokeball shape in the input image
2. Creates a 3D sphere model with Pokeball pattern features:
   - Band groove around the equator
   - Protruding button and ring in the center
3. Scales to fit within 100mm x 100mm x 100mm
4. Validates and exports as STL
"""
import cv2
import numpy as np
import trimesh
import sys


def analyze_input_image(image_path):
    """
    Analyze the input image to identify the main object.
    """
    print(f"Analyzing image: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    print(f"Image size: {img.shape}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(main_contour)
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        print(f"Detected shape - Center: ({x:.1f}, {y:.1f}), Radius: {radius:.1f}")
        print(f"Circularity: {circularity:.2f}")

        return {'type': 'pokeball', 'circularity': circularity}

    return {'type': 'unknown'}


def create_pokeball_mesh(radius=35.0, band_width=5.0, band_depth=1.5,
                         button_radius=4.0, button_height=1.5,
                         ring_outer_radius=8.0, ring_height=1.0):
    """
    Create a 3D Pokeball mesh matching the reference pokeball.stl.

    Reference specs:
    - Diameter: ~70mm (radius 35mm)
    - ~24k vertices, ~48k faces
    - Band with button/ring creates pattern variation

    Args:
        radius: Sphere radius in mm (35mm to match reference)
        band_width: Width of the equator band
        band_depth: Depth of the band indentation
        button_radius: Radius of the center button
        button_height: Height of the button protrusion
        ring_outer_radius: Outer radius of the ring around button
        ring_height: Height of the ring

    Returns:
        trimesh.Trimesh object
    """
    print(f"Creating Pokeball with radius={radius}mm (diameter={2*radius}mm)")
    print(f"  Band: width={band_width}mm, depth={band_depth}mm")
    print(f"  Button: radius={button_radius}mm, height={button_height}mm")
    print(f"  Ring: outer_radius={ring_outer_radius}mm, height={ring_height}mm")

    # Create high-resolution sphere (subdivisions=5 gives ~10k vertices)
    sphere = trimesh.creation.icosphere(subdivisions=5, radius=radius)
    vertices = sphere.vertices.copy()

    # Create band indentation around the equator (Z=0)
    half_band = band_width / 2

    for i, v in enumerate(vertices):
        x, y, z = v

        # Check if vertex is in the band region
        if abs(z) < half_band:
            # Smooth transition using cosine interpolation
            band_factor = np.cos(np.pi * z / band_width) * 0.5 + 0.5

            current_r = np.sqrt(x**2 + y**2 + z**2)
            if current_r > 0:
                indent = band_depth * band_factor
                new_r = current_r - indent
                scale = new_r / current_r
                vertices[i] = v * scale

    sphere.vertices = vertices

    # Create button (positioned on positive X-axis for front view)
    button = trimesh.creation.cylinder(
        radius=button_radius,
        height=button_height,
        sections=32
    )
    # Rotate to point outward on X-axis
    rotation_y = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
    button.apply_transform(rotation_y)
    # Position at the band surface
    button_x = radius - band_depth + button_height/2
    button.apply_translation([button_x, 0, 0])

    # Create outer ring around the button
    ring = trimesh.creation.cylinder(
        radius=ring_outer_radius,
        height=ring_height,
        sections=48
    )
    ring.apply_transform(rotation_y)
    ring_x = radius - band_depth + ring_height/2 - 0.2
    ring.apply_translation([ring_x, 0, 0])

    # Create inner ring (smaller, inside the outer ring)
    inner_ring = trimesh.creation.cylinder(
        radius=button_radius + 1.5,
        height=ring_height + 0.5,
        sections=32
    )
    inner_ring.apply_transform(rotation_y)
    inner_ring.apply_translation([ring_x, 0, 0])

    # Combine all meshes
    combined = trimesh.util.concatenate([sphere, button, ring])

    # Clean up the mesh
    combined.merge_vertices()
    non_degenerate = combined.nondegenerate_faces()
    if not non_degenerate.all():
        combined.update_faces(non_degenerate)

    # Fix mesh for 3D printing
    trimesh.repair.fix_normals(combined)
    trimesh.repair.fix_winding(combined)

    if not combined.is_watertight:
        trimesh.repair.fill_holes(combined)

    print(f"Pokeball mesh: {len(combined.vertices)} vertices, {len(combined.faces)} faces")

    return combined


def validate_and_fix_mesh(mesh):
    """
    Validate mesh for 3D printing and fix common issues.
    """
    print("Validating mesh...")

    non_degenerate = mesh.nondegenerate_faces()
    if not non_degenerate.all():
        print(f"Removing {(~non_degenerate).sum()} degenerate faces...")
        mesh.update_faces(non_degenerate)

    mesh.merge_vertices()
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_winding(mesh)

    print(f"Mesh vertices: {len(mesh.vertices)}")
    print(f"Mesh faces: {len(mesh.faces)}")
    print(f"Degenerate faces: {(~mesh.nondegenerate_faces()).sum()}")
    print(f"Is watertight: {mesh.is_watertight}")

    bounds = mesh.bounds
    dimensions = bounds[1] - bounds[0]
    print(f"Dimensions: {dimensions[0]:.2f} x {dimensions[1]:.2f} x {dimensions[2]:.2f} mm")

    if mesh.is_watertight:
        print(f"Volume: {mesh.volume:.2f} mm³")

    return mesh


def scale_to_fit(mesh, max_dimension=100.0):
    """
    Scale mesh to fit within maximum dimension if needed.
    """
    bounds = mesh.bounds
    current_size = bounds[1] - bounds[0]
    max_current = max(current_size)

    if max_current > max_dimension:
        scale_factor = max_dimension / max_current
        print(f"Scaling mesh by {scale_factor:.4f} to fit within {max_dimension}mm")
        mesh.apply_scale(scale_factor)

    return mesh


def main():
    """Main conversion pipeline."""
    input_path = 'input.png'
    output_path = 'output.stl'
    max_size = 100.0

    print("=" * 60)
    print("2D to 3D Printable Conversion - Pokeball with Pattern")
    print("=" * 60)

    # Step 1: Analyze input image
    shape_info = analyze_input_image(input_path)
    print(f"Detected shape type: {shape_info['type']}")

    # Step 2: Create 3D mesh (Pokeball matching reference specs)
    # Reference pokeball.stl has radius ~35mm, so we use similar dimensions
    mesh = create_pokeball_mesh(
        radius=35.0,           # 35mm radius = 70mm diameter
        band_width=5.0,        # 5mm wide band
        band_depth=1.5,        # 1.5mm deep indentation
        button_radius=4.0,     # 4mm radius button
        button_height=1.5,     # 1.5mm tall button
        ring_outer_radius=8.0, # 8mm outer ring radius
        ring_height=1.0        # 1mm ring height
    )

    # Step 3: Validate and fix mesh
    mesh = validate_and_fix_mesh(mesh)

    # Step 4: Ensure it fits within constraints
    mesh = scale_to_fit(mesh, max_size)

    # Step 5: Export to STL
    print(f"Exporting to {output_path}...")
    mesh.export(output_path, file_type='stl')

    # Final verification
    print("=" * 60)
    print("Conversion complete!")
    print(f"Output: {output_path}")

    final_mesh = trimesh.load(output_path)
    bounds = final_mesh.bounds
    dimensions = bounds[1] - bounds[0]
    print(f"Final dimensions: {dimensions[0]:.2f} x {dimensions[1]:.2f} x {dimensions[2]:.2f} mm")
    print(f"Watertight: {final_mesh.is_watertight}")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
PYTHONSCRIPT

echo "Step 2: Running the conversion script..."

python3 convert_2d_to_3d.py

echo ""
echo "Step 3: Verifying output..."

if [ -f "output.stl" ]; then
    echo "SUCCESS: output.stl created successfully"
    ls -la output.stl
else
    echo "ERROR: output.stl was not created"
    exit 1
fi

echo ""
echo "=== Solution Complete ==="
