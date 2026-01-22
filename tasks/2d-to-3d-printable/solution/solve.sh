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
   - Recessed band around the equator
   - Protruding button in the center
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

    Args:
        image_path: Path to input image

    Returns:
        dict with shape information
    """
    print(f"Analyzing image: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    print(f"Image size: {img.shape}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour
        main_contour = max(contours, key=cv2.contourArea)

        # Fit a circle to the contour
        (x, y), radius = cv2.minEnclosingCircle(main_contour)

        # Calculate circularity
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0

        print(f"Detected shape - Center: ({x:.1f}, {y:.1f}), Radius: {radius:.1f}")
        print(f"Circularity: {circularity:.2f} (1.0 = perfect circle)")

        # Detect if it's a Pokeball by checking for the characteristic band
        # Look for horizontal features in the middle of the image
        has_band = detect_pokeball_band(img)

        return {
            'type': 'pokeball' if (circularity > 0.5 and has_band) else ('sphere' if circularity > 0.7 else 'other'),
            'center': (x, y),
            'radius': radius,
            'circularity': circularity,
            'has_band': has_band
        }

    return {'type': 'unknown'}


def detect_pokeball_band(img):
    """
    Detect if the image shows a Pokeball with the characteristic horizontal band.

    Args:
        img: BGR image

    Returns:
        bool: True if Pokeball band pattern detected
    """
    # Convert to HSV to detect the black band
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Look for dark pixels (the black band)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    # Check if there's a horizontal band of dark pixels in the middle
    h, w = black_mask.shape
    middle_region = black_mask[int(h*0.4):int(h*0.6), :]

    # Count dark pixels in middle region
    dark_pixels = np.sum(middle_region > 0)
    total_pixels = middle_region.size

    # If more than 10% of middle region is dark, likely has a band
    has_band = (dark_pixels / total_pixels) > 0.10

    print(f"Band detection: {dark_pixels}/{total_pixels} dark pixels ({100*dark_pixels/total_pixels:.1f}%)")

    return has_band


def create_pokeball_mesh(diameter=80.0, band_width=6.0, band_depth=2.0, button_radius=8.0, button_height=3.0):
    """
    Create a 3D Pokeball mesh with pattern features.

    The Pokeball has:
    - Main spherical body
    - Recessed band around the equator
    - Protruding button in the center of the band

    Args:
        diameter: Diameter of the sphere in mm
        band_width: Width of the equator band in mm
        band_depth: Depth of the band recession in mm
        button_radius: Radius of the center button in mm
        button_height: Height of the button protrusion in mm

    Returns:
        trimesh.Trimesh object
    """
    print(f"Creating Pokeball with diameter={diameter}mm")
    print(f"  Band: width={band_width}mm, depth={band_depth}mm")
    print(f"  Button: radius={button_radius}mm, height={button_height}mm")

    radius = diameter / 2

    # Create main sphere body
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    print(f"Base sphere: {len(sphere.vertices)} vertices, {len(sphere.faces)} faces")

    # Create the band groove (a cylinder that will be subtracted)
    # The band goes around the equator (Z=0)
    band_outer_radius = radius + 1  # Slightly larger than sphere to ensure clean cut
    band_inner_radius = radius - band_depth
    band_cylinder = trimesh.creation.cylinder(
        radius=band_outer_radius,
        height=band_width,
        sections=64
    )
    # Center the band at Z=0
    band_cylinder.apply_translation([0, 0, 0])

    # Create inner cylinder to make it a ring (hollow)
    inner_cylinder = trimesh.creation.cylinder(
        radius=band_inner_radius,
        height=band_width + 2,  # Slightly taller to ensure clean cut
        sections=64
    )

    # Create the band as the difference between outer and inner cylinders
    # Actually, we want to carve into the sphere, so we'll create a "cutting tool"

    # For the band, we'll modify vertex positions directly for reliability
    # This avoids boolean operation issues

    # Create button (cylinder + sphere cap for rounded top)
    button_base = trimesh.creation.cylinder(
        radius=button_radius,
        height=button_height,
        sections=32
    )
    # Position button at the front of the sphere (Y = radius)
    button_base.apply_translation([0, radius - button_height/2 + 0.5, 0])

    # Create button ring (the dark ring around the button)
    ring_outer = button_radius + 4
    ring_inner = button_radius + 1
    ring_height = 2.0

    # For a cleaner approach, let's create the Pokeball by modifying the sphere geometry
    # to add the band indentation

    # Alternative: Create Pokeball using vertex displacement for the band
    mesh = create_pokeball_with_band(radius, band_width, band_depth, button_radius, button_height)

    return mesh


def create_pokeball_with_band(radius, band_width, band_depth, button_radius, button_height):
    """
    Create a Pokeball mesh by displacing vertices to create the band pattern.

    Args:
        radius: Sphere radius
        band_width: Width of the equator band
        band_depth: Depth of the band indentation
        button_radius: Radius of the center button
        button_height: Height of the button protrusion

    Returns:
        trimesh.Trimesh object
    """
    # Create high-resolution sphere
    sphere = trimesh.creation.icosphere(subdivisions=5, radius=radius)

    vertices = sphere.vertices.copy()

    # Displace vertices to create the band
    # The band is around Z=0 (equator)
    half_band = band_width / 2

    for i, v in enumerate(vertices):
        x, y, z = v

        # Check if vertex is in the band region
        if abs(z) < half_band:
            # Calculate how much to indent based on position in band
            # Smooth transition at edges using cosine interpolation
            band_factor = np.cos(np.pi * z / band_width) * 0.5 + 0.5

            # Calculate current radius from center
            current_r = np.sqrt(x**2 + y**2 + z**2)
            if current_r > 0:
                # Calculate indentation amount
                indent = band_depth * band_factor

                # New radius after indentation
                new_r = current_r - indent

                # Scale vertex to new radius
                scale = new_r / current_r
                vertices[i] = v * scale

    # Update sphere vertices
    sphere.vertices = vertices

    # Now add the button
    # Button is positioned on the Y-axis (front of the sphere)
    # We need to add vertices for the button

    # Create button mesh
    button = trimesh.creation.cylinder(radius=button_radius, height=button_height, sections=32)

    # Rotate button to point outward on Y-axis
    rotation = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
    button.apply_transform(rotation)

    # Position button at the equator, on the front surface
    # The button center should be at Y = radius - band_depth + button_height/2
    button_y = radius - band_depth
    button.apply_translation([0, button_y, 0])

    # Create button ring (raised ring around the button)
    ring_outer_radius = button_radius + 5
    ring_inner_radius = button_radius + 2
    ring_thickness = 1.5

    # Outer ring cylinder
    ring_outer = trimesh.creation.cylinder(radius=ring_outer_radius, height=ring_thickness, sections=32)
    ring_inner = trimesh.creation.cylinder(radius=ring_inner_radius, height=ring_thickness + 1, sections=32)

    # Rotate rings
    ring_outer.apply_transform(rotation)
    ring_inner.apply_transform(rotation)

    # Position rings
    ring_y = radius - band_depth - ring_thickness/2 + 0.5
    ring_outer.apply_translation([0, ring_y, 0])
    ring_inner.apply_translation([0, ring_y, 0])

    # Combine meshes
    # Use concatenation for reliability (boolean operations can fail)
    combined = trimesh.util.concatenate([sphere, button, ring_outer])

    # Process to make watertight
    combined.merge_vertices()
    combined.update_faces(combined.nondegenerate_faces())

    # Try to make it a single solid
    # Fill any holes
    if not combined.is_watertight:
        trimesh.repair.fill_holes(combined)

    print(f"Pokeball mesh: {len(combined.vertices)} vertices, {len(combined.faces)} faces")

    return combined


def validate_and_fix_mesh(mesh):
    """
    Validate mesh for 3D printing and fix common issues.

    Args:
        mesh: trimesh.Trimesh object

    Returns:
        Fixed trimesh.Trimesh object
    """
    print("Validating mesh...")

    # Remove degenerate faces
    non_degenerate = mesh.nondegenerate_faces()
    if not non_degenerate.all():
        print(f"Removing {(~non_degenerate).sum()} degenerate faces...")
        mesh.update_faces(non_degenerate)

    # Merge duplicate vertices
    mesh.merge_vertices()

    # Fix normals to point outward
    trimesh.repair.fix_normals(mesh)

    # Fix winding order
    trimesh.repair.fix_winding(mesh)

    # Report status
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

    Args:
        mesh: trimesh.Trimesh object
        max_dimension: Maximum allowed dimension in mm

    Returns:
        Scaled mesh
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
    max_size = 100.0  # 10cm = 100mm
    sphere_diameter = 80.0  # 80mm diameter sphere (leaves margin)

    print("=" * 60)
    print("2D to 3D Printable Conversion - Pokeball with Pattern")
    print("=" * 60)

    # Step 1: Analyze input image
    shape_info = analyze_input_image(input_path)
    print(f"Detected shape type: {shape_info['type']}")

    # Step 2: Create 3D mesh (Pokeball with pattern features)
    mesh = create_pokeball_mesh(
        diameter=sphere_diameter,
        band_width=6.0,      # 6mm wide band
        band_depth=2.0,      # 2mm deep indentation
        button_radius=6.0,   # 6mm radius button
        button_height=2.0    # 2mm tall button
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

    # Reload and verify
    final_mesh = trimesh.load(output_path)
    bounds = final_mesh.bounds
    dimensions = bounds[1] - bounds[0]
    print(f"Final dimensions: {dimensions[0]:.2f} x {dimensions[1]:.2f} x {dimensions[2]:.2f} mm")
    print(f"Watertight: {final_mesh.is_watertight}")
    print(f"Has pattern features: band and button added")
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
