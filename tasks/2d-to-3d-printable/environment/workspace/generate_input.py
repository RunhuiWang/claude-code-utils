#!/usr/bin/env python3
"""
Generate sample input image for 2D to 3D conversion task.
Creates a simple star shape on a white background.
"""
from PIL import Image, ImageDraw
import math


def generate_star(num_points=5, outer_radius=180, inner_radius=70, size=(400, 400)):
    """
    Generate a star shape image.

    Args:
        num_points: Number of star points
        outer_radius: Radius to outer points
        inner_radius: Radius to inner points
        size: Image size (width, height)
    """
    # Create white background
    img = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(img)

    center_x, center_y = size[0] // 2, size[1] // 2

    # Calculate star points
    points = []
    for i in range(num_points * 2):
        angle = math.pi / 2 + i * math.pi / num_points  # Start from top
        if i % 2 == 0:
            # Outer point
            r = outer_radius
        else:
            # Inner point
            r = inner_radius
        x = center_x + r * math.cos(angle)
        y = center_y - r * math.sin(angle)  # Negative because Y is inverted in images
        points.append((x, y))

    # Draw filled star (black on white background)
    draw.polygon(points, fill='black', outline='black')

    return img


def generate_hexagon(radius=180, size=(400, 400)):
    """
    Generate a regular hexagon shape image.
    """
    img = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(img)

    center_x, center_y = size[0] // 2, size[1] // 2

    # Calculate hexagon points
    points = []
    for i in range(6):
        angle = math.pi / 6 + i * math.pi / 3  # Start from right side
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        points.append((x, y))

    draw.polygon(points, fill='black', outline='black')

    return img


if __name__ == '__main__':
    # Generate a 5-pointed star as the input image
    star_img = generate_star(num_points=5, outer_radius=180, inner_radius=70)
    star_img.save('input.png')
    print("Generated input.png - 5-pointed star shape")
