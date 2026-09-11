#!/usr/bin/env python3
"""Create test fixture images for cross-modal translation tests."""

from PIL import Image, ImageDraw, ImageFont
import os

def create_scene_image(output_path="scene.png"):
    """Create a test scene image with visual elements."""
    # Create 128x128 test image (multiple of 16 for tiles)
    img_size = 128
    image = Image.new('RGB', (img_size, img_size), color='white')
    draw = ImageDraw.Draw(image)

    # Add a blue sky gradient
    for y in range(img_size):
        blue_val = int(100 + (y / img_size) * 155)
        for x in range(img_size):
            draw.point((x, y), fill=(200, 220, blue_val))

    # Add a green rectangle (ground)
    draw.rectangle([0, img_size//2, img_size, img_size], fill=(50, 150, 50))

    # Add a yellow circle (sun)
    draw.ellipse([10, 10, 30, 30], fill=(255, 255, 0))

    # Add a simple house
    draw.rectangle([40, img_size//2 - 20, 80, img_size//2], fill=(200, 100, 50))
    # Roof
    draw.polygon([(35, img_size//2 - 20), (60, img_size//2 - 40), (85, img_size//2 - 20)], fill=(150, 50, 50))
    # Door
    draw.rectangle([55, img_size//2 - 10, 65, img_size//2], fill=(100, 50, 0))
    # Window
    draw.rectangle([45, img_size//2 - 15, 50, img_size//2 - 10], fill=(150, 200, 255))

    # Add a red rectangle (simple building)
    draw.rectangle([90, img_size//2 - 30, 120, img_size//2], fill=(200, 50, 50))
    draw.rectangle([95, img_size//2 - 15, 105, img_size//2 - 5], fill=(150, 200, 255))
    draw.rectangle([110, img_size//2 - 15, 115, img_size//2], fill=(150, 200, 255))

    # Save the image
    image.save(output_path)
    print(f"Created test scene image: {output_path} ({img_size}x{img_size})")

if __name__ == '__main__':
    create_scene_image()