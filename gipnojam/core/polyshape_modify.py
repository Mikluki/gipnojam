import numpy as np


def rotate_polygons(poly_shape, degrees, prev_rotation=0):
    """
    Rotate all polygons in the polygon_stack by the specified degrees.
    Each item in polygon_stack is a list containing [polygon_array, label].

    Parameters:
    -----------
    poly_shape : PolyShape object
        Object containing polygon_stack attribute with list of [array, label] pairs
    degrees : float
        Degrees to rotate from the original position (positive is counterclockwise)
    prev_rotation : float, optional
        Previous rotation angle in degrees, used for continuous rotation

    Returns:
    --------
    float
        The new total rotation angle in degrees
    """
    # Convert degrees to radians
    theta = np.radians(degrees - prev_rotation)

    # Create rotation matrix
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    # Create a new list to store rotated polygons with their labels
    rotated_stack = []

    # Rotate each polygon in the stack
    for polygon_data in poly_shape.polygon_stack:
        polygon = polygon_data[0]  # Get the polygon array
        label = polygon_data[1]  # Get the label

        # Calculate centroid
        centroid = polygon.mean(axis=0)

        # Center the polygon at origin
        centered = polygon - centroid

        # Apply rotation
        rotated = np.dot(centered, rotation_matrix)

        # Move back to original position
        rotated = rotated + centroid

        # Add to new stack with original label
        rotated_stack.append([rotated, label])

    # Update the polygon stack
    poly_shape.polygon_stack = rotated_stack

    return degrees  # Return new total rotation angle
