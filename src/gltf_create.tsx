type Coordinate = [number, number];

function createGLTFObject(coordinates: Coordinate[]): any {
	const height = 10;

	// Convert 2D coordinates to 3D vertices for the base and top of the prism
	let vertices: number[] = [];
	coordinates.forEach((coord) => {
		// base vertices (z = 0)
		vertices.push(coord[0], coord[1], 0);
		// top vertices (z = height)
		vertices.push(coord[0], coord[1], height);
	});

	// Create indices for the triangles
	let indices: number[] = [];
	for (let i = 0; i < coordinates.length - 1; i++) {
		// base triangle
		indices.push(2 * i, 2 * (i + 1), 2 * coordinates.length + 2 * i);
		// top triangle
		indices.push(
			2 * (i + 1),
			2 * coordinates.length + 2 * (i + 1),
			2 * coordinates.length + 2 * i
		);
	}

	// Add triangles for the last coordinate connecting to the first
	indices.push(2 * (coordinates.length - 1), 0, 2 * coordinates.length);
	indices.push(0, 2 * coordinates.length + 1, 2 * coordinates.length);

	const gltfObject = {
		asset: {
			version: "2.0",
		},
		meshes: [
			{
				primitives: [
					{
						attributes: {
							POSITION: vertices,
						},
						indices: indices,
						mode: 4, // TRIANGLES
					},
				],
			},
		],
	};

	return gltfObject;
}
export default createGLTFObject;
