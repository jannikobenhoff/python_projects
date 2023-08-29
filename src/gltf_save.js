const { Document } = require("@gltf-transform/core");
const { Writer } = require("@gltf-transform/writer");

async function saveGLTF(data) {
	const doc = new Document();

	// Assuming `data` is your in-memory GLTF object.
	// For this example, we'll create a new mesh using data, but in a real-world scenario,
	// you'd likely want to traverse `data` to create a corresponding Document structure.
	const mesh = doc
		.createMesh("mesh")
		.addPrimitive(
			doc
				.createPrimitive()
				.setAttribute(
					"POSITION",
					doc
						.createAccessor()
						.setArray(
							new Float32Array(data.meshes[0].primitives[0].attributes.POSITION)
						)
						.setType("VEC3")
				)
		);

	doc.createNode().setMesh(mesh);

	const writer = new Writer();
	await writer.write(doc, { path: "path_to_output.glb" });
	console.log("WROTE");
}
export default saveGLTF;
