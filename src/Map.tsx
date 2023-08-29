import mapboxgl from "mapbox-gl";
import React from "react";
import { useEffect, useRef, useState } from "react";
import * as turf from "@turf/turf";
import MapboxGeocoder from "@mapbox/mapbox-gl-geocoder";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";
import ModelLayer from "./ModelLayer";

import geo from "./geo.json";
import hausmodel from "./haus2.gltf";
import createGLTFObject from "./gltf_create";
// import saveGLTF from "./gltf_save";

import "@mapbox/mapbox-gl-geocoder/dist/mapbox-gl-geocoder.css";
import "mapbox-gl/dist/mapbox-gl.css";

mapboxgl.accessToken =
	"pk.eyJ1IjoibWltaW8iLCJhIjoiY2l6ZjJoenBvMDA4eDJxbWVkd2IzZjR0ZCJ9.ppwGNP_-LS2K4jUvgXG2pA";

var clickedBuilding: mapboxgl.MapboxGeoJSONFeature[] = [];

interface DisplayFeature {
	[key: string]: any;
}
const displayFeat: DisplayFeature = {};

function convertGLTFToMesh(data) {
	const geometry = new THREE.BufferGeometry();

	const vertices = new Float32Array(
		data.meshes[0].primitives[0].attributes.POSITION
	);
	geometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));

	const indices = new Uint32Array(data.meshes[0].primitives[0].indices);
	geometry.setIndex(new THREE.BufferAttribute(indices, 1));

	const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 }); // Example green color
	return new THREE.Mesh(geometry, material);
}
function mergePolygons(
	polygons: mapboxgl.MapboxGeoJSONFeature[]
): turf.AllGeoJSON {
	let combinedPolygon: turf.GeoJSONObject;

	polygons.forEach((building) => {
		if (!combinedPolygon) {
			combinedPolygon = building;
		} else if (
			building.type === "Feature" &&
			building.geometry.type === "Polygon"
		) {
			combinedPolygon = turf.union(combinedPolygon, building.geometry);
		}
	});

	return combinedPolygon;
}

function calculateCircumference(polygonWithHoles: any): number {
	let totalCircumference = 0;
	// Calculate perimeter of the outer boundary
	const outerCoordinates = polygonWithHoles.geometry.coordinates[0];
	if (outerCoordinates.length < 3) {
		return 0;
	}
	if (outerCoordinates.length < 5) {
		const outerPolygon = turf.multiLineString([outerCoordinates]);
		totalCircumference += turf.lineDistance(outerPolygon, { units: "meters" });
	} else {
		const outerPolygon = turf.polygon([outerCoordinates]);
		totalCircumference += turf.lineDistance(outerPolygon, { units: "meters" });
	}

	// Calculate perimeter of any holes
	const numHoles = polygonWithHoles.geometry.coordinates.length - 1;

	for (let i = 0; i < numHoles; i++) {
		const holeCoordinates = polygonWithHoles.geometry.coordinates[i + 1];
		const holePolygon = turf.polygon([holeCoordinates]);
		totalCircumference += turf.lineDistance(holePolygon, { units: "meters" });
	}

	return totalCircumference;
}

let cmdKeyPressed = false;

document.addEventListener("keydown", (e) => {
	if (e.metaKey) {
		cmdKeyPressed = !cmdKeyPressed;
		console.log(cmdKeyPressed);
	}
});

// document.addEventListener("keyup", (e) => {
// 	if (e.metaKey) {
// 		cmdKeyPressed = false;
// 		console.log(cmdKeyPressed);
// 	}
// });

function Map() {
	const ref = useRef<HTMLDivElement | null>(null);
	const [, setMap] = useState<mapboxgl.Map | null>(null);
	const [loc, setLoc] = React.useState<[number, number]>([
		8.678813326977938, 50.10554171813183,
	]);
	const [roof, setRoof] = React.useState<string>("0");
	const [height, setHeight] = React.useState<string>("0");
	const [fassade, setFassade] = React.useState<string>("0");
	const [address, setAddress] = React.useState<string>("");
	const [gltfURL, setGltfURL] = useState<string | null>(hausmodel);
	let modelLayer = new ModelLayer({
		id: "layer-3d1",
		url: gltfURL,
		origin: loc,
		rotateY: 1,
		scale: 4.8,
	});
	useEffect(() => {
		if (ref?.current && typeof ref?.current !== undefined) {
			const map = new mapboxgl.Map({
				container: "map-container",
				center: loc,
				zoom: 17,
				pitch: 60,
				style: "mapbox://styles/mapbox/dark-v11",
			});

			const geocoder = new MapboxGeocoder({
				accessToken: mapboxgl.accessToken,
				mapboxgl: mapboxgl,
				marker: true,
				placeholder: "Enter an address or location",
				//flyTo: { duration: 2 },
			});

			map.addControl(geocoder);

			geocoder.on("result", function (e) {
				setLoc(e.result.geometry.coordinates);
				map.jumpTo({ center: e.result.geometry.coordinates });

				setTimeout(() => {
					const simulatedClickEvent = {
						lngLat: {
							lng: e.result.geometry.coordinates[0],
							lat: e.result.geometry.coordinates[1],
						},
						point: map.project([
							e.result.geometry.coordinates[0],
							e.result.geometry.coordinates[1],
						]),
						originalEvent: {}, // Any other properties you need
					};

					map.fire("click", simulatedClickEvent);
				}, 200);

				var initBuilding: any[] = [];
				var rendered_buildings = map.queryRenderedFeatures();
				for (let i = 0; i < rendered_buildings.length; i++) {
					if (
						(rendered_buildings[i].geometry.type === "MultiLineString" ||
							rendered_buildings[i].geometry.type === "MultiPolygon" ||
							rendered_buildings[i].geometry.type === "Polygon") &&
						rendered_buildings[i].sourceLayer === "building"
					) {
						if (
							turf.inside(
								turf.point([e.lngLat.lng, e.lngLat.lat]),
								rendered_buildings[i].geometry as turf.Polygon
							)
						) {
							initBuilding.push(rendered_buildings[i]);
						}
					}
				}
				if (initBuilding.length > 0) {
					const result = mergePolygons(initBuilding);
					const unifiedPolygon = result as turf.Feature<
						turf.Polygon | turf.MultiPolygon
					>;

					let totalHeight = 0;
					initBuilding.forEach((building) => {
						totalHeight += building.properties.height || 0; // Assuming height is stored directly on the properties.
					});

					const averageHeight = totalHeight / initBuilding.length;

					if (unifiedPolygon.properties) {
						unifiedPolygon.properties.height = averageHeight;
						unifiedPolygon.properties.min_height = 0;
					}

					var a = turf.area(unifiedPolygon);

					setRoof(a.toFixed(1));

					if (unifiedPolygon.properties) {
						const umfang = calculateCircumference(unifiedPolygon);
						setHeight(unifiedPolygon.properties.height.toFixed(1));
						setFassade((umfang * unifiedPolygon.properties.height).toFixed(1));
					}

					const source: mapboxgl.GeoJSONSource = map.getSource(
						"currentBuildings"
					) as mapboxgl.GeoJSONSource;
					source.setData({
						type: "FeatureCollection",
						features: [unifiedPolygon],
					});

					const source3d: mapboxgl.GeoJSONSource = map.getSource(
						"currentBuildings3d"
					) as mapboxgl.GeoJSONSource;
					source3d.setData({
						type: "FeatureCollection",
						features: initBuilding,
					});
				}
			});

			//new mapboxgl.Marker({ color: "blue" }).setLngLat(loc).addTo(map);

			map.on("load", () => {
				map.addControl(new mapboxgl.NavigationControl());
				// if (gltfURL != null) {
				// 	console.log("GLTF URL:" + gltfURL);
				// 	map.addLayer(modelLayer);
				// }
				// map.addLayer(
				// 	new ModelLayer({
				// 		id: "layer-3d",
				// 		url: hausmodel,
				// 		origin: [loc[0], loc[1]],
				// 		rotateY: 1,
				// 		scale: 28,
				// 	})
				// );
				const layers: mapboxgl.Layer[] =
					(map.getStyle().layers as mapboxgl.Layer[]) || [];

				var labelLayerId;
				for (var i = 0; i < layers.length; i++) {
					if (layers[i].type === "symbol") {
						//&& layers[i].layout["text-field"]) {
						labelLayerId = layers[i].id;
						break;
					}
				}

				map.addSource("buildings", {
					type: "geojson",
					data: geo,
				});

				map.addLayer({
					id: "building-layer",
					type: "line",
					source: "buildings",
					paint: {
						"line-width": 1,
						"line-color": "#ff0000",
					},
				});

				map.addSource("init", {
					type: "geojson",
					data: {
						type: "FeatureCollection",
						features: [],
					},
				});
				map.addSource("currentBuildings", {
					type: "geojson",
					data: {
						type: "FeatureCollection",
						features: [],
					},
				});
				map.addSource("currentBuildings3d", {
					type: "geojson",
					data: {
						type: "FeatureCollection",
						features: [],
					},
				});

				map.addLayer(
					{
						id: "3d-buildings",
						source: "composite",
						"source-layer": "building",
						filter: ["all", ["==", "extrude", "true"]],
						type: "fill-extrusion",
						minzoom: 15,
						paint: {
							"fill-extrusion-color": "#aaa",

							// Use an 'interpolate' expression to
							// add a smooth transition effect to
							// the buildings as the user zooms in.
							"fill-extrusion-height": [
								"interpolate",
								["linear"],
								["zoom"],
								15,
								0,
								15.05,
								["get", "height"],
							],
							"fill-extrusion-base": [
								"interpolate",
								["linear"],
								["zoom"],
								15,
								0,
								15.05,
								["get", "min_height"],
							],
							"fill-extrusion-opacity": 0.0, // TODO
						},
					},
					labelLayerId
				);
				map.addLayer(
					{
						id: "3d-buildings-line",
						source: "composite",
						"source-layer": "building",
						type: "line",
						minzoom: 15,
						paint: {
							"line-color": "#aaa",
							"line-width": 1,
						},
					},
					labelLayerId
				);
				map.addLayer(
					{
						id: "init",
						source: "init",
						type: "line",
						minzoom: 15,
						paint: {
							"line-color": "blue",
							"line-width": 2,
						},
					},
					labelLayerId
				);

				map.addLayer(
					{
						id: "highlight",
						source: "currentBuildings",
						type: "line",
						minzoom: 15,
						paint: {
							"line-color": "#FFC300",
							"line-width": 2,
						},
					},
					labelLayerId
				);

				map.addLayer(
					{
						id: "3d-buildings2",
						source: "currentBuildings3d",
						//"source-layer": "building",
						//filter: ["==", "extrude", "true"],
						type: "fill-extrusion",
						minzoom: 15,
						paint: {
							"fill-extrusion-color": "#aaa",

							// Use an 'interpolate' expression to
							// add a smooth transition effect to
							// the buildings as the user zooms in.
							"fill-extrusion-height": [
								"interpolate",
								["linear"],
								["zoom"],
								15,
								0,
								15.05,
								["get", "height"],
							],
							"fill-extrusion-base": [
								"interpolate",
								["linear"],
								["zoom"],
								15,
								0,
								15.05,
								["get", "min_height"],
							],
							"fill-extrusion-opacity": 0.0,
						},
					},
					labelLayerId
				);
			});

			map.on("click", "3d-buildings", async function (e) {
				if (!cmdKeyPressed) {
					clickedBuilding = [];
				}
				fetch(
					`https://api.mapbox.com/geocoding/v5/mapbox.places/${e.lngLat.lng},${e.lngLat.lat}.json?access_token=${mapboxgl.accessToken}`
				)
					.then((response) => response.json())
					.then((data) => {
						if (data.features && data.features.length > 0) {
							const placeName = data.features[0].place_name;
							if (cmdKeyPressed) {
								setAddress("Multi-Building-Select");
							} else {
								setAddress(placeName);
							}
						} else {
							console.log("No address found for this location");
						}
					})
					.catch((error) => {
						console.error("Error fetching geocoding data:", error);
					});
				var rendered_buildings = map.queryRenderedFeatures();

				for (let i = 0; i < rendered_buildings.length; i++) {
					if (
						(rendered_buildings[i].geometry.type === "MultiLineString" ||
							rendered_buildings[i].geometry.type === "MultiPolygon" ||
							rendered_buildings[i].geometry.type === "Polygon") &&
						rendered_buildings[i].sourceLayer === "building"
					) {
						if (
							turf.inside(
								turf.point([e.lngLat.lng, e.lngLat.lat]),
								rendered_buildings[i].geometry as turf.Polygon
							)
						) {
							clickedBuilding.push(rendered_buildings[i]);
						}
					}
				}

				if (clickedBuilding.length > 0) {
					for (let i = 0; i < rendered_buildings.length; i++) {
						if (clickedBuilding[0].id === rendered_buildings[i].id) {
							clickedBuilding.push(rendered_buildings[i]);
						}
					}

					const result = mergePolygons(clickedBuilding);
					const unifiedPolygon = result as turf.Feature<
						turf.Polygon | turf.MultiPolygon
					>;

					let totalHeight = 0;
					clickedBuilding.forEach((building) => {
						totalHeight += building.properties.height || 0; // Assuming height is stored directly on the properties.
					});

					const averageHeight = totalHeight / clickedBuilding.length;

					if (unifiedPolygon.properties) {
						unifiedPolygon.properties.height = averageHeight;
						unifiedPolygon.properties.min_height = 0;
					}

					var a = turf.area(unifiedPolygon);

					setRoof(a.toFixed(1));

					if (unifiedPolygon.properties) {
						const umfang = calculateCircumference(unifiedPolygon);
						setHeight(unifiedPolygon.properties.height.toFixed(1));
						setFassade((umfang * unifiedPolygon.properties.height).toFixed(1));
					}

					const source: mapboxgl.GeoJSONSource = map.getSource(
						"currentBuildings"
					) as mapboxgl.GeoJSONSource;
					source.setData({
						type: "FeatureCollection",
						features: [unifiedPolygon],
					});

					const source3d: mapboxgl.GeoJSONSource = map.getSource(
						"currentBuildings3d"
					) as mapboxgl.GeoJSONSource;
					source3d.setData({
						type: "FeatureCollection",
						features: clickedBuilding,
					});
					const coords: Coordinate[] = [
						[0, 0],
						[1, 0],
						[1, 1],
						[0, 1],
					];

					const data = createGLTFObject(unifiedPolygon.geometry.coordinates);
					// const blob = new Blob([JSON.stringify(data)], {
					// 	type: "model/gltf+json",
					// });
					// const objectURL = URL.createObjectURL(blob);
					// console.log("Object Url" + objectURL);
					// setGltfURL(objectURL);
					//LTF(data);
					// map.addLayer(
					// 	new ModelLayer({
					// 		id: "layer-3d12",
					// 		url: "path_to_output.glb",
					// 		origin: [e.lngLat.lng, e.lngLat.lat],
					// 		rotateY: 1,
					// 		scale: 14.8,
					// 	})
					// );
					const fetchGLTF = async () => {
						try {
							const hi = await fetch(`http://127.0.0.1:8080/`);
							console.log(await hi.json());
							console.log(turf.center(unifiedPolygon.geometry));
							const response = await fetch(
								`http://127.0.0.1:8080/create_gltf?coords=${unifiedPolygon.geometry.coordinates}&center=${turf.centerOfMass(unifiedPolygon.geometry).geometry.coordinates}&height=${unifiedPolygon.properties.height}`
							);
							
							if (!response.ok) {
								throw new Error("Network response was not ok");
							}
							
							const blob = await response.blob();
							const objectURL = URL.createObjectURL(blob);
							
							if (map.getLayer("layer-3d")) {
								map.removeLayer("layer-3d");
							}
							
							map.addLayer(
								new ModelLayer({
									id: "layer-3d",
									url: objectURL,
									origin: [turf.centerOfMass(unifiedPolygon.geometry).geometry.coordinates[0],turf.centerOfMass(unifiedPolygon.geometry).geometry.coordinates[1]],
									//origin: [e.lngLat.lng, e.lngLat.lat],
									rotateY: 0,
									scale:2.48,
								})
							);
							URL.revokeObjectURL(objectURL);
						} catch (error) {
							console.error(
								"There was a problem with the fetch operation:",
								error
							);
						}
					};
					await fetchGLTF();
					console.log(unifiedPolygon.geometry.coordinates);
				}
			});
		}
	}, [ref]);
	return (
		<div style={{ width: "100vw", height: "100vh" }}>
			<div
				id="map-container"
				style={{ width: "100vw", height: "100vh" }}
				ref={ref}
			/>
			<div className="overlay-text">
				<div className="addressContainer">{address}</div>
				<div className="gridContainer">
					<div>
						<span className="value">
							{height} <span className="unit">m</span>
						</span>
						<span className="label">Height</span>
					</div>
					<div>
						<span className="value">
							{roof}
							<span className="unit">
								m<sup>2</sup>
							</span>
						</span>
						<span className="label">Roof</span>
					</div>
					<div>
						<span className="value">
							{roof}
							<span className="unit">
								m<sup>2</sup>
							</span>
						</span>
						<span className="label">Floor</span>
					</div>
					<div>
						<span className="value">
							{fassade}
							<span className="unit">
								m<sup>2</sup>
							</span>
						</span>
						<span className="label">Fassade</span>
					</div>
				</div>
			</div>
		</div>
	);
}

export default Map;
