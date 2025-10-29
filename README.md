# Project5-WebGPU-Gaussian-Splat-Viewer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 5**

* Harry Guan
* Tested on: (TODO) **Google Chrome 222.2** on
  Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### Live Demo

Link: https://hazingoo.github.io/Project5-WebGPU-Gaussian-Splat-Viewer/
[![](img/thumb.png)](http://TODO.github.io/Project4-WebGPU-Forward-Plus-and-Clustered-Deferred)

### Demo Video/GIF

[![](img/video.mp4)](TODO)

## Description
I implemented a 3D scene viewer that uses a technique based on the paper "3D Gaussian Splatting for Real-Time Radiance Field Rendering" to render beautiful, realistic scenes in your web browser. Unlike traditional 3D rendering that uses polygons, Gaussian Splatting represents every piece of a scene as a cloud of many tiny, colored points that blur together to create smooth surfaces and lighting effects. The viewer loads 3D scenes (saved as PLY files) and lets you navigate around them, rotating and zooming in real-time to explore objects like bonsai trees, bicycles. You can adjust how the points are displayed and switch between different visualization modes to see both the raw point cloud data and the final beautiful splat rendering. 


## Feature Overview

### Preprocessing Pipeline

Before rendering, the system processes all 3D Gaussian points to determine which ones are visible and how they should appear on screen. Starting with each point's position, rotation, and size, the code unpacks the data from a compact 16-bit format. The scale values, which are stored in logarithmic form, are converted back to actual sizes. The opacity values are also converted from their stored format to a usable range using a sigmoid function. Each point is then transformed from 3D world space into the camera's 2D view space using standard camera matrices.

The first optimization is view frustum culling, which removes any points that are outside the visible screen area. This check uses a slightly larger boundary (1.2 times the screen size) to ensure that points right on the edge don't get accidentally discarded. Only points that pass this check continue to the next steps.

For the remaining visible points, the system calculates how each 3D Gaussian blob should appear when projected onto the 2D screen. This involves computing the shape and size by combining the point's rotation and scale information. The mathematical approach projects the 3D shape through the camera lens to determine its elliptical appearance on screen, accounting for the viewing angle and distance. A small stability factor is added to prevent numerical issues.

The color of each point is computed based on the viewing direction using spherical harmonics, a mathematical technique that allows the appearance to change naturally as you look at it from different angles. This gives the scene realistic lighting that responds to your viewpoint. The opacity is also extracted and combined with the color.

Finally, all this information for each visible point is packed into a compact format and stored in memory. The system also prepares depth values for sorting, carefully handling the bit representation to ensure the sorting works correctly for both positive and negative depths. A counter tracks how many visible points there are, which is used later to determine how many quads to actually render.

### Rendering Pipeline

The rendering system draws each visible point as a small quad on the screen. The number of quads to draw is determined automatically by counting how many visible points there are from the preprocessing step. Each point's position, size, color, and shape information is unpacked from memory and used to position a quad centered at the point's 2D screen location. The quad is scaled to match the point's computed radius, and the color information is passed along for shading.

The fragment shader determines what each pixel of the quad should look like. It checks if the pixel is actually inside the elliptical shape of the Gaussian splat by computing its distance from the center and using a mathematical formula to determine if it falls within the boundary. Pixels outside the splat are discarded entirely. For pixels inside the splat, the opacity falls off smoothly as you move away from the center—this creates the soft, blurry edges that make the individual points blend together to form smooth surfaces. Very transparent pixels far from the center are also discarded since they contribute almost nothing to the final image.

All these semi-transparent splats are blended together using standard transparency blending, which combines the colors and opacities of overlapping splats in the correct way.


### Performance analysis

The point cloud

### Credits

- [Vite](https://vitejs.dev/)
- [tweakpane](https://tweakpane.github.io/docs//v3/monitor-bindings/)
- [stats.js](https://github.com/mrdoob/stats.js)
- [wgpu-matrix](https://github.com/greggman/wgpu-matrix)
- Special Thanks to: Shrek Shao (Google WebGPU team) & [Differential Guassian Renderer](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
