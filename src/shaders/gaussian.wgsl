struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

struct Splat {
    xy_x: u32,  // packed f16: xy position (x, y) in NDC
    xy_y: u32,  // packed f16: quad size (width, height) in NDC
};

@group(0) @binding(0)
var<storage, read> splats: array<Splat>;

@group(0) @binding(1)
var<storage, read> sort_indices: array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_idx: u32,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    var out: VertexOutput;
    
    let splat_idx = sort_indices[instance_idx];
    let splat = splats[splat_idx];
    let center_ndc = unpack2x16float(splat.xy_x);
    let quad_size = unpack2x16float(splat.xy_y);
    
    // Generate quad vertices 
    // Triangle 1: 0,1,2  Triangle 2: 0,2,3
    // 0: bottom-left, 1: bottom-right, 2: top-right, 3: top-left
    var offset = vec2<f32>(0.0, 0.0);
    switch (vertex_idx) {
        case 0u, 3u: { offset = vec2<f32>(-0.5, -0.5); } // bottom-left
        case 1u: { offset = vec2<f32>(0.5, -0.5); }      // bottom-right
        case 2u, 4u: { offset = vec2<f32>(0.5, 0.5); }   // top-right
        case 5u: { offset = vec2<f32>(-0.5, 0.5); }      // top-left
        default: {}
    }
    
    let pos_ndc = center_ndc + offset * quad_size;
    out.position = vec4<f32>(pos_ndc, 0.0, 1.0);
    out.color = vec4<f32>(quad_size.x * 10.0, quad_size.y * 10.0, 0.0, 1.0);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}