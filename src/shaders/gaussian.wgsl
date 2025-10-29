struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) conic: vec3<f32>,
    @location(2) center_ndc: vec2<f32>,
};

struct Splat {
    xy_x: u32,          
    xy_y: u32,           
    color: u32,         
    color_ba: u32,       
    conic_xy: u32,       
    conic_z_radius: u32,
};

@group(0) @binding(0)
var<storage, read> splats: array<Splat>;

@group(0) @binding(1)
var<storage, read> sort_indices: array<u32>;

@group(0) @binding(2)
var<uniform> camera: CameraUniforms;

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
    
    // Unpack color from spherical harmonics
    let color_rg = unpack2x16float(splat.color);
    let color_ba = unpack2x16float(splat.color_ba);
    let color = vec4<f32>(color_rg.r, color_rg.g, color_ba.r, color_ba.g);
    
    // Unpack conic matrix 
    let conic_xy = unpack2x16float(splat.conic_xy);
    let conic_z_radius = unpack2x16float(splat.conic_z_radius);
    let conic = vec3<f32>(conic_xy.x, conic_xy.y, conic_z_radius.x);
    
    // Generate quad vertices 
    // Triangle 1: 0,1,2  Triangle 2: 0,2,3
    // 0: bottom-left, 1: bottom-right, 2: top-right, 3: top-left
    var offset = vec2<f32>(0.0, 0.0);
    switch (vertex_idx) {
        case 0u, 3u: { offset = vec2<f32>(-1.0, -1.0); } // bottom-left
        case 1u: { offset = vec2<f32>(1.0, -1.0); }      // bottom-right
        case 2u, 4u: { offset = vec2<f32>(1.0, 1.0); }   // top-right
        case 5u: { offset = vec2<f32>(-1.0, 1.0); }      // top-left
        default: {}
    }
    
    let pos_ndc = center_ndc + offset * quad_size;
    out.position = vec4<f32>(pos_ndc, 0.0, 1.0);
    out.color = color;
    out.conic = conic;
    out.center_ndc = center_ndc;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let center_pixel = vec2<f32>(
        (in.center_ndc.x + 1.0) * 0.5 * camera.viewport.x,
        (1.0 - in.center_ndc.y) * 0.5 * camera.viewport.y  
    );
    
    let d = in.position.xy - center_pixel;
    
    let mahal_dist_sq = (
        in.conic.x * d.x * d.x +
        in.conic.z * d.y * d.y +
        2.0 * in.conic.y * d.x * d.y
    );
    
    let power = -0.5 * mahal_dist_sq;
    
    // Early discard for fragments too far from center
    if (power < -3.125) {
        discard;
    }
    
    let alpha = min(0.99, in.color.a * exp(power));
    
    if (alpha < 0.005) {
        discard;
    }
    
    return vec4<f32>(in.color.rgb * alpha, alpha);
}