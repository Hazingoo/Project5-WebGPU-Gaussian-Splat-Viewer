const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    xy_x: u32,        
    xy_y: u32,
    color: u32,
    color_ba: u32,
    conic_xy: u32,    
    conic_z_radius: u32, 
};

// Bind group 0: Camera
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

// Bind group 1: Gaussian data
@group(1) @binding(0)
var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1)
var<storage, read_write> splats: array<Splat>;
@group(1) @binding(2)
var<storage, read> sh_coeffs: array<u32>;
@group(1) @binding(3)
var<uniform> render_settings: RenderSettings;

// Bind group 2: Sort data
@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let base_f16_idx = splat_idx * 48u;
    
    let coef_f16_idx = base_f16_idx + c_idx * 3u;
    
    let u32_idx = coef_f16_idx / 2u;
    
    let rg_packed = unpack2x16float(sh_coeffs[u32_idx]);      
    let b_next_packed = unpack2x16float(sh_coeffs[u32_idx + 1u]); 
    
    let is_even = (coef_f16_idx % 2u) == 0u;
    
    let r = select(rg_packed.y, rg_packed.x, is_even);
    let g = select(b_next_packed.x, rg_packed.y, is_even);
    let b = select(b_next_packed.y, b_next_packed.x, is_even);
    
    return vec3<f32>(r, g, b);
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return clamp(result, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn unpack4x16float(a: u32, b: u32) -> vec4<f32> {
    let xy = unpack2x16float(a);
    let zw = unpack2x16float(b);
    return vec4<f32>(xy.x, xy.y, zw.x, zw.y);
}

// Build rotation matrix from quaternion
fn quat_to_mat(q: vec4<f32>) -> mat3x3<f32> {
    // Normalize quaternion
    let qn = normalize(q);
    let x = qn.x;
    let y = qn.y;
    let z = qn.z;
    let w = qn.w;
    
    // Compute rotation matrix from quaternion
    return mat3x3<f32>(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
        2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
        2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)
    );
}

fn compute_cov3d(scale: vec3<f32>, rot: vec4<f32>, gaussian_scaling: f32) -> mat3x3<f32> {
    // Scale with user multiplier
    let s = scale * gaussian_scaling;
    
    // Build rotation matrix
    let R = quat_to_mat(rot);
    
    let S = mat3x3<f32>(
        vec3<f32>(s.x, 0.0, 0.0),
        vec3<f32>(0.0, s.y, 0.0),
        vec3<f32>(0.0, 0.0, s.z)
    );
    
    let M = S * R;
    let Sigma = transpose(M) * M;
    
    return Sigma;
}

// Compute 2D covariance from 3D covariance
// Based on EWA splatting: https://github.com/kwea123/gaussian_splatting_notes
fn compute_cov2d(
    pos_view: vec3<f32>,
    cov3d_world: mat3x3<f32>,
    view_matrix: mat3x3<f32>,
    focal: vec2<f32>
) -> vec3<f32> {
    // Transform 3D covariance to view space
    let cov3d_view = view_matrix * cov3d_world * transpose(view_matrix);
    
    let t = pos_view;
    let limx = 1.3 * camera.viewport.x;
    let limy = 1.3 * camera.viewport.y;
    let txtz = t.x / t.z;
    let tytz = t.y / t.z;
    
    // Jacobian of perspective projection
    let J = mat3x3<f32>(
        focal.x / t.z, 0.0, -(focal.x * txtz) / t.z,
        0.0, focal.y / t.z, -(focal.y * tytz) / t.z,
        0.0, 0.0, 0.0
    );
    
    // Project to 2D: J * Cov3D_view * J^T
    let T = J * cov3d_view;
    let cov2d = mat3x3<f32>(
        T[0][0] * J[0][0] + T[0][1] * J[0][1] + T[0][2] * J[0][2],
        T[0][0] * J[1][0] + T[0][1] * J[1][1] + T[0][2] * J[1][2],
        0.0,
        T[1][0] * J[0][0] + T[1][1] * J[0][1] + T[1][2] * J[0][2],
        T[1][0] * J[1][0] + T[1][1] * J[1][1] + T[1][2] * J[1][2],
        0.0,
        0.0, 0.0, 0.0
    );
    
    let cov_a = cov2d[0][0] + 0.3;
    let cov_b = cov2d[0][1];
    let cov_c = cov2d[1][1] + 0.3;
    
    return vec3<f32>(cov_a, cov_b, cov_c);
}

fn compute_radius(cov2d: vec3<f32>) -> f32 {
    let a = cov2d.x;
    let b = cov2d.y;
    let c = cov2d.z;
    
    // Calculate determinant
    let det = max(0.0, a * c - b * b);
    
    // Compute eigenvalues 
    let mid = 0.5 * (a + c);
    let discriminant = max(0.0, mid * mid - det);
    let lambda1 = mid + sqrt(discriminant);
    let lambda2 = mid - sqrt(discriminant);
    
    // Radius is 3 sigma 
    let max_lambda = max(lambda1, lambda2);
    return ceil(3.0 * sqrt(max(0.1, max_lambda)));
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    
    // Check bounds
    if (idx >= arrayLength(&gaussians)) {
        return;
    }
    
    // Read gaussian data
    let gaussian = gaussians[idx];
    
    // Unpack position
    let a = unpack2x16float(gaussian.pos_opacity[0]);
    let b = unpack2x16float(gaussian.pos_opacity[1]);
    let pos_world = vec4<f32>(a.x, a.y, b.x, 1.0);
    
    let rot_WX = unpack2x16float(gaussian.rot[0]); // W, X
    let rot_YZ = unpack2x16float(gaussian.rot[1]); // Y, Z
    let rotation = vec4<f32>(rot_WX.y, rot_YZ.x, rot_YZ.y, rot_WX.x); // Reorder to X, Y, Z, W
    
    let scale_packed = unpack2x16float(gaussian.scale[0]);
    let scale_z = unpack2x16float(gaussian.scale[1]).x;
    let scale = vec3<f32>(exp(scale_packed.x), exp(scale_packed.y), exp(scale_z));
    
    // Transform to view space
    let pos_view = camera.view * pos_world;
    let pos_clip = camera.proj * pos_view;
    let pos_ndc = pos_clip.xy / pos_clip.w;
    
    let culling_bounds = 1.2;
    if (abs(pos_ndc.x) > culling_bounds || abs(pos_ndc.y) > culling_bounds || pos_clip.w <= 0.001) {
        // Outside frustum or behind camera, skip this Gaussian
        return;
    }
    
    // Compute 3D covariance in world space
    let cov3d_world = compute_cov3d(scale, rotation, render_settings.gaussian_scaling);
    
    // Build Jacobian of perspective projection
    let t = pos_view.xyz;
    let J = mat3x3<f32>(
        camera.focal.x / t.z, 0.0, -(camera.focal.x * t.x) / (t.z * t.z),
        0.0, camera.focal.y / t.z, -(camera.focal.y * t.y) / (t.z * t.z),
        0.0, 0.0, 0.0
    );
    
    // Extract view matrix rotation 
    let W = transpose(mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz
    ));
    
    // Combine transformations
    let T = W * J;
    
    let V = mat3x3<f32>(
        cov3d_world[0][0], cov3d_world[0][1], cov3d_world[0][2],
        cov3d_world[0][1], cov3d_world[1][1], cov3d_world[1][2],
        cov3d_world[0][2], cov3d_world[1][2], cov3d_world[2][2]
    );
    
    var cov2d_mat = transpose(T) * transpose(V) * T;
    cov2d_mat[0][0] += 0.3;
    cov2d_mat[1][1] += 0.3;
    
    let cov2d = vec3<f32>(
        cov2d_mat[0][0],
        cov2d_mat[0][1],
        cov2d_mat[1][1]
    );
    
    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    
    if (det <= 0.000001) {
        return;
    }
    
    // Compute conic 
    let det_inv = 1.0 / det;
    let conic = vec3<f32>(
        cov2d.z * det_inv,  
        -cov2d.y * det_inv, 
        cov2d.x * det_inv   
    );
    
    if (abs(conic.x) > 10000.0 || abs(conic.y) > 10000.0 || abs(conic.z) > 10000.0) {
        return;
    }
    
    let radius_pixels = compute_radius(cov2d);
    
    let quad_size_ndc = vec2<f32>(
        radius_pixels / camera.viewport.x,
        radius_pixels / camera.viewport.y
    );
    
    // Compute color from spherical harmonics
    let cam_pos = camera.view_inv[3].xyz;
    let view_dir = normalize(vec3<f32>(pos_world.x, pos_world.y, pos_world.z) - cam_pos);
    let color = computeColorFromSH(view_dir, idx, u32(render_settings.sh_deg));
    
    // Unpack opacity 
    let opacity_raw = b.y;
    let opacity = clamp(1.0 / (1.0 + exp(-opacity_raw)), 0.0, 0.99);
    
    if (opacity < 0.01) {
        return;
    }
    
    // Increment visible counter for this Gaussian 
    let visible_idx = atomicAdd(&sort_infos.keys_size, 1u);
    
    // Store in splat buffer at compacted visible index
    splats[visible_idx].xy_x = pack2x16float(pos_ndc);
    splats[visible_idx].xy_y = pack2x16float(quad_size_ndc);
    splats[visible_idx].color = pack2x16float(vec2<f32>(color.r, color.g));
    splats[visible_idx].color_ba = pack2x16float(vec2<f32>(color.b, opacity));
    splats[visible_idx].conic_xy = pack2x16float(vec2<f32>(conic.x, conic.y));
    splats[visible_idx].conic_z_radius = pack2x16float(vec2<f32>(conic.z, radius_pixels));

    let depth_uint = bitcast<u32>(-pos_view.z);
    let is_negative = (depth_uint & 0x80000000u) != 0u;
    let flipped = select(depth_uint ^ 0x80000000u, ~depth_uint, is_negative);
    sort_depths[visible_idx] = flipped;
    sort_indices[visible_idx] = visible_idx;

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    let new_count = visible_idx + 1u;
    if (new_count % keys_per_dispatch == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}