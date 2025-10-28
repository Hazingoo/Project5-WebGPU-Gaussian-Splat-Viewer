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
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    return vec3<f32>(0.0);
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

    return  max(vec3<f32>(0.), result);
}

// Helper to unpack 4 f16 values from 2 u32
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
    let r00 = 1.0 - 2.0 * (y * y + z * z);
    let r01 = 2.0 * (x * y - w * z);
    let r02 = 2.0 * (x * z + w * y);
    
    let r10 = 2.0 * (x * y + w * z);
    let r11 = 1.0 - 2.0 * (x * x + z * z);
    let r12 = 2.0 * (y * z - w * x);
    
    let r20 = 2.0 * (x * z - w * y);
    let r21 = 2.0 * (y * z + w * x);
    let r22 = 1.0 - 2.0 * (x * x + y * y);
    
    return mat3x3<f32>(
        vec3<f32>(r00, r10, r20),
        vec3<f32>(r01, r11, r21),
        vec3<f32>(r02, r12, r22)
    );
}

fn compute_cov3d(scale: vec3<f32>, rot: vec4<f32>, gaussian_scaling: f32) -> mat3x3<f32> {
    // Scale with user multiplier
    let s = scale * gaussian_scaling;
    
    // Build rotation matrix
    let R = quat_to_mat(rot);
    
    // Build scale matrix S 
    let S = mat3x3<f32>(
        vec3<f32>(s.x * s.x, 0.0, 0.0),
        vec3<f32>(0.0, s.y * s.y, 0.0),
        vec3<f32>(0.0, 0.0, s.z * s.z)
    );
    
    // Compute covariance: R * S * R^T
    let M = R * S;
    let Sigma = M * transpose(R);
    
    return Sigma;
}

// Compute 2D covariance from 3D covariance
// https://github.com/kwea123/gaussian_splatting_notes
fn compute_cov2d(
    pos_view: vec3<f32>,
    cov3d: mat3x3<f32>,
    focal: vec2<f32>,
    viewport: vec2<f32>
) -> vec3<f32> {
    // Compute Jacobian of perspective projection
    let z = pos_view.z;
    let z2 = z * z;
    let fx = focal.x;
    let fy = focal.y;
    
    // Jacobian of projection
    let J = mat3x2<f32>(
        vec2<f32>(fx / z, 0.0),
        vec2<f32>(0.0, fy / z),
        vec2<f32>(-fx * pos_view.x / z2, -fy * pos_view.y / z2)
    );
    
    // Compute 2D covariance;
    let T = J * cov3d;  // 2x3 * 3x3 = 2x3
    let cov2d_mat = T * transpose(J);  // 2x3 * 3x2 = 2x2
    
    return vec3<f32>(cov2d_mat[0][0], cov2d_mat[0][1], cov2d_mat[1][1]);
}

// Compute radius from 2D covariance
fn compute_radius(cov2d: vec3<f32>) -> f32 {
    let a = cov2d.x;
    let b = cov2d.y;
    let c = cov2d.z;
    
    // Eigenvalues of 2x2 symmetric matrix:
    let mid = 0.5 * (a + c);
    let det = sqrt(max(0.0, 0.25 * (a - c) * (a - c) + b * b));
    let lambda1 = mid + det;
    let lambda2 = mid - det;
    
    // Maximum eigenvalue
    let max_eig = max(lambda1, lambda2);
    
    // Radius is 3 standard deviations
    return 3.0 * sqrt(max(0.0, max_eig));
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
    
    // Unpack rotation (quaternion)
    let rot_packed = unpack4x16float(gaussian.rot[0], gaussian.rot[1]);
    let rotation = vec4<f32>(rot_packed.x, rot_packed.y, rot_packed.z, rot_packed.w);
    
    // Unpack scale (in log space, need to exp)
    let scale_packed = unpack2x16float(gaussian.scale[0]);
    let scale_z = unpack2x16float(gaussian.scale[1]).x;
    let scale = vec3<f32>(exp(scale_packed.x), exp(scale_packed.y), exp(scale_z));
    
    // Transform to view space
    let pos_view = camera.view * pos_world;
    let pos_clip = camera.proj * pos_view;
    let pos_ndc = pos_clip.xy / pos_clip.w;
    
    // View-frustum culling 
    let culling_bounds = 1.2;
    if (abs(pos_ndc.x) > culling_bounds || abs(pos_ndc.y) > culling_bounds || pos_clip.w <= 0.0) {
        // Outside frustum, skip this Gaussian
        return;
    }
    
    // Compute 3D covariance in world space
    let cov3d_world = compute_cov3d(scale, rotation, render_settings.gaussian_scaling);
    
    // Transform covariance to view space
    // Σ_view = W * Σ_world * W^T where W is upper 3x3 of view matrix
    let W = mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz
    );
    let cov3d_view = W * cov3d_world * transpose(W);
    
    // Compute 2D covariance in screen space
    let cov2d = compute_cov2d(pos_view.xyz, cov3d_view, camera.focal, camera.viewport);
    
    // Compute radius in pixels
    let radius_pixels = compute_radius(cov2d);
    
    // Convert radius to NDC space 
    // NDC is [-1, 1], viewport is in pixels
    let radius_ndc = vec2<f32>(
        radius_pixels / camera.viewport.x * 2.0,
        radius_pixels / camera.viewport.y * 2.0
    );
    
    // Quad size is 2 * radius (diameter)
    let quad_size = radius_ndc * 2.0;
    
    // Store in splat buffer
    splats[idx].xy_x = pack2x16float(pos_ndc);
    splats[idx].xy_y = pack2x16float(quad_size);
    
    // Atomically increment the count of visible Gaussians
    let visible_idx = atomicAdd(&sort_infos.keys_size, 1u);

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
}