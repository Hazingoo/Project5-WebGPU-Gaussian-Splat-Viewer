import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter, c_histogram_block_rows, C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
    updateGaussianScaling: (scaling: number) => void;
}

// Utility to create GPU buffers
const createBuffer = (
    device: GPUDevice,
    label: string,
    size: number,
    usage: GPUBufferUsageFlags,
    data?: BufferSource
) => {
    const buffer = device.createBuffer({ label, size, usage });
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
};

export default function get_renderer(
    pc: PointCloud,
    device: GPUDevice,
    presentation_format: GPUTextureFormat,
    camera_buffer: GPUBuffer,
): GaussianRenderer {

    const sorter = get_sorter(pc.num_points, device);

    // ===============================================
    //            Initialize GPU Buffers
    // ===============================================

    const nulling_data = new Uint32Array([0]);

    // Indirect draw buffer: stores draw call parameters
    const indirect_draw_buffer = createBuffer(
        device,
        'indirect draw buffer',
        16, // 4 x u32 = 16 bytes
        GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        new Uint32Array([6, 0, 0, 0]) // 6 vertices per quad, 0 instance
    );

    // Splat buffer: stores processed 2D gaussian data from compute shader
    const splat_buffer = createBuffer(
        device,
        'splat buffer',
        pc.num_points * 32, // 32 bytes per splat (8 floats: pos xy, color rgb, conic abc)
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );

    const render_settings_buffer = createBuffer(
        device,
        'render settings',
        8,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        new Float32Array([1.0, pc.sh_deg])
    );

    // ===============================================
    //    Create Compute Pipeline and Bind Groups
    // ===============================================
    const preprocess_pipeline = device.createComputePipeline({
        label: 'preprocess',
        layout: 'auto',
        compute: {
            module: device.createShaderModule({ code: preprocessWGSL }),
            entryPoint: 'preprocess',
            constants: {
                workgroupSize: C.histogram_wg_size,
                sortKeyPerThread: c_histogram_block_rows,
            },
        },
    });

    const sort_bind_group = device.createBindGroup({
        label: 'sort',
        layout: preprocess_pipeline.getBindGroupLayout(2),
        entries: [
            { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
            { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
            { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
            { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
        ],
    });

    // Bind groups for preprocess compute shader
    const preprocess_camera_bind_group = device.createBindGroup({
        label: 'preprocess camera',
        layout: preprocess_pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: camera_buffer } }],
    });

    const preprocess_gaussian_bind_group = device.createBindGroup({
        label: 'preprocess gaussians',
        layout: preprocess_pipeline.getBindGroupLayout(1),
        entries: [
            { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
            { binding: 1, resource: { buffer: splat_buffer } },
            { binding: 2, resource: { buffer: render_settings_buffer } },
        ],
    });

    // ===============================================
    //    Create Render Pipeline and Bind Groups
    // ===============================================

    const render_shader = device.createShaderModule({ code: renderWGSL });

    const render_pipeline = device.createRenderPipeline({
        label: 'gaussian render',
        layout: 'auto',
        vertex: {
            module: render_shader,
            entryPoint: 'vs_main',
        },
        fragment: {
            module: render_shader,
            entryPoint: 'fs_main',
            targets: [{
                format: presentation_format,
                blend: {
                    color: {
                        srcFactor: 'one',
                        dstFactor: 'one-minus-src-alpha',
                        operation: 'add',
                    },
                    alpha: {
                        srcFactor: 'one',
                        dstFactor: 'one-minus-src-alpha',
                        operation: 'add',
                    },
                },
            }],
        },
        primitive: {
            topology: 'triangle-list',
        },
        depthStencil: {
            depthWriteEnabled: false,
            depthCompare: 'less-equal',
            format: 'depth24plus',
        },
    });


    const render_splat_bind_group = device.createBindGroup({
        label: 'render splats',
        layout: render_pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: splat_buffer } },
            { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
        ],
    });

    // Create depth texture for proper rendering order
    const canvas = document.querySelector('canvas');
    let depth_texture = device.createTexture({
        size: [canvas.width, canvas.height],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    let depth_texture_view = depth_texture.createView();

    // ===============================================
    //    Command Encoder Functions
    // ===============================================

    const preprocess = (encoder: GPUCommandEncoder) => {
        // Reset sort info
        device.queue.writeBuffer(sorter.sort_info_buffer, 0, nulling_data);
        device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, nulling_data);

        const pass = encoder.beginComputePass({ label: 'preprocess compute' });
        pass.setPipeline(preprocess_pipeline);
        pass.setBindGroup(0, preprocess_camera_bind_group);
        pass.setBindGroup(1, preprocess_gaussian_bind_group);
        pass.setBindGroup(2, sort_bind_group);

        // Dispatch one thread per gaussian
        const workgroup_size = C.histogram_wg_size;
        const workgroup_count = Math.ceil(pc.num_points / workgroup_size);
        pass.dispatchWorkgroups(workgroup_count);
        pass.end();

        // Copy visible count from sort_infos.keys_size to indirect draw buffer's instance count
        encoder.copyBufferToBuffer(
            sorter.sort_info_buffer, 0,  // source: keys_size at offset 0
            indirect_draw_buffer, 4,      // destination: instanceCount at offset 4
            4                              // size: 4 bytes (one u32)
        );
    };

    const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
        // Recreate depth texture if canvas size changed
        if (canvas.width !== depth_texture.width || canvas.height !== depth_texture.height) {
            depth_texture.destroy();
            depth_texture = device.createTexture({
                size: [canvas.width, canvas.height],
                format: 'depth24plus',
                usage: GPUTextureUsage.RENDER_ATTACHMENT,
            });
            depth_texture_view = depth_texture.createView();
        }

        const pass = encoder.beginRenderPass({
            label: 'gaussian render',
            colorAttachments: [{
                view: texture_view,
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            }],
            depthStencilAttachment: {
                view: depth_texture_view,
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        });

        pass.setPipeline(render_pipeline);

        pass.setBindGroup(0, render_splat_bind_group);

        // Use indirect draw 
        pass.drawIndirect(indirect_draw_buffer, 0);
        pass.end();
    };

    // ===============================================
    //    Return Render Object
    // ===============================================
    return {
        frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
            preprocess(encoder);
            sorter.sort(encoder);
            render(encoder, texture_view);
        },
        camera_buffer,
        updateGaussianScaling: (scaling: number) => {
            device.queue.writeBuffer(
                render_settings_buffer,
                0,
                new Float32Array([scaling])
            );
        },
    };
}
