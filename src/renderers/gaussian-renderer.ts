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

    const null_buffer = createBuffer(
        device,
        'null buffer',
        4,
        GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        nulling_data
    );

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
        pc.num_points * 24,
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
            { binding: 2, resource: { buffer: pc.sh_buffer } },
            { binding: 3, resource: { buffer: render_settings_buffer } },
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
                        srcFactor: 'src-alpha',
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
    });


    const render_splat_bind_group = device.createBindGroup({
        label: 'render splats',
        layout: render_pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: splat_buffer } },
            { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
            { binding: 2, resource: { buffer: camera_buffer } },
        ],
    });

    // ===============================================
    //    Command Encoder Functions
    // ===============================================

    const preprocess = (encoder: GPUCommandEncoder) => {
        encoder.copyBufferToBuffer(null_buffer, 0, sorter.sort_info_buffer, 0, 4);
        encoder.copyBufferToBuffer(null_buffer, 0, sorter.sort_dispatch_indirect_buffer, 0, 4);

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
    };

    const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
        const pass = encoder.beginRenderPass({
            label: 'gaussian render',
            colorAttachments: [{
                view: texture_view,
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            }],
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

            encoder.copyBufferToBuffer(
                sorter.sort_info_buffer, 0,
                indirect_draw_buffer, 4,
                4
            );

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
