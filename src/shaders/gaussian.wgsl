struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};


@vertex
fn vs_main(
) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(1. ,1. , 0., 1.);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.);
}