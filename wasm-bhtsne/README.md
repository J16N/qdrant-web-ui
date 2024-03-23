# WASM SUPPORT FOR BHTSNE

This is a WebAssembly (WASM) port of the Barnes-Hut t-SNE algorithm. The original Rust implementation can be found [here](https://github.com/frjnn/bhtsne). Make sure `wasm-pack` is installed on your system. If not, you can install it by running:

```bash
cargo install wasm-pack
```

To build the WASM module, run:

```bash
wasm-pack build --target web
```

This will generate a `pkg` directory containing the WASM module and a JavaScript wrapper.
