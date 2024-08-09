mod algorithm;
mod distance;
#[cfg(test)]
mod test;
mod utils;

use wasm_bindgen::prelude::*;
pub use wasm_bindgen_rayon::init_thread_pool;

use crate::algorithm::TsneEncoder;
use crate::utils::set_panic_hook;

#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
pub struct Tsne {
    tsne_encoder: TsneEncoder,
    result: Vec<f64>,
}

#[wasm_bindgen]
impl Tsne {
    #[wasm_bindgen(constructor)]
    pub fn new(data: &[f64], cols: usize) -> Tsne {
        set_panic_hook();
        // console::log_1(&"Innnnn....".into());
        let rows = data.len() / cols;
        // console::log_1(&"Instantiating t-SNE encoder...".into());
        let mut tsne_encoder = TsneEncoder::new(rows, cols);
        // console::log_1(&"Initializing t-SNE encoder...".into());
        tsne_encoder.init(data);

        Tsne {
            tsne_encoder,
            result: vec![0.0; data.len() * 2],
        }
    }

    pub fn step(&mut self) {
        self.tsne_encoder.next();
    }

    pub fn embedding(&mut self) -> *const f64 {
        let mut i = 0;
        for row in self.tsne_encoder.y.rows() {
            for el in row.iter() {
                self.result[i] = *el;
                i += 1;
            }
        }
        self.result.as_ptr()
    }
}
