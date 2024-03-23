mod algorithm;
mod distance;
pub mod tsne;
mod utils;

// #[cfg(test)]
// mod test;

use algorithm::tSNE;
use distance::DistanceFunctions;
use wasm_bindgen::prelude::*;
pub use wasm_bindgen_rayon::init_thread_pool;

// A macro to provide `println!(..)`-style syntax for `console.log` logging.
macro_rules! log {
    ( $( $t:tt )* ) => {
        web_sys::console::log_1(&format!( $( $t )* ).into());
    }
}

#[wasm_bindgen]
#[allow(non_camel_case_types)]
pub struct tSNEf32(tSNE<f32, f32>);

#[wasm_bindgen]
impl tSNEf32 {
    pub fn new(data: &[f32], col: usize) -> Self {
        utils::set_panic_hook();
        let data = data.chunks(col).map(|x| x.to_vec()).collect();
        tSNEf32(tSNE::new(data))
    }

    pub fn embedding_dim(&mut self, dim: u8) {
        self.0.embedding_dim(dim);
    }

    pub fn epochs(&mut self, epochs: usize) {
        self.0.epochs(epochs);
    }

    pub fn perplexity(&mut self, perplexity: f32) {
        self.0.perplexity(perplexity);
    }

    pub fn barnes_hut(&mut self, theta: f32) -> Vec<f32> {
        self.0
            .barnes_hut(theta, DistanceFunctions::Euclidean.get_closure::<f32>());
        self.0.embedding()
    }

    pub fn exact(&mut self) -> Vec<f32> {
        self.0
            .exact(DistanceFunctions::Euclidean.get_closure::<f32>());
        self.0.embedding()
    }
}

#[wasm_bindgen]
#[allow(non_camel_case_types)]
pub struct tSNEf64(tSNE<f64, f64>);

#[wasm_bindgen]
impl tSNEf64 {
    pub fn new(data: &[f64], col: usize) -> Self {
        utils::set_panic_hook();
        let data = data.chunks(col).map(|x| x.to_vec()).collect();
        tSNEf64(tSNE::new(data))
    }

    pub fn embedding_dim(&mut self, dim: u8) {
        self.0.embedding_dim(dim);
    }

    pub fn epochs(&mut self, epochs: usize) {
        self.0.epochs(epochs);
    }

    pub fn perplexity(&mut self, perplexity: f64) {
        self.0.perplexity(perplexity);
    }

    pub fn barnes_hut(&mut self, theta: f64) -> Vec<f64> {
        self.0
            .barnes_hut(theta, DistanceFunctions::Euclidean.get_closure::<f64>());
        self.0.embedding()
    }

    pub fn exact(&mut self) -> Vec<f64> {
        self.0
            .exact(DistanceFunctions::Euclidean.get_closure::<f64>());
        self.0.embedding()
    }
}
