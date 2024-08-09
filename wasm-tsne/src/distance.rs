use ndarray::ArrayView1;

pub struct Distance;

impl Distance {
    pub fn euclidean_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let diff = x - y;
                diff * diff
            })
            .sum::<f64>()
            .sqrt()
    }
}
