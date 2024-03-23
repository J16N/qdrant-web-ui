use num_traits::Float;
use std::iter::Sum;

pub enum DistanceFunctions {
    Euclidean,
}

impl DistanceFunctions {
    pub fn get_closure<T: Float + Sum<T>>(&self) -> fn(&[T], &[T]) -> T {
        match self {
            DistanceFunctions::Euclidean => |a: &[T], b: &[T]| {
                a.iter()
                    .zip(b.iter())
                    .map(|(&x, &y)| (x - y).powi(2))
                    .sum::<T>()
                    .sqrt()
            },
        }
    }
}
