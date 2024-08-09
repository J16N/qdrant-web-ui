use crate::distance::Distance;
use ndarray::parallel::prelude::*;
use ndarray::{s, Array1, Array2, Zip};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use rand::rngs::StdRng;
use rand::SeedableRng;

pub struct TsneEncoder {
    rows: usize,
    cols: usize,
    iter: usize,
    perplexity: i32,
    epsilon: f64,
    target_dim: usize,
    ystep: Array2<f64>,
    gains: Array2<f64>,
    distance: Array2<f64>,
    p: Array2<f64>,
    pu: Array2<f64>,
    q: Array2<f64>,
    pub y: Array2<f64>,
}

impl TsneEncoder {
    pub fn new(rows: usize, cols: usize) -> Self {
        let perplexity = 50;
        let epsilon = 10.0;
        let target_dim = 2;
        let ystep = Array2::zeros((rows, target_dim));
        let gains = Array2::ones((rows, target_dim));
        let p = Array2::zeros((rows, rows));
        let pu = Array2::zeros((rows, rows));
        let q = Array2::zeros((rows, rows));
        let mut rng = StdRng::seed_from_u64(1212);
        let y = Array2::random_using(
            (rows, target_dim),
            Normal::new(0.0, 1e-4).unwrap(),
            &mut rng,
        );
        let distance = Array2::zeros((rows, rows));

        Self {
            rows,
            cols,
            iter: 0,
            perplexity,
            epsilon,
            target_dim,
            ystep,
            gains,
            distance,
            p,
            pu,
            q,
            y,
        }
    }

    pub fn init(&mut self, data: &[f64]) {
        let h_target = f64::ln(self.perplexity as f64);
        let data = Array2::from_shape_vec((self.rows, self.cols), data.to_vec()).unwrap();

        // calculate distance matrix
        Zip::indexed(&mut self.distance).par_for_each(|(i, j), dist| {
            let d = Distance::euclidean_distance(&data.row(i), &data.row(j));
            *dist = d;
        });

        // search for fitting sigma
        let tol = 1e-4;
        let maxtries = 50;
        let mut buffer: Array2<f64> = Array2::zeros((self.rows, self.rows));

        Zip::indexed(self.pu.rows_mut())
            .and(self.distance.rows())
            .and(buffer.rows_mut())
            .par_for_each(|i, mut p_row, dist_row, mut buff_row| {
                let mut betamin = f64::NEG_INFINITY;
                let mut betamax = f64::INFINITY;
                let mut beta = 1.0;
                let mut cnt = maxtries;
                let mut done = false;
                let mut psum = 0.0;

                while !done && cnt > 0 {
                    // compute entropy and kernel with beta precision
                    cnt -= 1;

                    Zip::indexed(&mut p_row)
                        .and(&dist_row)
                        .and(&mut buff_row)
                        .par_for_each(|j, p, dist, buff| {
                            let pj = if i == j { 0.0 } else { f64::exp(-beta * *dist) };
                            *buff = *dist * pj;
                            *p = pj;
                        });

                    psum = p_row.sum();
                    let dp_sum = buff_row.sum();

                    // compute entropy
                    let h = if psum > 0.0 {
                        f64::ln(psum) + (beta * dp_sum) / psum
                    } else {
                        0.0
                    };

                    if h > h_target {
                        betamin = beta;
                        beta = if betamax == f64::INFINITY {
                            beta * 2.0
                        } else {
                            (beta + betamax) / 2.0
                        };
                    } else {
                        betamax = beta;
                        beta = if betamin == f64::NEG_INFINITY {
                            beta / 2.0
                        } else {
                            (beta + betamin) / 2.0
                        };
                    }
                    done = f64::abs(h - h_target) < tol;
                }

                // normalize p
                p_row.par_map_inplace(|p| *p /= psum);
            });

        // compute probabilities
        let n2 = (2 * self.rows) as f64;
        Zip::indexed(&mut self.p).par_for_each(|(i, j), prob| {
            *prob = f64::max((self.pu[[i, j]] + self.pu[[j, i]]) / n2, 1e-100);
        });
    }

    pub fn next(&mut self) {
        self.iter += 1;

        // calc cost gradient
        let pmul = if self.iter < 100 { 4.0 } else { 1.0 };

        // compute Q dist (unnormalized)
        // we are reusing pu matrix here
        let qu = &mut self.pu;
        Zip::indexed(&mut *qu).par_for_each(|(i, j), quu| {
            let mut dsum = 0.0;
            for d in 0..self.target_dim {
                let dhere = self.y[[i, d]] - self.y[[j, d]];
                dsum += dhere * dhere;
            }
            let qu = if i != j { 1.0 / (1.0 + dsum) } else { 0.0 };
            *quu = qu;
        });

        let qsum;
        {
            let mut buffer = self.distance.row_mut(0);
            Zip::indexed(&mut buffer).par_for_each(|i, b| {
                *b = qu.column(i).sum();
            });
            qsum = buffer.sum();
        }

        // normalize Q dist
        Zip::indexed(&mut self.q).par_for_each(|(i, j), q| {
            *q = f64::max(qu[[i, j]] / qsum, 1e-100);
            if i == j {
                *q = 0.0;
            }
        });

        // Reusing distance matrix for calculating gradient
        let mut grad;
        {
            grad = self.distance.slice_mut(s![.., ..self.target_dim]);

            Zip::indexed(&mut grad).par_for_each(|(i, d), gradval| {
                *gradval = 0.0;

                for j in 0..self.rows {
                    let premult = 4.0 * (pmul * self.p[[i, j]] - self.q[[i, j]]) * qu[[i, j]];
                    *gradval += (self.y[[i, d]] - self.y[[j, d]]) * premult;
                }
            });
        }

        // reuse pu for ymean
        let mut ymean = self.pu.slice_mut(s![.., ..self.target_dim]);

        Zip::from(&mut self.y)
            .and(&grad)
            .and(&mut self.ystep)
            .and(&mut self.gains)
            .and(&mut ymean)
            .par_for_each(|yid, gid, sid, gainid, mean| {
                let mut newgain = match (*gid, *sid) {
                    (x, y) if x < 0.0 && y < 0.0 => *gainid * 0.8,
                    (x, y) if x == 0.0 && y == 0.0 => *gainid * 0.8,
                    (x, y) if x > 0.0 && y > 0.0 => *gainid * 0.8,
                    (_, _) => *gainid + 0.2,
                };
                newgain = if newgain < 0.01 { 0.01 } else { newgain };
                *gainid = newgain;

                let momval = if self.iter < 250 { 0.5 } else { 0.8 };
                let newsid = momval * *sid - self.epsilon * newgain * *gid;
                *sid = newsid;
                *yid += newsid;
                *mean = *yid;
            });

        // reuse distance for total_ymean
        let mut buffer = self.distance.row_mut(0);
        let mut total_ymean = buffer.slice_mut(s![..self.target_dim]);
        Zip::indexed(&mut total_ymean).par_for_each(|d, mean| {
            *mean = ymean.column(d).sum() / (self.rows as f64);
        });

        Zip::indexed(&mut self.y).par_for_each(|(_, d), yval| {
            *yval -= total_ymean[d];
        });
    }

    pub fn initSync(&mut self, data: &[f64]) {
        let h_target = f64::ln(self.perplexity as f64);
        let data = Array2::from_shape_vec((self.rows, self.cols), data.to_vec()).unwrap();

        for i in 0..self.rows {
            let drow = data.row(i);
            for j in (i + 1)..self.rows {
                let dist = Distance::euclidean_distance(&drow, &data.row(j));
                self.distance[[i, j]] = dist;
                self.distance[[j, i]] = dist;
            }
        }

        // search for fitting sigma
        let tol = 1e-4;
        let maxtries = 50;
        for i in 0..self.rows {
            let dist_i = self.distance.row(i);
            let mut prow = self.p.row_mut(i);
            let mut betamin = f64::NEG_INFINITY;
            let mut betamax = f64::INFINITY;
            let mut beta = 1.0;
            let mut cnt = maxtries;
            let mut done = false;
            let mut psum = 0.0;

            while !done && cnt > 0 {
                // compute entropy and kernel row with beta precision
                cnt -= 1;
                psum = 0.0;
                let mut dp_sum = 0.0;
                for j in 0..self.rows {
                    let dist = dist_i[j];
                    let pj = if i == j { 0.0 } else { f64::exp(-dist * beta) };
                    dp_sum += dist * pj;
                    prow[j] = pj;
                    psum += pj;
                }

                // compute entropy
                let h = if psum > 0.0 {
                    f64::ln(psum) + (beta * dp_sum) / psum
                } else {
                    0.0
                };

                if h > h_target {
                    betamin = beta;
                    beta = if betamax == f64::INFINITY {
                        beta * 2.0
                    } else {
                        (beta + betamax) / 2.0
                    };
                } else {
                    betamax = beta;
                    beta = if betamin == f64::NEG_INFINITY {
                        beta / 2.0
                    } else {
                        (beta + betamin) / 2.0
                    };
                }
                done = f64::abs(h - h_target) < tol;
            }
            // normalize p
            for j in 0..self.rows {
                prow[j] /= psum;
            }
        }

        // compute probabilities
        let n2 = (2 * self.rows) as f64;
        for i in 0..self.rows {
            for j in i..self.rows {
                let puu = f64::max((self.p[[i, j]] + self.p[[j, i]]) / n2, 1e-100);
                self.p[[i, j]] = puu;
                self.p[[j, i]] = puu;
            }
        }
    }

    pub fn nextSync(&mut self) {
        self.iter += 1;

        // calc cost gradient
        let pmul = if self.iter < 100 { 4.0 } else { 1.0 };

        // calculate Q dist (unnormalized)
        let mut qu: Array2<f64> = Array2::zeros((self.rows, self.rows));
        let mut qsum = 0.0;
        for i in 0..self.rows {
            for j in (i + 1)..self.rows {
                let mut dsum = 0.0;
                for d in 0..self.target_dim {
                    let dhere = self.y[[i, d]] - self.y[[j, d]];
                    dsum += dhere * dhere;
                }
                let quu = 1.0 / (1.0 + dsum);
                qu[[i, j]] = quu;
                qu[[j, i]] = quu;
                qsum += 2.0 * quu;
            }
        }

        // normalize Q dist
        let mut q: Array2<f64> = Array2::zeros((self.rows, self.rows));
        for i in 0..self.rows {
            for j in (i + 1)..self.rows {
                let val = f64::max(qu[[i, j]] / qsum, 1e-100);
                q[[i, j]] = val;
                q[[j, i]] = val;
            }
        }

        let mut grad: Array2<f64> = Array2::zeros((self.rows, self.target_dim));
        for i in 0..self.rows {
            for j in 0..self.rows {
                let premult = 4.0 * (pmul * self.p[[i, j]] - q[[i, j]]) * qu[[i, j]];
                for d in 0..self.target_dim {
                    grad[[i, d]] += (self.y[[i, d]] - self.y[[j, d]]) * premult;
                }
            }
        }

        // perform gradient step
        let mut ymean: Array1<f64> = Array1::zeros(self.target_dim);
        for i in 0..self.rows {
            for d in 0..self.target_dim {
                let gid = grad[[i, d]];
                let sid = self.ystep[[i, d]];
                let gainid = self.gains[[i, d]];

                let mut newgain = match (gid, sid) {
                    (x, y) if x < 0.0 && y < 0.0 => gainid * 0.8,
                    (x, y) if x == 0.0 && y == 0.0 => gainid * 0.8,
                    (x, y) if x > 0.0 && y > 0.0 => gainid * 0.8,
                    (_, _) => gainid + 0.2,
                };

                newgain = if newgain < 0.01 { 0.01 } else { newgain };
                self.gains[[i, d]] = newgain;

                let momval = if self.iter < 250 { 0.5 } else { 0.8 };
                let newsid = momval * sid - self.epsilon * newgain * gid;
                self.ystep[[i, d]] = newsid;

                self.y[[i, d]] += newsid;
                ymean[d] += self.y[[i, d]];
            }
        }

        for i in 0..self.rows {
            for d in 0..self.target_dim {
                self.y[[i, d]] -= ymean[d] / (self.rows as f64);
            }
        }
    }
}
