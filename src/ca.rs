extern crate rand;
extern crate rayon;

pub struct CellA {
    pub cells: Vec<u8>,
    size: usize,
    lower_lim: usize,
    upper_lim: usize,
}

impl CellA {
    pub fn new(size: usize) -> Self {
        let cells = vec![0; size * size * size];

        // upper and lower limit for indices so that counting neighbors doesn't explode
        let lower_lim = size * size + size + 1;
        let upper_lim = size * size * size - lower_lim;

        Self {
            cells,
            size,
            upper_lim,
            lower_lim,
        }
    }

    pub fn compute(&mut self) {
        let w = 2.0;
        let base_coef = w / (2.0 * std::f32::consts::PI);

        // borrow checker sucks!
        let s = self.size;
        let ll = self.lower_lim;
        let ul = self.upper_lim;
        self.cells = (0..s)
            .map(move |z| {
                (0..s).map(move |y| {
                    (0..s).map(move |x| {
                        let idx = z * s * s + y * s + x;
                        let within_bounds = idx > ll && idx < ul;
                        let coef = base_coef * (((z + 1) / (s / 2)) as f32);

                        if (x as f32 * coef).sin()
                            + (y as f32 * coef).sin()
                            + (z as f32 * coef).sin()
                            > 0.0
                            && within_bounds
                        {
                            1
                        } else {
                            0
                        }
                    })
                })
            })
            .flatten()
            .flatten()
            .collect::<Vec<_>>();
    }
}

pub fn count_neighbors(cells: &[u8], idx: usize) -> u8 {
    use super::SIZE as size;

    let neighbors = [
        cells[idx + (size * size) + size + 1],
        cells[idx + (size * size) + size],
        cells[idx + (size * size) + size - 1],
        cells[idx + (size * size) + 1],
        cells[idx + (size * size)],
        cells[idx + (size * size) - 1],
        cells[idx + (size * size) - size + 1],
        cells[idx + (size * size) - size],
        cells[idx + (size * size) - size - 1],
        cells[idx + size + 1],
        cells[idx + size],
        cells[idx + size - 1],
        cells[idx + 1],
        cells[idx - 1],
        cells[idx - size + 1],
        cells[idx - size],
        cells[idx - size - 1],
        cells[idx - (size * size) + size + 1],
        cells[idx - (size * size) + size],
        cells[idx - (size * size) + size - 1],
        cells[idx - (size * size) + 1],
        cells[idx - (size * size)],
        cells[idx - (size * size) - 1],
        cells[idx - (size * size) - size + 1],
        cells[idx - (size * size) - size],
        cells[idx - (size * size) - size - 1],
    ];

    neighbors.iter().sum()
}
