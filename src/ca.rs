extern crate rand;
extern crate rayon;
use self::rayon::prelude::*;

pub struct CellA {
    pub cells: Vec<u8>,
    size: usize,
    min_surv: u8,
    max_surv: u8,
    min_birth: u8,
    max_birth: u8,
    max_age: u8,
}

impl CellA {
    pub fn new(
        size: usize,
        min_surv: u8,
        max_surv: u8,
        min_birth: u8,
        max_birth: u8,
    ) -> Self {
        let cells = vec![0; size * size * size];
        let max_age = 1;

        Self {
            cells,
            size,
            min_surv,
            max_surv,
            min_birth,
            max_birth,
            max_age,
        }
    }

    pub fn randomize(&mut self) {
        let cells = (0..self.size * self.size * self.size)
            .map(|_| if rand::random() { 1 } else { 0 })
            .collect();

        self.cells = cells;
    }

    pub fn next_gen(&mut self) {
        let new_cells = (0..self.size * self.size * self.size)
            .into_par_iter()
            .map(|idx| {
                if (idx > self.size * self.size + self.size)
                    && (idx
                        < (self.size * self.size * self.size)
                            - (self.size * self.size)
                            - self.size
                            - 1)
                {
                    let cur_state = self.cells[idx];
                    let count = count_neighbors(&self.cells, idx);

                    if cur_state > 0 {
                        if count >= self.min_surv && count <= self.max_surv {
                            let new_state = cur_state + 1;
                            if new_state > self.max_age {
                                self.max_age
                            } else {
                                new_state
                            }
                        } else {
                            0
                        }
                    } else if count >= self.min_birth && count <= self.max_birth {
                        1
                    } else {
                        0
                    }
                } else {
                    0
                }
            })
            .collect();

        self.cells = new_cells;
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
