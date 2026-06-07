//! # Semantic Field
//!
//! Force fields in embedding space for semantic navigation and manipulation.
//!
//! Provides tools for computing potential fields, gradients, attractors,
//! repulsors, and superposition of multiple fields in high-dimensional
//! embedding spaces.

/// Distance-based potential fields in embedding space.
pub mod potential {
    use std::f64;

    /// A point in embedding space.
    #[derive(Debug, Clone, PartialEq)]
    pub struct EmbeddingPoint {
        pub(crate) coords: Vec<f64>,
    }

    impl EmbeddingPoint {
        /// Create a new embedding point.
        pub fn new(coords: Vec<f64>) -> Self {
            Self { coords }
        }

        /// Get the dimensionality.
        pub fn dim(&self) -> usize {
            self.coords.len()
        }

        /// Get coordinate at index.
        pub fn get(&self, i: usize) -> Option<f64> {
            self.coords.get(i).copied()
        }

        /// Euclidean distance to another point.
        pub fn distance_to(&self, other: &EmbeddingPoint) -> f64 {
            self.coords
                .iter()
                .zip(other.coords.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        }

        /// Add two embedding points component-wise.
        pub fn add(&self, other: &EmbeddingPoint) -> EmbeddingPoint {
            EmbeddingPoint::new(
                self.coords
                    .iter()
                    .zip(other.coords.iter())
                    .map(|(a, b)| a + b)
                    .collect(),
            )
        }

        /// Scale embedding by scalar.
        pub fn scale(&self, s: f64) -> EmbeddingPoint {
            EmbeddingPoint::new(self.coords.iter().map(|c| c * s).collect())
        }

        /// Magnitude (L2 norm).
        pub fn magnitude(&self) -> f64 {
            self.coords.iter().map(|c| c * c).sum::<f64>().sqrt()
        }

        /// Normalize to unit length.
        pub fn normalize(&self) -> EmbeddingPoint {
            let mag = self.magnitude();
            if mag == 0.0 {
                return self.clone();
            }
            self.scale(1.0 / mag)
        }

        /// Dot product with another point.
        pub fn dot(&self, other: &EmbeddingPoint) -> f64 {
            self.coords
                .iter()
                .zip(other.coords.iter())
                .map(|(a, b)| a * b)
                .sum()
        }
    }

    /// Types of potential field functions.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum PotentialType {
        /// Coulomb-like: V = k / r
        Coulomb,
        /// Gaussian: V = A * exp(-r^2 / (2*sigma^2))
        Gaussian,
        /// Lennard-Jones-like: V = epsilon * ((sigma/r)^12 - 2*(sigma/r)^6)
        LennardJones,
        /// Linear: V = k * r
        Linear,
        /// Quadratic: V = k * r^2
        Quadratic,
    }

    /// A potential field centered at a point.
    #[derive(Debug, Clone)]
    pub struct PotentialField {
        center: EmbeddingPoint,
        field_type: PotentialType,
        strength: f64,
        sigma: f64,
    }

    impl PotentialField {
        /// Create a new potential field.
        pub fn new(center: EmbeddingPoint, field_type: PotentialType, strength: f64) -> Self {
            Self {
                center,
                field_type,
                strength,
                sigma: 1.0,
            }
        }

        /// Set sigma parameter.
        pub fn with_sigma(mut self, sigma: f64) -> Self {
            self.sigma = sigma;
            self
        }

        /// Compute potential at a given point.
        pub fn potential_at(&self, point: &EmbeddingPoint) -> f64 {
            let r = self.center.distance_to(point);
            if r < 1e-10 {
                return match self.field_type {
                    PotentialType::Coulomb => f64::INFINITY,
                    PotentialType::Gaussian => self.strength,
                    PotentialType::LennardJones => 0.0,
                    PotentialType::Linear => 0.0,
                    PotentialType::Quadratic => 0.0,
                };
            }
            match self.field_type {
                PotentialType::Coulomb => self.strength / r,
                PotentialType::Gaussian => {
                    self.strength * (-r.powi(2) / (2.0 * self.sigma.powi(2))).exp()
                }
                PotentialType::LennardJones => {
                    let sr = self.sigma / r;
                    self.strength * (sr.powi(12) - 2.0 * sr.powi(6))
                }
                PotentialType::Linear => self.strength * r,
                PotentialType::Quadratic => self.strength * r.powi(2),
            }
        }

        /// Get the center point.
        pub fn center(&self) -> &EmbeddingPoint {
            &self.center
        }

        /// Get field type.
        pub fn field_type(&self) -> PotentialType {
            self.field_type
        }

        /// Get strength.
        pub fn strength(&self) -> f64 {
            self.strength
        }
    }

    /// Compute the numerical gradient of potential at a point.
    pub fn numerical_gradient(field: &PotentialField, point: &EmbeddingPoint, epsilon: f64) -> EmbeddingPoint {
        let base = field.potential_at(point);
        let mut grad_coords = Vec::with_capacity(point.dim());
        for i in 0..point.dim() {
            let mut shifted = point.clone();
            shifted.coords[i] += epsilon;
            let shifted_potential = field.potential_at(&shifted);
            grad_coords.push((shifted_potential - base) / epsilon);
        }
        EmbeddingPoint::new(grad_coords)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_embedding_point_creation() {
            let p = EmbeddingPoint::new(vec![1.0, 2.0, 3.0]);
            assert_eq!(p.dim(), 3);
            assert_eq!(p.get(0), Some(1.0));
            assert_eq!(p.get(2), Some(3.0));
            assert_eq!(p.get(3), None);
        }

        #[test]
        fn test_distance_same_point() {
            let p = EmbeddingPoint::new(vec![1.0, 2.0, 3.0]);
            assert!((p.distance_to(&p) - 0.0).abs() < 1e-10);
        }

        #[test]
        fn test_distance_unit_vectors() {
            let a = EmbeddingPoint::new(vec![1.0, 0.0]);
            let b = EmbeddingPoint::new(vec![0.0, 1.0]);
            assert!((a.distance_to(&b) - 2.0_f64.sqrt()).abs() < 1e-10);
        }

        #[test]
        fn test_distance_3d() {
            let a = EmbeddingPoint::new(vec![0.0, 0.0, 0.0]);
            let b = EmbeddingPoint::new(vec![1.0, 2.0, 2.0]);
            assert!((a.distance_to(&b) - 3.0).abs() < 1e-10);
        }

        #[test]
        fn test_add_points() {
            let a = EmbeddingPoint::new(vec![1.0, 2.0]);
            let b = EmbeddingPoint::new(vec![3.0, 4.0]);
            let c = a.add(&b);
            assert_eq!(c.coords, vec![4.0, 6.0]);
        }

        #[test]
        fn test_scale_point() {
            let p = EmbeddingPoint::new(vec![1.0, 2.0, 3.0]);
            let s = p.scale(2.0);
            assert_eq!(s.coords, vec![2.0, 4.0, 6.0]);
        }

        #[test]
        fn test_magnitude() {
            let p = EmbeddingPoint::new(vec![3.0, 4.0]);
            assert!((p.magnitude() - 5.0).abs() < 1e-10);
        }

        #[test]
        fn test_normalize() {
            let p = EmbeddingPoint::new(vec![3.0, 4.0]);
            let n = p.normalize();
            assert!((n.magnitude() - 1.0).abs() < 1e-10);
        }

        #[test]
        fn test_normalize_zero() {
            let p = EmbeddingPoint::new(vec![0.0, 0.0]);
            let n = p.normalize();
            assert_eq!(n.coords, vec![0.0, 0.0]);
        }

        #[test]
        fn test_dot_product() {
            let a = EmbeddingPoint::new(vec![1.0, 2.0, 3.0]);
            let b = EmbeddingPoint::new(vec![4.0, 5.0, 6.0]);
            assert!((a.dot(&b) - 32.0).abs() < 1e-10);
        }

        #[test]
        fn test_coulomb_potential() {
            let center = EmbeddingPoint::new(vec![0.0]);
            let field = PotentialField::new(center, PotentialType::Coulomb, 1.0);
            let p = EmbeddingPoint::new(vec![1.0]);
            assert!((field.potential_at(&p) - 1.0).abs() < 1e-10);
        }

        #[test]
        fn test_coulomb_potential_distance() {
            let center = EmbeddingPoint::new(vec![0.0]);
            let field = PotentialField::new(center, PotentialType::Coulomb, 2.0);
            let p = EmbeddingPoint::new(vec![4.0]);
            assert!((field.potential_at(&p) - 0.5).abs() < 1e-10);
        }

        #[test]
        fn test_gaussian_potential_center() {
            let center = EmbeddingPoint::new(vec![0.0]);
            let field = PotentialField::new(center, PotentialType::Gaussian, 5.0).with_sigma(1.0);
            let p = EmbeddingPoint::new(vec![0.0]);
            assert!((field.potential_at(&p) - 5.0).abs() < 1e-10);
        }

        #[test]
        fn test_gaussian_potential_far() {
            let center = EmbeddingPoint::new(vec![0.0]);
            let field = PotentialField::new(center, PotentialType::Gaussian, 5.0).with_sigma(1.0);
            let p = EmbeddingPoint::new(vec![10.0]);
            assert!(field.potential_at(&p).abs() < 0.01);
        }

        #[test]
        fn test_linear_potential() {
            let center = EmbeddingPoint::new(vec![0.0]);
            let field = PotentialField::new(center, PotentialType::Linear, 3.0);
            let p = EmbeddingPoint::new(vec![2.0]);
            assert!((field.potential_at(&p) - 6.0).abs() < 1e-10);
        }

        #[test]
        fn test_quadratic_potential() {
            let center = EmbeddingPoint::new(vec![0.0]);
            let field = PotentialField::new(center, PotentialType::Quadratic, 2.0);
            let p = EmbeddingPoint::new(vec![3.0]);
            assert!((field.potential_at(&p) - 18.0).abs() < 1e-10);
        }

        #[test]
        fn test_lennard_jones_equilibrium() {
            let center = EmbeddingPoint::new(vec![0.0]);
            let field = PotentialField::new(center, PotentialType::LennardJones, 1.0).with_sigma(1.0);
            // At r = sigma, LJ should be -1
            let p = EmbeddingPoint::new(vec![1.0]);
            assert!((field.potential_at(&p) - (-1.0)).abs() < 1e-10);
        }

        #[test]
        fn test_numerical_gradient_direction() {
            let center = EmbeddingPoint::new(vec![0.0]);
            let field = PotentialField::new(center, PotentialType::Linear, 1.0);
            let p = EmbeddingPoint::new(vec![2.0]);
            let grad = numerical_gradient(&field, &p, 0.001);
            // Gradient of k*r should be positive (pointing away from center)
            assert!(grad.get(0).unwrap() > 0.0);
        }

        #[test]
        fn test_field_accessors() {
            let center = EmbeddingPoint::new(vec![1.0, 2.0]);
            let field = PotentialField::new(center.clone(), PotentialType::Coulomb, 3.0);
            assert_eq!(field.center().coords, vec![1.0, 2.0]);
            assert_eq!(field.field_type(), PotentialType::Coulomb);
            assert!((field.strength() - 3.0).abs() < 1e-10);
        }

        #[test]
        fn test_equality() {
            let a = EmbeddingPoint::new(vec![1.0, 2.0]);
            let b = EmbeddingPoint::new(vec![1.0, 2.0]);
            assert_eq!(a, b);
        }

        #[test]
        fn test_inequality() {
            let a = EmbeddingPoint::new(vec![1.0, 2.0]);
            let b = EmbeddingPoint::new(vec![1.0, 3.0]);
            assert_ne!(a, b);
        }
    }
}

/// Force direction toward semantic targets.
pub mod gradient {
    use super::potential::{EmbeddingPoint, PotentialField, numerical_gradient};

    /// Result of a gradient computation.
    #[derive(Debug, Clone)]
    pub struct GradientResult {
        pub direction: EmbeddingPoint,
        pub magnitude: f64,
    }

    impl GradientResult {
        /// Create a new gradient result.
        pub fn new(direction: EmbeddingPoint, magnitude: f64) -> Self {
            Self { direction, magnitude }
        }

        /// Get unit direction vector.
        pub fn unit_direction(&self) -> EmbeddingPoint {
            self.direction.normalize()
        }
    }

    /// Compute the gradient of a field at a point.
    pub fn compute_gradient(field: &PotentialField, point: &EmbeddingPoint) -> GradientResult {
        let dir = numerical_gradient(field, point, 1e-5);
        let mag = dir.magnitude();
        GradientResult::new(dir, mag)
    }

    /// Take a gradient descent step.
    pub fn gradient_step(
        field: &PotentialField,
        point: &EmbeddingPoint,
        step_size: f64,
    ) -> EmbeddingPoint {
        let grad = compute_gradient(field, point);
        let step = grad.direction.scale(-step_size);
        point.add(&step)
    }

    /// Take a gradient ascent step.
    pub fn gradient_ascent_step(
        field: &PotentialField,
        point: &EmbeddingPoint,
        step_size: f64,
    ) -> EmbeddingPoint {
        let grad = compute_gradient(field, point);
        let step = grad.direction.scale(step_size);
        point.add(&step)
    }

    /// Perform gradient descent for multiple steps.
    pub fn gradient_descent(
        field: &PotentialField,
        start: &EmbeddingPoint,
        step_size: f64,
        steps: usize,
    ) -> EmbeddingPoint {
        let mut current = start.clone();
        for _ in 0..steps {
            current = gradient_step(field, &current, step_size);
        }
        current
    }

    /// Perform gradient ascent for multiple steps.
    pub fn gradient_ascent(
        field: &PotentialField,
        start: &EmbeddingPoint,
        step_size: f64,
        steps: usize,
    ) -> EmbeddingPoint {
        let mut current = start.clone();
        for _ in 0..steps {
            current = gradient_ascent_step(field, &current, step_size);
        }
        current
    }

    /// Compute gradient with momentum.
    pub fn gradient_descent_with_momentum(
        field: &PotentialField,
        start: &EmbeddingPoint,
        step_size: f64,
        momentum: f64,
        steps: usize,
    ) -> EmbeddingPoint {
        let mut current = start.clone();
        let mut velocity = EmbeddingPoint::new(vec![0.0; start.dim()]);
        for _ in 0..steps {
            let grad = compute_gradient(field, &current);
            velocity = velocity.scale(momentum).add(&grad.direction.scale(-step_size));
            current = current.add(&velocity);
        }
        current
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::potential::{EmbeddingPoint, PotentialField, PotentialType};

        #[test]
        fn test_gradient_result_creation() {
            let dir = EmbeddingPoint::new(vec![1.0, 0.0]);
            let result = GradientResult::new(dir.clone(), 1.0);
            assert!((result.magnitude - 1.0).abs() < 1e-10);
        }

        #[test]
        fn test_unit_direction() {
            let dir = EmbeddingPoint::new(vec![3.0, 4.0]);
            let result = GradientResult::new(dir, 5.0);
            let unit = result.unit_direction();
            assert!((unit.magnitude() - 1.0).abs() < 1e-10);
        }

        #[test]
        fn test_compute_gradient() {
            let center = EmbeddingPoint::new(vec![0.0]);
            let field = PotentialField::new(center, PotentialType::Linear, 1.0);
            let p = EmbeddingPoint::new(vec![2.0]);
            let grad = compute_gradient(&field, &p);
            assert!(grad.magnitude > 0.0);
        }

        #[test]
        fn test_gradient_step_moves() {
            let center = EmbeddingPoint::new(vec![0.0]);
            let field = PotentialField::new(center.clone(), PotentialType::Quadratic, 1.0);
            let start = EmbeddingPoint::new(vec![5.0]);
            let after = gradient_step(&field, &start, 0.1);
            // Should move toward center
            assert!(after.distance_to(&center) < start.distance_to(&center));
        }

        #[test]
        fn test_gradient_ascent_step_moves_away() {
            let center = EmbeddingPoint::new(vec![0.0]);
            let field = PotentialField::new(center.clone(), PotentialType::Quadratic, 1.0);
            let start = EmbeddingPoint::new(vec![1.0]);
            let after = gradient_ascent_step(&field, &start, 0.1);
            // Should move away from center
            assert!(after.distance_to(&center) > start.distance_to(&center));
        }

        #[test]
        fn test_gradient_descent_converges() {
            let center = EmbeddingPoint::new(vec![0.0, 0.0]);
            let field = PotentialField::new(center.clone(), PotentialType::Quadratic, 1.0);
            let start = EmbeddingPoint::new(vec![10.0, 10.0]);
            let result = gradient_descent(&field, &start, 0.01, 1000);
            assert!(result.distance_to(&center) < 1.0);
        }

        #[test]
        fn test_gradient_ascent_diverges() {
            let center = EmbeddingPoint::new(vec![0.0]);
            let field = PotentialField::new(center, PotentialType::Quadratic, 1.0);
            let start = EmbeddingPoint::new(vec![1.0]);
            let result = gradient_ascent(&field, &start, 0.1, 10);
            assert!(result.distance_to(&EmbeddingPoint::new(vec![0.0])) > 1.0);
        }

        #[test]
        fn test_momentum_converges() {
            let center = EmbeddingPoint::new(vec![0.0]);
            let field = PotentialField::new(center.clone(), PotentialType::Quadratic, 1.0);
            let start = EmbeddingPoint::new(vec![5.0]);
            let result = gradient_descent_with_momentum(&field, &start, 0.01, 0.9, 200);
            assert!(result.distance_to(&center) < 2.0);
        }

        #[test]
        fn test_momentum_faster_than_vanilla() {
            let center = EmbeddingPoint::new(vec![0.0]);
            let field = PotentialField::new(center.clone(), PotentialType::Quadratic, 1.0);
            let start = EmbeddingPoint::new(vec![10.0]);
            let vanilla = gradient_descent(&field, &start, 0.01, 50);
            let with_momentum = gradient_descent_with_momentum(&field, &start, 0.01, 0.9, 50);
            // Momentum should get closer
            assert!(with_momentum.distance_to(&center) <= vanilla.distance_to(&center));
        }

        #[test]
        fn test_gradient_magnitude_zero_at_center() {
            let center = EmbeddingPoint::new(vec![5.0]);
            let field = PotentialField::new(center.clone(), PotentialType::Quadratic, 1.0);
            // At the center of quadratic potential, gradient is zero
            let grad = compute_gradient(&field, &center);
            assert!(grad.magnitude < 0.01);
        }
    }
}

/// Fixed points in embedding space (attractors).
pub mod attractor {
    use super::potential::{EmbeddingPoint, PotentialField, PotentialType};

    /// An attractor in embedding space.
    #[derive(Debug, Clone)]
    pub struct Attractor {
        position: EmbeddingPoint,
        strength: f64,
        basin_radius: f64,
    }

    impl Attractor {
        /// Create a new attractor.
        pub fn new(position: EmbeddingPoint, strength: f64, basin_radius: f64) -> Self {
            Self { position, strength, basin_radius }
        }

        /// Get the position.
        pub fn position(&self) -> &EmbeddingPoint {
            &self.position
        }

        /// Get the strength.
        pub fn strength(&self) -> f64 {
            self.strength
        }

        /// Get the basin of attraction radius.
        pub fn basin_radius(&self) -> f64 {
            self.basin_radius
        }

        /// Check if a point is within the basin of attraction.
        pub fn is_in_basin(&self, point: &EmbeddingPoint) -> bool {
            self.position.distance_to(point) <= self.basin_radius
        }

        /// Compute the attractive force on a point.
        pub fn force_on(&self, point: &EmbeddingPoint) -> EmbeddingPoint {
            let diff = self.position.add(&point.scale(-1.0));
            let dist = diff.magnitude();
            if dist < 1e-10 || dist > self.basin_radius {
                return EmbeddingPoint::new(vec![0.0; point.dim()]);
            }
            diff.normalize().scale(self.strength / dist)
        }

        /// Convert to a potential field.
        pub fn to_potential_field(&self) -> PotentialField {
            PotentialField::new(self.position.clone(), PotentialType::Coulomb, -self.strength)
        }
    }

    /// A set of attractors.
    #[derive(Debug, Clone)]
    pub struct AttractorSet {
        attractors: Vec<Attractor>,
    }

    impl AttractorSet {
        /// Create an empty attractor set.
        pub fn new() -> Self {
            Self { attractors: Vec::new() }
        }

        /// Add an attractor.
        pub fn add(&mut self, attractor: Attractor) {
            self.attractors.push(attractor);
        }

        /// Get the number of attractors.
        pub fn len(&self) -> usize {
            self.attractors.len()
        }

        /// Check if empty.
        pub fn is_empty(&self) -> bool {
            self.attractors.is_empty()
        }

        /// Find the nearest attractor to a point.
        pub fn nearest(&self, point: &EmbeddingPoint) -> Option<&Attractor> {
            self.attractors.iter().min_by(|a, b| {
                a.position().distance_to(point)
                    .partial_cmp(&b.position().distance_to(point))
                    .unwrap()
            })
        }

        /// Compute the combined force from all attractors on a point.
        pub fn combined_force(&self, point: &EmbeddingPoint) -> EmbeddingPoint {
            let forces: Vec<EmbeddingPoint> = self.attractors.iter().map(|a| a.force_on(point)).collect();
            if forces.is_empty() {
                return EmbeddingPoint::new(vec![0.0; point.dim()]);
            }
            forces.iter().skip(1).fold(forces[0].clone(), |acc, f| acc.add(f))
        }

        /// Find all attractors whose basin contains the point.
        pub fn containing_attractors(&self, point: &EmbeddingPoint) -> Vec<&Attractor> {
            self.attractors.iter().filter(|a| a.is_in_basin(point)).collect()
        }
    }

    impl Default for AttractorSet {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_attractor_creation() {
            let pos = EmbeddingPoint::new(vec![1.0, 2.0]);
            let a = Attractor::new(pos.clone(), 5.0, 10.0);
            assert_eq!(a.position().coords, vec![1.0, 2.0]);
            assert!((a.strength() - 5.0).abs() < 1e-10);
            assert!((a.basin_radius() - 10.0).abs() < 1e-10);
        }

        #[test]
        fn test_in_basin_inside() {
            let pos = EmbeddingPoint::new(vec![0.0, 0.0]);
            let a = Attractor::new(pos, 1.0, 5.0);
            let p = EmbeddingPoint::new(vec![3.0, 4.0]);
            assert!(a.is_in_basin(&p));
        }

        #[test]
        fn test_in_basin_outside() {
            let pos = EmbeddingPoint::new(vec![0.0, 0.0]);
            let a = Attractor::new(pos, 1.0, 3.0);
            let p = EmbeddingPoint::new(vec![3.0, 4.0]);
            assert!(!a.is_in_basin(&p));
        }

        #[test]
        fn test_force_direction() {
            let pos = EmbeddingPoint::new(vec![10.0]);
            let a = Attractor::new(pos, 1.0, 20.0);
            let p = EmbeddingPoint::new(vec![0.0]);
            let force = a.force_on(&p);
            // Force should point toward attractor (positive direction)
            assert!(force.get(0).unwrap() > 0.0);
        }

        #[test]
        fn test_force_outside_basin() {
            let pos = EmbeddingPoint::new(vec![10.0]);
            let a = Attractor::new(pos, 1.0, 5.0);
            let p = EmbeddingPoint::new(vec![0.0]);
            let force = a.force_on(&p);
            assert!((force.magnitude() - 0.0).abs() < 1e-10);
        }

        #[test]
        fn test_force_at_center() {
            let pos = EmbeddingPoint::new(vec![5.0]);
            let a = Attractor::new(pos.clone(), 1.0, 10.0);
            let force = a.force_on(&pos);
            assert!((force.magnitude() - 0.0).abs() < 1e-10);
        }

        #[test]
        fn test_to_potential_field() {
            let pos = EmbeddingPoint::new(vec![5.0]);
            let a = Attractor::new(pos.clone(), 2.0, 10.0);
            let field = a.to_potential_field();
            assert_eq!(field.center().coords, vec![5.0]);
        }

        #[test]
        fn test_attractor_set_empty() {
            let set = AttractorSet::new();
            assert!(set.is_empty());
            assert_eq!(set.len(), 0);
        }

        #[test]
        fn test_attractor_set_add() {
            let mut set = AttractorSet::new();
            set.add(Attractor::new(EmbeddingPoint::new(vec![0.0]), 1.0, 5.0));
            assert_eq!(set.len(), 1);
        }

        #[test]
        fn test_attractor_set_nearest() {
            let mut set = AttractorSet::new();
            set.add(Attractor::new(EmbeddingPoint::new(vec![10.0]), 1.0, 5.0));
            set.add(Attractor::new(EmbeddingPoint::new(vec![1.0]), 1.0, 5.0));
            let p = EmbeddingPoint::new(vec![0.0]);
            let nearest = set.nearest(&p).unwrap();
            assert_eq!(nearest.position().coords, vec![1.0]);
        }

        #[test]
        fn test_combined_force() {
            let mut set = AttractorSet::new();
            set.add(Attractor::new(EmbeddingPoint::new(vec![10.0]), 1.0, 20.0));
            set.add(Attractor::new(EmbeddingPoint::new(vec![-10.0]), 1.0, 20.0));
            let p = EmbeddingPoint::new(vec![0.0]);
            let force = set.combined_force(&p);
            // Symmetric attractors should cancel out
            assert!(force.get(0).unwrap().abs() < 0.01);
        }

        #[test]
        fn test_containing_attractors() {
            let mut set = AttractorSet::new();
            set.add(Attractor::new(EmbeddingPoint::new(vec![0.0]), 1.0, 5.0));
            set.add(Attractor::new(EmbeddingPoint::new(vec![100.0]), 1.0, 5.0));
            let p = EmbeddingPoint::new(vec![1.0]);
            let contained = set.containing_attractors(&p);
            assert_eq!(contained.len(), 1);
        }

        #[test]
        fn test_default() {
            let set = AttractorSet::default();
            assert!(set.is_empty());
        }
    }
}

/// Push embeddings apart for diversity.
pub mod repulsor {
    use super::potential::EmbeddingPoint;

    /// A repulsor that pushes points away.
    #[derive(Debug, Clone)]
    pub struct Repulsor {
        position: EmbeddingPoint,
        strength: f64,
        range: f64,
    }

    impl Repulsor {
        /// Create a new repulsor.
        pub fn new(position: EmbeddingPoint, strength: f64, range: f64) -> Self {
            Self { position, strength, range }
        }

        /// Get position.
        pub fn position(&self) -> &EmbeddingPoint {
            &self.position
        }

        /// Compute repulsive force on a point.
        pub fn force_on(&self, point: &EmbeddingPoint) -> EmbeddingPoint {
            let diff = point.add(&self.position.scale(-1.0));
            let dist = diff.magnitude();
            if dist < 1e-10 || dist > self.range {
                return EmbeddingPoint::new(vec![0.0; point.dim()]);
            }
            diff.normalize().scale(self.strength / dist.powi(2))
        }

        /// Check if point is within repulsion range.
        pub fn is_in_range(&self, point: &EmbeddingPoint) -> bool {
            self.position.distance_to(point) <= self.range
        }
    }

    /// Compute pairwise repulsion between a set of points.
    pub fn pairwise_repulsion(points: &[EmbeddingPoint], strength: f64) -> Vec<EmbeddingPoint> {
        points
            .iter()
            .enumerate()
            .map(|(i, point)| {
                let mut total_force = EmbeddingPoint::new(vec![0.0; point.dim()]);
                for (j, other) in points.iter().enumerate() {
                    if i != j {
                        let rep = Repulsor::new(other.clone(), strength, f64::MAX);
                        total_force = total_force.add(&rep.force_on(point));
                    }
                }
                total_force
            })
            .collect()
    }

    /// Maximize diversity by iteratively applying repulsion.
    pub fn diversify(points: &[EmbeddingPoint], strength: f64, steps: usize, step_size: f64) -> Vec<EmbeddingPoint> {
        let mut current: Vec<EmbeddingPoint> = points.to_vec();
        for _ in 0..steps {
            let forces = pairwise_repulsion(&current, strength);
            current = current
                .iter()
                .zip(forces.iter())
                .map(|(p, f)| p.add(&f.scale(step_size)))
                .collect();
        }
        current
    }

    /// Compute the diversity (average pairwise distance) of a set of points.
    pub fn diversity(points: &[EmbeddingPoint]) -> f64 {
        if points.len() < 2 {
            return 0.0;
        }
        let mut total = 0.0;
        let mut count = 0;
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                total += points[i].distance_to(&points[j]);
                count += 1;
            }
        }
        total / count as f64
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_repulsor_creation() {
            let pos = EmbeddingPoint::new(vec![5.0]);
            let r = Repulsor::new(pos, 2.0, 10.0);
            assert_eq!(r.position().coords, vec![5.0]);
        }

        #[test]
        fn test_repulsive_force_direction() {
            let pos = EmbeddingPoint::new(vec![0.0]);
            let r = Repulsor::new(pos, 1.0, 20.0);
            let p = EmbeddingPoint::new(vec![1.0]);
            let force = r.force_on(&p);
            // Should push away from 0 (positive direction)
            assert!(force.get(0).unwrap() > 0.0);
        }

        #[test]
        fn test_repulsive_force_decreases_with_distance() {
            let pos = EmbeddingPoint::new(vec![0.0]);
            let r = Repulsor::new(pos, 1.0, 100.0);
            let near = EmbeddingPoint::new(vec![1.0]);
            let far = EmbeddingPoint::new(vec![10.0]);
            assert!(r.force_on(&near).magnitude() > r.force_on(&far).magnitude());
        }

        #[test]
        fn test_repulsive_force_outside_range() {
            let pos = EmbeddingPoint::new(vec![0.0]);
            let r = Repulsor::new(pos, 1.0, 5.0);
            let p = EmbeddingPoint::new(vec![10.0]);
            assert!((r.force_on(&p).magnitude() - 0.0).abs() < 1e-10);
        }

        #[test]
        fn test_is_in_range_true() {
            let pos = EmbeddingPoint::new(vec![0.0]);
            let r = Repulsor::new(pos, 1.0, 5.0);
            assert!(r.is_in_range(&EmbeddingPoint::new(vec![3.0])));
        }

        #[test]
        fn test_is_in_range_false() {
            let pos = EmbeddingPoint::new(vec![0.0]);
            let r = Repulsor::new(pos, 1.0, 5.0);
            assert!(!r.is_in_range(&EmbeddingPoint::new(vec![10.0])));
        }

        #[test]
        fn test_pairwise_repulsion_symmetry() {
            let points = vec![
                EmbeddingPoint::new(vec![0.0]),
                EmbeddingPoint::new(vec![2.0]),
            ];
            let forces = pairwise_repulsion(&points, 1.0);
            // Forces should be equal and opposite
            assert!((forces[0].get(0).unwrap() + forces[1].get(0).unwrap()).abs() < 1e-6);
        }

        #[test]
        fn test_pairwise_repulsion_three_points() {
            let points = vec![
                EmbeddingPoint::new(vec![0.0]),
                EmbeddingPoint::new(vec![1.0]),
                EmbeddingPoint::new(vec![2.0]),
            ];
            let forces = pairwise_repulsion(&points, 1.0);
            assert_eq!(forces.len(), 3);
            // Endpoints should have nonzero force
            assert!(forces[0].magnitude() > 0.0);
            assert!(forces[2].magnitude() > 0.0);
        }

        #[test]
        fn test_diversify_increases_diversity() {
            let points = vec![
                EmbeddingPoint::new(vec![0.0]),
                EmbeddingPoint::new(vec![0.1]),
                EmbeddingPoint::new(vec![0.2]),
            ];
            let before = diversity(&points);
            let after_points = diversify(&points, 1.0, 100, 0.01);
            let after = diversity(&after_points);
            assert!(after > before);
        }

        #[test]
        fn test_diversity_single_point() {
            let points = vec![EmbeddingPoint::new(vec![1.0])];
            assert!((diversity(&points) - 0.0).abs() < 1e-10);
        }

        #[test]
        fn test_diversity_uniform() {
            let points = vec![
                EmbeddingPoint::new(vec![0.0]),
                EmbeddingPoint::new(vec![1.0]),
                EmbeddingPoint::new(vec![2.0]),
            ];
            let d = diversity(&points);
            // avg of (0,1)=1, (1,2)=1, (0,2)=2 = 4/3
            assert!((d - 4.0 / 3.0).abs() < 1e-10);
        }
    }
}

/// Combine multiple fields via superposition.
pub mod superposition {
    use super::potential::{EmbeddingPoint, PotentialField};

    /// A superposition of multiple potential fields.
    #[derive(Debug, Clone)]
    pub struct Superposition {
        fields: Vec<(PotentialField, f64)>,
    }

    impl Superposition {
        /// Create an empty superposition.
        pub fn new() -> Self {
            Self { fields: Vec::new() }
        }

        /// Add a field with a weight.
        pub fn add_field(&mut self, field: PotentialField, weight: f64) {
            self.fields.push((field, weight));
        }

        /// Compute the combined potential at a point.
        pub fn potential_at(&self, point: &EmbeddingPoint) -> f64 {
            self.fields
                .iter()
                .map(|(field, weight)| field.potential_at(point) * weight)
                .sum()
        }

        /// Get the number of fields.
        pub fn len(&self) -> usize {
            self.fields.len()
        }

        /// Check if empty.
        pub fn is_empty(&self) -> bool {
            self.fields.is_empty()
        }

        /// Compute numerical gradient of the superposition.
        pub fn gradient(&self, point: &EmbeddingPoint) -> EmbeddingPoint {
            let eps = 1e-5;
            let base = self.potential_at(point);
            let mut grad_coords = Vec::with_capacity(point.dim());
            for i in 0..point.dim() {
                let mut shifted = point.clone();
                shifted.coords[i] += eps;
                grad_coords.push((self.potential_at(&shifted) - base) / eps);
            }
            EmbeddingPoint::new(grad_coords)
        }

        /// Find a local minimum via gradient descent.
        pub fn find_minimum(&self, start: &EmbeddingPoint, step_size: f64, max_steps: usize) -> EmbeddingPoint {
            let mut current = start.clone();
            for _ in 0..max_steps {
                let grad = self.gradient(&current);
                let step = grad.scale(-step_size);
                let next = current.add(&step);
                if next.distance_to(&current) < 1e-8 {
                    break;
                }
                current = next;
            }
            current
        }
    }

    impl Default for Superposition {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Create a superposition from multiple fields with equal weights.
    pub fn equal_superposition(fields: Vec<PotentialField>) -> Superposition {
        let mut sup = Superposition::new();
        for field in fields {
            sup.add_field(field, 1.0);
        }
        sup
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::potential::PotentialType;

        #[test]
        fn test_empty_superposition() {
            let sup = Superposition::new();
            let p = EmbeddingPoint::new(vec![1.0]);
            assert!((sup.potential_at(&p) - 0.0).abs() < 1e-10);
        }

        #[test]
        fn test_single_field() {
            let mut sup = Superposition::new();
            let center = EmbeddingPoint::new(vec![0.0]);
            sup.add_field(
                PotentialField::new(center, PotentialType::Linear, 2.0),
                1.0,
            );
            let p = EmbeddingPoint::new(vec![3.0]);
            assert!((sup.potential_at(&p) - 6.0).abs() < 1e-10);
        }

        #[test]
        fn test_weighted_single_field() {
            let mut sup = Superposition::new();
            let center = EmbeddingPoint::new(vec![0.0]);
            sup.add_field(
                PotentialField::new(center, PotentialType::Linear, 2.0),
                0.5,
            );
            let p = EmbeddingPoint::new(vec![3.0]);
            assert!((sup.potential_at(&p) - 3.0).abs() < 1e-10);
        }

        #[test]
        fn test_two_fields_additive() {
            let mut sup = Superposition::new();
            let c1 = EmbeddingPoint::new(vec![0.0]);
            let c2 = EmbeddingPoint::new(vec![0.0]);
            sup.add_field(PotentialField::new(c1, PotentialType::Linear, 1.0), 1.0);
            sup.add_field(PotentialField::new(c2, PotentialType::Linear, 2.0), 1.0);
            let p = EmbeddingPoint::new(vec![5.0]);
            // 1*5 + 2*5 = 15
            assert!((sup.potential_at(&p) - 15.0).abs() < 1e-10);
        }

        #[test]
        fn test_len_and_empty() {
            let mut sup = Superposition::new();
            assert!(sup.is_empty());
            sup.add_field(
                PotentialField::new(EmbeddingPoint::new(vec![0.0]), PotentialType::Linear, 1.0),
                1.0,
            );
            assert_eq!(sup.len(), 1);
            assert!(!sup.is_empty());
        }

        #[test]
        fn test_gradient_direction() {
            let mut sup = Superposition::new();
            sup.add_field(
                PotentialField::new(EmbeddingPoint::new(vec![0.0]), PotentialType::Quadratic, 1.0),
                1.0,
            );
            let p = EmbeddingPoint::new(vec![5.0]);
            let grad = sup.gradient(&p);
            // Gradient of x^2 at x=5 should be ~10
            assert!(grad.get(0).unwrap() > 5.0);
        }

        #[test]
        fn test_find_minimum_simple() {
            let mut sup = Superposition::new();
            // Minimum at x=5
            sup.add_field(
                PotentialField::new(EmbeddingPoint::new(vec![5.0]), PotentialType::Quadratic, 1.0),
                1.0,
            );
            let start = EmbeddingPoint::new(vec![0.0]);
            let result = sup.find_minimum(&start, 0.01, 10000);
            assert!(result.distance_to(&EmbeddingPoint::new(vec![5.0])) < 0.5);
        }

        #[test]
        fn test_equal_superposition() {
            let fields = vec![
                PotentialField::new(EmbeddingPoint::new(vec![0.0]), PotentialType::Linear, 1.0),
                PotentialField::new(EmbeddingPoint::new(vec![0.0]), PotentialType::Linear, 2.0),
            ];
            let sup = equal_superposition(fields);
            assert_eq!(sup.len(), 2);
        }

        #[test]
        fn test_default() {
            let sup = Superposition::default();
            assert!(sup.is_empty());
        }

        #[test]
        fn test_negative_weight() {
            let mut sup = Superposition::new();
            sup.add_field(
                PotentialField::new(EmbeddingPoint::new(vec![0.0]), PotentialType::Linear, 3.0),
                -1.0,
            );
            let p = EmbeddingPoint::new(vec![2.0]);
            assert!((sup.potential_at(&p) - (-6.0)).abs() < 1e-10);
        }
    }
}

pub use potential::{EmbeddingPoint, PotentialField, PotentialType};
