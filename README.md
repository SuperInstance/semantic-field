# Semantic Field

[![crates.io](https://img.shields.io/crates/v/semantic-field.svg)](https://crates.io/crates/semantic-field)
[![docs.rs](https://docs.rs/semantic-field/badge.svg)](https://docs.rs/semantic-field)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Force fields in embedding space — attractors, repulsors, gradients, and superposition for semantic navigation.**

---

## The Problem

Embedding spaces are flat — vectors exist as points in high-dimensional space with no notion of "landscape." But semantic relationships create natural gradients: related concepts attract, contradictory ones repel, and navigating through embedding space should follow these forces.

## Why This Exists

Semantic Field implements physics-inspired potential fields in embedding space:
- **Attractors**: Regions that pull nearby embeddings (related concepts)
- **Repulsors**: Regions that push embeddings away (contradictions)
- **Potential types**: Linear, quadratic, inverse, and inverse-quadratic
- **Superposition**: Combine multiple fields with weights
- **Gradient computation**: Find the direction of steepest change
- **Minimum finding**: Gradient descent to find semantic equilibria

## Architecture

```
  Embedding Space (high-dimensional)
  
     ┌─────────┐                    ┌─────────┐
     │Attractor│ ← pulls toward →   │Attractor│
     │ (concept│                    │(concept │
     │  "dog") │                    │ "wolf") │
     └────┬────┘                    └────┬────┘
          │                              │
          └────── Superposition ─────────┘
                    │
              ┌─────▼──────┐
              │  Gradient  │
              │  Descent   │
              │     ↓      │
              │  Minimum   │ = semantic equilibrium
              └────────────┘
```

## Installation

```toml
[dependencies]
semantic-field = "0.1"
```

## API Reference

### `EmbeddingPoint`

A point in high-dimensional embedding space:

```rust
use semantic_field::potential::EmbeddingPoint;

let p = EmbeddingPoint::new(vec![1.0, 2.0, 3.0]);
let other = EmbeddingPoint::new(vec![4.0, 5.0, 6.0]);
let dist = p.distance_to(&other);
```

### `PotentialField` & `PotentialType`

Distance-based potential fields:

```rust
use semantic_field::potential::*;

let center = EmbeddingPoint::new(vec![0.0, 0.0]);
let field = PotentialField::new(center, PotentialType::Quadratic, 2.0);

let point = EmbeddingPoint::new(vec![3.0, 0.0]);
let potential = field.potential_at(&point);
let gradient = field.gradient_at(&point);
```

### `Superposition`

Weighted combination of multiple fields:

```rust
use semantic_field::potential::*;

let mut sup = Superposition::new();
sup.add_field(PotentialField::new(
    EmbeddingPoint::new(vec![0.0]), PotentialType::Linear, 1.0
), 1.0);
sup.add_field(PotentialField::new(
    EmbeddingPoint::new(vec![0.0]), PotentialType::Linear, 2.0
), 1.0);

let p = EmbeddingPoint::new(vec![3.0]);
let total = sup.potential_at(&p); // additive superposition
```

### Gradient Descent

Find semantic minima:

```rust
use semantic_field::potential::*;

let mut sup = Superposition::new();
sup.add_field(PotentialField::new(
    EmbeddingPoint::new(vec![5.0]), PotentialType::Quadratic, 1.0
), 1.0);

let start = EmbeddingPoint::new(vec![0.0]);
let minimum = sup.find_minimum(&start, 0.01, 10000);
// Should converge near x=5.0
```

## Potential Types

| Type | Formula | Use Case |
|------|---------|----------|
| Linear | k × r | Constant force (uniform gravity) |
| Quadratic | k × r² | Distance-penalized attraction |
| Inverse | k / r | Long-range attraction |
| InverseQuadratic | k / r² | Short-range strong attraction |

## Usage Examples

### Example 1: Semantic Attraction

```rust
use semantic_field::potential::*;

let dog = EmbeddingPoint::new(vec![1.0, 0.5, 0.3]);
let wolf = EmbeddingPoint::new(vec![0.9, 0.4, 0.35]);

let attractor = PotentialField::new(dog, PotentialType::Quadratic, 1.0);
let potential = attractor.potential_at(&wolf);
// Low potential = similar concepts
```

### Example 2: Multi-Field Landscape

```rust
use semantic_field::potential::*;

let mut landscape = Superposition::new();
// Attract toward "science"
landscape.add_field(PotentialField::new(
    EmbeddingPoint::new(vec![1.0, 0.0, 0.0]), PotentialType::Quadratic, 1.0), 1.0);
// Repel from "pseudoscience"
landscape.add_field(PotentialField::new(
    EmbeddingPoint::new(vec![-1.0, 0.0, 0.0]), PotentialType::Quadratic, -1.0), 0.5);
```

### Example 3: Find Semantic Equilibrium

```rust
use semantic_field::potential::*;

let mut field = Superposition::new();
field.add_field(PotentialField::new(
    EmbeddingPoint::new(vec![5.0]), PotentialType::Quadratic, 1.0), 1.0);

let start = EmbeddingPoint::new(vec![0.0]);
let minimum = field.find_minimum(&start, 0.01, 10000);
```

## Performance

| Operation | Complexity |
|-----------|-----------|
| Potential at point | O(F × d) |
| Gradient computation | O(F × d) |
| Find minimum | O(I × F × d) |

Where F = fields, d = dimensions, I = iterations.

## License

Licensed under the [MIT License](LICENSE).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests
4. Push and open a Pull Request
