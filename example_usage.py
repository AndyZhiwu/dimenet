"""
Example usage of the PyTorch DimeNet/DimeNet++ implementation.
This script demonstrates how to instantiate and use the models.
"""

import torch
from dimenet.model.dimenet_pp import DimeNetPP
from dimenet.model.dimenet import DimeNet
from dimenet.model.activations import swish


def example_dimenet_pp():
    """Example of using DimeNet++ model."""
    print("=" * 60)
    print("DimeNet++ Example")
    print("=" * 60)
    
    # Create model instance
    model = DimeNetPP(
        emb_size=128,
        out_emb_size=256,
        int_emb_size=64,
        basis_emb_size=8,
        num_blocks=4,
        num_spherical=7,
        num_radial=6,
        cutoff=5.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_dense_output=3,
        num_targets=1,
        activation=swish,
        extensive=True,
        output_init='zeros'
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Example input (you would prepare this from your molecular data)
    # These are dummy values for demonstration
    n_atoms = 10
    n_edges = 20
    n_triplets = 40
    
    inputs = {
        'Z': torch.randint(1, 10, (n_atoms,), dtype=torch.int32),  # Atomic numbers
        'R': torch.randn(n_atoms, 3),  # Atomic coordinates
        'batch_seg': torch.zeros(n_atoms, dtype=torch.long),  # Batch assignment
        'idnb_i': torch.randint(0, n_atoms, (n_edges,), dtype=torch.long),  # Edge source
        'idnb_j': torch.randint(0, n_atoms, (n_edges,), dtype=torch.long),  # Edge target
        'id_expand_kj': torch.randint(0, n_edges, (n_triplets,), dtype=torch.long),
        'id_reduce_ji': torch.randint(0, n_edges, (n_triplets,), dtype=torch.long),
        'id3dnb_i': torch.randint(0, n_atoms, (n_triplets,), dtype=torch.long),
        'id3dnb_j': torch.randint(0, n_atoms, (n_triplets,), dtype=torch.long),
        'id3dnb_k': torch.randint(0, n_atoms, (n_triplets,), dtype=torch.long),
    }
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(inputs)
    
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    print()


def example_dimenet():
    """Example of using DimeNet model."""
    print("=" * 60)
    print("DimeNet Example")
    print("=" * 60)
    
    # Create model instance
    model = DimeNet(
        emb_size=128,
        num_blocks=6,
        num_bilinear=8,
        num_spherical=7,
        num_radial=6,
        cutoff=5.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_dense_output=3,
        num_targets=1,
        activation=swish,
        output_init='zeros'
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Example input (same format as DimeNet++)
    n_atoms = 10
    n_edges = 20
    n_triplets = 40
    
    inputs = {
        'Z': torch.randint(1, 10, (n_atoms,), dtype=torch.int32),
        'R': torch.randn(n_atoms, 3),
        'batch_seg': torch.zeros(n_atoms, dtype=torch.long),
        'idnb_i': torch.randint(0, n_atoms, (n_edges,), dtype=torch.long),
        'idnb_j': torch.randint(0, n_atoms, (n_edges,), dtype=torch.long),
        'id_expand_kj': torch.randint(0, n_edges, (n_triplets,), dtype=torch.long),
        'id_reduce_ji': torch.randint(0, n_edges, (n_triplets,), dtype=torch.long),
        'id3dnb_i': torch.randint(0, n_atoms, (n_triplets,), dtype=torch.long),
        'id3dnb_j': torch.randint(0, n_atoms, (n_triplets,), dtype=torch.long),
        'id3dnb_k': torch.randint(0, n_atoms, (n_triplets,), dtype=torch.long),
    }
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(inputs)
    
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    print()


if __name__ == "__main__":
    print("\nPyTorch DimeNet Implementation Examples\n")
    
    # Run examples
    example_dimenet_pp()
    example_dimenet()
    
    print("Examples completed successfully!")
