# GitHub Repository Structure

knowledge-xtraction/
├── README.md
├── docs/
│   ├── survey/              # Survey paper materials
│   ├── presentations/       # Proposal and other presentations
│   └── experiments/         # Experiment documentation
├── src/
│   ├── extraction/          # Knowledge extraction components
│   │   ├── entity_extraction.py
│   │   ├── relation_extraction.py
│   │   └── rau_enhancement.py
│   ├── graph_construction/  # KG construction modules
│   │   ├── kg_builder.py
│   │   └── kg_storage.py
│   ├── embeddings/          # Graph Kernel Mean Embedding
│   │   ├── graph_kernel.py
│   │   ├── mean_embedding.py
│   │   └── edge_importance.py
│   ├── uncertainty/         # Uncertainty quantification
│   │   ├── triple_uncertainty.py
│   │   ├── path_uncertainty.py
│   │   └── calibration.py
│   └── evaluation/          # Evaluation metrics
│       ├── extraction_metrics.py
│       ├── graph_metrics.py
│       └── uncertainty_metrics.py
├── data/
│   ├── raw/                 # Original datasets
│   ├── processed/           # Processed knowledge graphs
│   └── embeddings/          # Generated graph embeddings
├── notebooks/               # Exploratory analysis
└── website/                 # Project webpage files
    ├── index.html
    ├── visualization/
    └── publications/