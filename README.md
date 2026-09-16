# fl_vs_cl_ids_unseen

Code for *Empirical Comparison of Centralized and Federated Learning for Intrusion Detection on Unseen Domains* - comparing a centralized VAE-based anomaly detector against 4 federated methods (FedAvg, FedAdam, FedProx, SCAFFOLD) under a strict unseen-domain evaluation protocol across three NetFlow datasets.

## Repository Structure

```
main/
├── README.md                     # overview and setup
├── requirements.txt              # Python dependencies
├── data/
│   ├── splits/                   # pkl files for each dataset's train, val, and test split
│   └── preprocessing/            # loads and creates pkl files
├── centralized/                  # centralized-learning pipeline
├── federated/                    # federated-learning pipeline - FedAvg, FedAdam, FedProx, and SCAFFOLD
├── hpo/                          # Optuna hyperparameter search (80 trials per method / domain combination)
├── models/                       # VAE anomaly-detector architecture
├── task/
│   └── base.py                   # base abstract class representing each client's task
└── out/                          # RESULTS
    ├── federated_multiseed/      # per-seed federated results 
    └── centralized_multiseed/    # per-seed centralized results 
```
