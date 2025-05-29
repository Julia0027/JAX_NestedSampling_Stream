# StreamModel

A JAX-based implementation for modeling and fitting stellar streams in galactic potentials using GPU-accelerated nested sampling in BlackJAX.

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
.
├── model.py               # JAX stream model and log-likelihood function
├── priors.py              # Functions to sample from prior distributions
├── utils.py               # Helper functions for analysis or plotting
├── run.py                 # Example script to run the model
├── requirements.txt       # List of dependencies
└── README.md              # Project overview
```