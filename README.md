# StreamModel

A JAX-based implementation for modeling and fitting stellar streams in galactic potentials using GPU-accelerated nested sampling in Blackjax.

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
.
├── data.py                      # mock data function
├── jax_nestedsampling_colab     # Python copy of the colab notebook
├── loglikelihood.py             # log-likelihood function
├── main.py                      # Example script to run the inference
├── model.py                     # JAX stream model
├── prior.py                     # Functions to sample from prior distributions
├── README.md                    # Project overview
└── requirements.txt             # List of dependencies

```