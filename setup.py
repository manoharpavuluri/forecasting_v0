"""
Setup file for the Solar Time Series Forecasting project
"""

from setuptools import setup, find_packages

setup(
    name="solar-forecast",
    version="0.1.0",
    description="Multi-agent AI system for solar time series forecasting",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "polars>=0.20.0",
        "pyspark>=3.5.0",
        "duckdb>=0.9.0",
        "prophet>=1.1.5",
        "xgboost>=2.0.0",
        "scikit-learn>=1.3.0",
        "requests>=2.31.0",
        "httpx>=0.25.0",
        "ftplib3>=0.2.0",
        "streamlit>=1.30.0",
        "plotly>=5.18.0",
        "seaborn>=0.13.0",
        "langchain>=0.1.0",
        "openai>=1.3.0",
        "faiss-cpu>=1.7.4",
        "chromadb>=0.4.0",
        "langgraph>=0.0.15",
        "apache-airflow>=2.7.0",
        "mlflow>=2.8.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ]
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
) 