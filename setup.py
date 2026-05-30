from setuptools import setup, find_packages

setup(
    name="ecomeal-ai",
    version="1.0.0",
    description="AI-powered food waste prediction and recommendation system for restaurants",
    author="EcoMeal AI Team",
    author_email="team@ecomeal.ai",
    python_requires=">=3.11",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0",
        "lightgbm>=4.0.0",
        "anthropic>=0.39.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.4.0",
        "statsmodels>=0.14.0",
        "shap>=0.42.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ecomeal-api=src.api.main:start",
            "ecomeal-train=scripts.train:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
