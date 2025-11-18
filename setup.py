from setuptools import setup, find_packages

setup(
    name="channelpy",
    version="0.1.0",
    description="Channel Algebra for Structured Data Analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Channel Algebra Team",
    author_email="contact@channelalgebra.org",
    url="https://github.com/channelalgebra/channelpy",
    packages=find_packages(exclude=["tests", "docs"]),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
        "viz": [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "applications": [
            "yfinance>=0.1.60",  # For trading examples
            "ta>=0.7.0",  # Technical analysis
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)