from setuptools import setup, find_packages

setup(
    name="Tabular_to_Neo4j",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "langgraph>=0.0.10",
        "requests>=2.25.0",
        "python-dotenv>=0.19.0",
        "langdetect>=1.0.9",
        "langchain-openai>=0.0.1",
        "langchain-community>=0.0.1",
        "neo4j>=5.18",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=2.12.0",
            "pytest-mock>=3.6.0",
            "tox>=3.24.0",
            "flake8>=4.0.0",
            "black>=22.0.0",
        ],
    },
    python_requires=">=3.8",
)
